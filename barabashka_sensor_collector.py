"""Barabashka Sensor Collector – Full Updated Version.

This version fully integrates with the new Barabashka options from config_flow.py:
- Respects barabashka_enabled toggle
- Uses user-selected sensor_entities or auto-discovers ambient sensors
- Applies configurable sensitivity (1–10) to scale pattern detection thresholds
- Boosts pattern strength during user-defined "spirit hours" (default 2–4 AM)
- Persists learned pattern weights
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, List

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.entity_registry import async_get as async_get_er
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BARABASHKA_ENABLED,
    CONF_SENSOR_ENTITIES,
    CONF_SENSITIVITY,
    CONF_SPIRIT_HOURS_END,
    CONF_SPIRIT_HOURS_START,
)

_LOGGER = logging.getLogger(__name__)

# Default ambient device classes for auto-discovery
DEFAULT_AMBIENT_CLASSES = {
    "temperature",
    "humidity",
    "pressure",
    "illuminance",
    "sound",
    "power",
    "voltage",
    "current",
    "signal_strength",
}

# Rolling history retention
HISTORY_HOURS = 48

# How often to recompute the spirit message
MESSAGE_UPDATE_INTERVAL = 300  # 5 minutes

# Storage for learned pattern weights
STORAGE_KEY = "barabashka_learning"
STORAGE_VERSION = 1


@dataclass
class SensorReading:
    """Single sensor reading."""
    timestamp: datetime
    value: float | str | None
    unit: str | None = None


@dataclass
class SensorHistory:
    """Rolling history per entity."""
    readings: deque[SensorReading] = field(default_factory=deque)
    entity_id: str = ""
    device_class: str | None = None
    name: str = ""

    def append(self, reading: SensorReading) -> None:
        self.readings.append(reading)
        cutoff = dt_util.utcnow() - timedelta(hours=HISTORY_HOURS)
        while self.readings and self.readings[0].timestamp < cutoff:
            self.readings.popleft()


@dataclass
class PatternWeights:
    """Learned interpretation weights."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "temp_drop_night": 0.7,
        "sudden_light_change": 0.5,
        "silence_3am": 0.8,
        "emf_spike": 0.6,
        "motion_without_cause": 0.65,
        "humidity_rise_night": 0.55,
        "pressure_drop": 0.6,
    })


class BarabashkaSensorCollector:
    """Collects house ambient data and speaks as the spirit Barabashka."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        _LOGGER.info("Initializing BarabashkaSensorCollector")
        self.hass = hass
        self.entry = entry
        self.entry_id = entry.entry_id

        # Configurable options
        self.enabled = entry.options.get(CONF_BARABASHKA_ENABLED, True)
        self.sensitivity = float(entry.options.get(CONF_SENSITIVITY, 5)) / 5.0  # 0.2–2.0 multiplier
        self.spirit_start_str = entry.options.get(CONF_SPIRIT_HOURS_START, "02:00:00")
        self.spirit_end_str = entry.options.get(CONF_SPIRIT_HOURS_END, "04:00:00")

        # Convert spirit hours to time objects
        try:
            self.spirit_start = datetime.strptime(self.spirit_start_str, "%H:%M:%S").time()
            self.spirit_end = datetime.strptime(self.spirit_end_str, "%H:%M:%S").time()
        except ValueError:
            self.spirit_start = time(2, 0, 0)
            self.spirit_end = time(4, 0, 0)

        # Runtime state
        self.histories: Dict[str, SensorHistory] = {}
        self._unsub_trackers: List[callback] = []
        self._current_message: str = "The house is calm and quiet. Barabashka rests in silence."
        self._last_message_update: datetime = dt_util.utcnow()

        # Persistent learning
        self.store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self.weights = PatternWeights()
        
        # New runtime diagnostics for monitoring
        self.anomaly_log: deque[tuple[datetime, str, float | None, float]] = deque(maxlen=100)
        self.recent_messages: deque[str] = deque(maxlen=50)  # Store last spoken messages
        self.sample_count: int = 0
        self.last_patterns: Dict[str, float] = {}  # For detecting weight changes

    async def async_start(self) -> None:
        """Start the collector if Barabashka mode is enabled."""
        if not self.enabled:
            _LOGGER.info("Barabashka mode disabled in options – collector not starting.")
            return

        await self._async_load_weights()
        await self._discover_and_subscribe_sensors()

        # Periodic spirit message updates
        self.hass.loop.create_task(self._message_update_loop())

        _LOGGER.info(
            "Barabashka sensor collector started (sensitivity: %.1fx, spirit hours: %s–%s)",
            self.sensitivity,
            self.spirit_start_str,
            self.spirit_end_str,
        )

    async def async_stop(self) -> None:
        """Unload: stop all trackers."""
        _LOGGER.info("Stop all trackers")
        for unsub in self._unsub_trackers:
            unsub()
        self._unsub_trackers.clear()
        await self.async_save_weights()

    async def _discover_and_subscribe_sensors(self) -> None:
        """Subscribe to user-selected sensors or auto-discover ambient ones."""
        entity_reg = async_get_er(self.hass)

        configured_entities: List[str] = self.entry.options.get(CONF_SENSOR_ENTITIES, [])

        if configured_entities:
            entities_to_watch = [e for e in configured_entities if e in entity_reg.entities]
            _LOGGER.info("Barabashka manually configured to watch %d sensors", len(entities_to_watch))
        else:
            # Auto-discovery
            entities_to_watch = [
                ent.entity_id
                for ent in entity_reg.entities.values()
                if ent.device_class in DEFAULT_AMBIENT_CLASSES
                and not ent.disabled_by
                and ent.domain in ("sensor", "binary_sensor")
            ]
            _LOGGER.info("Barabashka auto-discovered %d ambient sensors", len(entities_to_watch))

        for entity_id in entities_to_watch:
            await self._subscribe_entity(entity_id)

    async def _subscribe_entity(self, entity_id: str) -> None:
        """Subscribe to one entity and seed history."""
        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug("Entity %s unavailable, skipping subscription", entity_id)
            return

        history = SensorHistory(
            entity_id=entity_id,
            device_class=state.attributes.get("device_class"),
            name=state.attributes.get("friendly_name") or entity_id,
        )

        # Seed with current value
        try:
            value = float(state.state) if state.state not in ("unknown", "unavailable") else None
        except (ValueError, TypeError):
            value = state.state

        if value is not None:
            history.append(
                SensorReading(timestamp=dt_util.utcnow(), value=value, unit=state.attributes.get("unit_of_measurement"))
            )

        self.histories[entity_id] = history

        unsub = async_track_state_change_event(self.hass, [entity_id], self._handle_state_change)
        self._unsub_trackers.append(unsub)

    @callback
    def _handle_state_change(self, event: Event) -> None:
        """Append new reading."""
        entity_id = event.data["entity_id"]
        new_state = event.data["new_state"]
        if new_state is None:
            return

        try:
            value = float(new_state.state) if new_state.state not in ("unknown", "unavailable") else None
        except (ValueError, TypeError):
            value = new_state.state

        reading = SensorReading(
            timestamp=dt_util.utcnow(),
            value=value,
            unit=new_state.attributes.get("unit_of_measurement"),
        )

        if entity_id in self.histories:
            self.histories[entity_id].append(reading)

    async def _message_update_loop(self) -> None:
        """Background task: recompute spirit message periodically."""
        while True:
            await asyncio.sleep(MESSAGE_UPDATE_INTERVAL)
            await self._update_spirit_message()
            _LOGGER.info("Spirit Message updated")

    async def async_update_anomalies(self) -> None:
        """public wrapper for self._update_spirit_message()"""
        await self._update_spirit_message()

    async def _update_spirit_message(self) -> None:
        """Generate new message from current patterns and log for monitoring."""
        now = dt_util.utcnow()
        if now - self._last_message_update < timedelta(minutes=4):
            return  # debounce

        patterns = self._detect_patterns()

        # Log significant patterns as "anomalies" with deviation score
        total_score = sum(patterns.values())
        for pattern, strength in patterns.items():
            if strength > 0.3:  # Only log meaningful signals
                self.anomaly_log.append((
                    now,
                    pattern,
                    strength,
                    strength  # using strength as "deviation" for z-like scoring
                ))

        # Update sample count
        self.sample_count += 1

        message = self._interpret_patterns(patterns)
        if message != self._current_message:
            self._current_message = message
            self.recent_messages.append(f"{now.isoformat()}: {message}")
            _LOGGER.info("Barabashka speaks: %s", message)

            # Optional: boost learning on strong signals
            if total_score > 1.0:
                strongest = max(patterns, key=patterns.get)
                self.weights.weights[strongest] = min(
                    1.5, self.weights.weights.get(strongest, 0.7) + 0.05 * self.sensitivity
                )
                await self.async_save_weights()

        self.last_patterns = patterns
        self._last_message_update = now

    def _is_spirit_hour(self) -> bool:
        """Check if current time falls within configured spirit hours."""
        now = dt_util.utcnow().time()
        if self.spirit_start <= self.spirit_end:
            return self.spirit_start <= now <= self.spirit_end
        else:  # crosses midnight
            return now >= self.spirit_start or now <= self.spirit_end
            
    def current_anomaly_score(self) -> float:
        """Current overall supernatural activity level."""
        if not self.last_patterns:
            return 0.0
        return sum(self.last_patterns.values())

    def generate_learning_notes(self) -> str:
        """Human-readable notes on what Barabashka is learning."""
        if not self.last_patterns:
            return "No patterns detected yet. Listening to the house..."

        top_patterns = sorted(self.last_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        notes = []
        for pattern, strength in top_patterns:
            current_weight = self.weights.weights.get(pattern, 0.7)
            notes.append(f"{pattern.replace('_', ' ').title()} (strength: {strength:.2f}, weight: {current_weight:.2f})")

        if any(strength > 0.8 for strength in self.last_patterns.values()):
            notes.append("Strong signals detected – Barabashka is active tonight.")

        return "; ".join(notes)

    def _detect_patterns(self) -> Dict[str, float]:
        """Detect supernatural patterns with sensitivity and spirit-hour boost."""
        now = dt_util.utcnow()
        hour = now.hour
        is_night = 22 <= hour or hour <= 6
        is_spirit_hour = self._is_spirit_hour()

        patterns: Dict[str, float] = defaultdict(float)
        spirit_boost = 1.5 if is_spirit_hour else 1.0

        for history in self.histories.values():
            if len(history.readings) < 2:
                continue

            recent = list(history.readings)[-10:]
            values = [r.value for r in recent if isinstance(r.value, (int, float))]
            if not values:
                continue

            base_threshold = 1.0 / max(self.sensitivity, 0.5)  # higher sensitivity → lower threshold

            # Temperature drop at night
            if history.device_class == "temperature" and is_night:
                drop = values[0] - values[-1]
                if drop > 1.0 * base_threshold:
                    patterns["temp_drop_night"] += drop * 0.4 * self.sensitivity * spirit_boost

            # Sudden light change
            if history.device_class == "illuminance":
                change = abs(values[-1] - values[0]) / (max(values) + 1)
                if change > 0.3 * base_threshold:
                    patterns["sudden_light_change"] += change * self.sensitivity * spirit_boost

            # Silence during spirit hours (sound sensor)
            if history.device_class == "sound" and is_spirit_hour:
                if values[-1] < 35:  # very quiet
                    patterns["silence_3am"] += (1.0 - values[-1] / 60) * self.sensitivity * 1.2

            # Pressure drop (storm coming or veil thinning)
            if history.device_class == "pressure":
                if values[0] - values[-1] > 2.0 * base_threshold:
                    patterns["pressure_drop"] += 0.7 * self.sensitivity * spirit_boost

            # Humidity rise at night
            if history.device_class == "humidity" and is_night:
                rise = values[-1] - values[0]
                if rise > 5.0 * base_threshold:
                    patterns["humidity_rise_night"] += rise * 0.1 * self.sensitivity * spirit_boost

        # Apply learned weights
        for key in patterns:
            patterns[key] = min(1.5, patterns[key] * self.weights.weights.get(key, 0.7))

        return patterns

    def _interpret_patterns(self, patterns: Dict[str, float]) -> str:
        """Translate strongest pattern into Barabashka's voice."""
        if not patterns:
            return "The house breathes slowly. Barabashka rests in the walls, watching."

        strongest = max(patterns, key=patterns.get)

        messages = {
            "temp_drop_night": (
                "A sudden chill sweeps through the rooms. Barabashka stirs uneasily, "
                "drawing close to the hearth that is no longer lit."
            ),
            "sudden_light_change": (
                "Lights flicker of their own accord. The house spirit plays with electricity — "
                "or warns that something crosses the threshold."
            ),
            "silence_3am": (
                "In the deepest hour, all sound vanishes. Barabashka listens at the edge of the veil. "
                "The night holds its breath."
            ),
            "pressure_drop": (
                "The air thickens and presses down. Barabashka feels the weight of worlds brushing against this one."
            ),
            "humidity_rise_night": (
                "Dampness rises from nowhere. The house spirit remembers old rivers that once flowed beneath the foundations."
            ),
        }

        return messages.get(strongest, "Barabashka murmurs softly. Something stirs in the quiet spaces between things.")

    def get_current_spirit_message(self) -> str:
        """Public accessor for the conversation agent."""
        return self._current_message

    async def _async_load_weights(self) -> None:
        """Load learned weights from storage."""
        data = await self.store.async_load()
        if data and "weights" in data:
            self.weights.weights.update(data["weights"])
            _LOGGER.debug("Loaded Barabashka learned pattern weights")

    async def async_save_weights(self) -> None:
        """Persist current weights."""
        await self.store.async_save({"weights": self.weights.weights})
        
    def get_monitor_data(self) -> dict:
        """Compile data for the Barabashka Spirit Monitor entity."""
        # Derive mood from recent activity (folklore flavor)
        recent_scores = [dev for _, _, _, dev in list(self.anomaly_log)[-5:]]
        recent_score = sum(recent_scores)
        
        if recent_score > 8:
            mood = "restless"      # Poltergeist activity rising
        elif recent_score > 4:
            mood = "playful"       # Mischievous knocking and flickers
        elif recent_score == 0 and self.sample_count > 10:
            mood = "content"       # Peaceful domovoi, household in harmony
        else:
            mood = "listening"     # Waiting at the threshold

        return {
            "mood": mood,
            "weights": dict(self.weights.weights),  # Convert to plain dict for JSON
            "recent_anomalies": [
                {
                    "time": t.isoformat(),
                    "pattern": s,
                    "strength": float(v) if v is not None else None,
                    "score": float(d),
                }
                for t, s, v, d in list(self.anomaly_log)[-10:]
            ],
            "decoded_messages": list(self.recent_messages)[-20:],
            "total_samples": self.sample_count,
            "current_score": round(self.current_anomaly_score(), 3),
            "last_anomaly": self.anomaly_log[-1][0].isoformat() if self.anomaly_log else None,
            "notes": self.generate_learning_notes(),
        }
