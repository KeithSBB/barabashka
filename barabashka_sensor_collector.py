"""Barabashka Sensor Collector.

This module is the heart of the Barabashka integration.
It continuously monitors selected ambient sensors, maintains a rolling history,
detects anomalies and patterns, and translates them into a textual "message"
that represents the current voice/mood of the house spirit (Barabashka).

The generated message is then injected as a system prompt into every Grok
conversation so the spirit can influence responses.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.entity_registry import async_get as async_get_er
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Default sensor device classes we consider "ambient" and potentially supernatural
DEFAULT_AMBIENT_CLASSES = {
    "temperature",
    "humidity",
    "pressure",
    "illuminance",
    "sound",
    "power",  # sudden draws
    "voltage",
    "current",
    "signal_strength",  # Wi-Fi/Zigbee fluctuations
}

# How long we keep raw history in memory (hours)
HISTORY_HOURS = 48

# How often we recompute the spirit message (seconds)
MESSAGE_UPDATE_INTERVAL = 300  # 5 minutes

# Storage key for persistent weights / learned patterns
STORAGE_KEY = "barabashka_learning"
STORAGE_VERSION = 1


@dataclass
class SensorReading:
    """Single sensor reading with timestamp."""
    timestamp: datetime
    value: float | str | None
    unit: str | None = None


@dataclass
class SensorHistory:
    """Rolling history for one entity."""
    readings: deque[SensorReading] = field(default_factory=deque)
    entity_id: str = ""
    device_class: str | None = None
    name: str = ""

    def append(self, reading: SensorReading) -> None:
        self.readings.append(reading)
        # Prune old readings
        cutoff = dt_util.utcnow() - timedelta(hours=HISTORY_HOURS)
        while self.readings and self.readings[0].timestamp < cutoff:
            self.readings.popleft()


@dataclass
class PatternWeights:
    """Learned weights for interpreting patterns."""
    # Example keys: "temp_drop_night", "emf_spike", "silence_3am"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "temp_drop_night": 0.7,
        "sudden_light_change": 0.5,
        "silence_3am": 0.8,
        "emf_spike": 0.6,
        "motion_without_cause": 0.65,
    })


class BarabashkaSensorCollector:
    """Collects ambient sensor data and speaks as the house spirit."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.entry_id = entry.entry_id

        # Runtime data
        self.histories: Dict[str, SensorHistory] = {}
        self._unsub_trackers: List[callback] = []
        self._current_message: str = "The house is calm and quiet. Barabashka rests."
        self._last_message_update: datetime = dt_util.utcnow()

        # Persistent learning
        self.store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self.weights = PatternWeights()

    async def async_start(self) -> None:
        """Start collecting data and generating spirit messages."""
        await self._async_load_weights()
        await self._discover_and_subscribe_sensors()

        # Periodic message recomputation
        self.hass.loop.create_task(self._message_update_loop())

        _LOGGER.info("Barabashka sensor collector started for entry %s", self.entry_id)

    async def async_stop(self) -> None:
        """Stop all listeners."""
        for unsub in self._unsub_trackers:
            unsub()
        self._unsub_trackers.clear()

    async def _discover_and_subscribe_sensors(self) -> None:
        """Find ambient sensors and subscribe to state changes."""
        entity_reg = async_get_er(self.hass)

        # Get user-configured entity_ids from options (fallback to auto-discovery)
        configured_entities: List[str] = self.entry.options.get("sensor_entities", [])

        if configured_entities:
            entities_to_watch = configured_entities
        else:
            # Auto-discovery: any sensor with relevant device_class
            entities_to_watch = [
                entity.entry.entity_id
                for entity in entity_reg.entities.values()
                if entity.device_class in DEFAULT_AMBIENT_CLASSES
                and not entity.disabled_by
            ]

        _LOGGER.debug("Barabashka watching %d sensors: %s", len(entities_to_watch), entities_to_watch)

        for entity_id in entities_to_watch:
            await self._subscribe_entity(entity_id)

    async def _subscribe_entity(self, entity_id: str) -> None:
        """Subscribe to one entity and initialize its history."""
        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Entity %s not available, skipping", entity_id)
            return

        # Initialize history
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
            history.append(SensorReading(
                timestamp=dt_util.utcnow(),
                value=value,
                unit=state.attributes.get("unit_of_measurement"),
            ))

        self.histories[entity_id] = history

        # Subscribe to future changes
        unsub = async_track_state_change_event(
            self.hass, [entity_id], self._handle_state_change
        )
        self._unsub_trackers.append(unsub)

    @callback
    def _handle_state_change(self, event: Event) -> None:
        """Handle state change and append to history."""
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
        else:
            # Rare race condition – re-subscribe
            asyncio.create_task(self._subscribe_entity(entity_id))

    async def _message_update_loop(self) -> None:
        """Periodically recompute the spirit's current message."""
        while True:
            await asyncio.sleep(MESSAGE_UPDATE_INTERVAL)
            await self._update_spirit_message()

    async def _update_spirit_message(self) -> None:
        """Analyze recent history and generate new Barabashka message."""
        now = dt_util.utcnow()
        if now - self._last_message_update < timedelta(minutes=4):
            return  # Debounce rapid calls

        patterns = self._detect_patterns()
        message = self._interpret_patterns(patterns)

        if message != self._current_message:
            self._current_message = message
            _LOGGER.debug("New Barabashka message: %s", message)

        self._last_message_update = now

    def _detect_patterns(self) -> Dict[str, float]:
        """Detect current supernatural patterns and return strength (0.0–1.0)."""
        now = dt_util.utcnow()
        hour = now.hour
        is_night = 22 <= hour or hour <= 6
        is_3am_window = 2 <= hour <= 4

        patterns: Dict[str, float] = defaultdict(float)

        for history in self.histories.values():
            if len(history.readings) < 2:
                continue

            recent = list(history.readings)[-10:]  # last 10 readings
            values = [r.value for r in recent if isinstance(r.value, (int, float))]

            if not values:
                continue

            # Temperature drop at night
            if history.device_class == "temperature" and is_night:
                if values[-1] < values[0] - 1.5:  # >1.5°C drop
                    patterns["temp_drop_night"] = min(1.0, patterns["temp_drop_night"] + 0.8)

            # Sudden light change
            if history.device_class == "illuminance":
                if abs(values[-1] - values[0]) > max(values) * 0.5:
                    patterns["sudden_light_change"] = 0.7

            # Silence at 3 AM (low sound level)
            if history.device_class == "sound" and is_3am_window:
                if values[-1] < 30:  # very quiet
                    patterns["silence_3am"] = 0.9

        # Boost learned weights
        for key, strength in patterns.items():
            patterns[key] = min(1.0, strength * self.weights.weights.get(key, 0.7))

        return patterns

    def _interpret_patterns(self, patterns: Dict[str, float]) -> str:
        """Turn detected patterns into poetic spirit speech."""
        if not patterns:
            return "The house sleeps deeply. Barabashka watches in silence."

        # Sort by strength
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_patterns[0][0]

        messages = {
            "temp_drop_night": (
                "A chill moves through the rooms. Barabashka stirs, restless in the dark. "
                "Something unseen passes close by."
            ),
            "sudden_light_change": (
                "Light flickers without cause. The house spirit plays with shadows, "
                "or warns of a presence just beyond sight."
            ),
            "silence_3am": (
                "At the witching hour, all sound falls away. Barabashka listens intently. "
                "The veil is thin tonight."
            ),
            "emf_spike": (
                "Invisible energy surges through the walls. Barabashka laughs softly – "
                "or growls at an intruder from the other side."
            ),
            "motion_without_cause": (
                "Footsteps where no feet walk. The house spirit follows you, curious or protective."
            ),
        }

        # Use strongest pattern, fall back to generic
        return messages.get(strongest, "Barabashka whispers faintly. The house remembers.")

    def get_current_spirit_message(self) -> str:
        """Return the latest message for injection into Grok chat."""
        return self._current_message

    async def _async_load_weights(self) -> None:
        """Load learned weights from storage."""
        data = await self.store.async_load()
        if data and "weights" in data:
            self.weights.weights.update(data["weights"])
            _LOGGER.debug("Loaded Barabashka learned weights")

    async def async_save_weights(self) -> None:
        """Save current weights to persistent storage."""
        await self.store.async_save({"weights": self.weights.weights})
