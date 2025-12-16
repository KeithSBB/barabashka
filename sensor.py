"""Barabashka Spirit Monitor sensor."""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Final

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, CoordinatorEntity

from .const import DOMAIN, BARABASHKA_COLLECTOR
from .barabashka_sensor_collector import BarabashkaSensorCollector  # Import your collector class

_LOGGER = logging.getLogger(__name__)

UPDATE_INTERVAL: Final = timedelta(minutes=5)  # How often to refresh monitor


async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities):
    """Set up the Barabashka Spirit Monitor sensor."""
    collector: BarabashkaSensorCollector = hass.data[DOMAIN][entry.entry_id][BARABASHKA_COLLECTOR]

    async def async_update_data():
        """Fetch latest spirit data."""
        await collector.async_update_anomalies()  # Trigger fresh processing if needed
        return collector.get_monitor_data()

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name="barabashka_spirit_monitor",
        update_method=async_update_data,
        update_interval=UPDATE_INTERVAL,
    )

    # Initial load
    await coordinator.async_config_entry_first_refresh()

    async_add_entities([BarabashkaSpiritMonitor(coordinator, entry)])


class BarabashkaSpiritMonitor(CoordinatorEntity, SensorEntity):
    """Representation of the Barabashka Spirit Monitor sensor."""

    _attr_icon = "mdi:ghost-outline"
    _attr_name = "Barabashka Spirit Monitor"
    _attr_unique_id = "barabashka_spirit_monitor"

    def __init__(self, coordinator: DataUpdateCoordinator, entry):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Barabashka",
            "manufacturer": "xAI + Home Assistant",
            "model": "House Spirit Interface",
        }

    @property
    def native_value(self) -> str:
        """Return the state: current mood of the Barabashka."""
        return self.coordinator.data.get("mood", "listening")

    @property
    def extra_state_attributes(self) -> dict:
        """Return detailed diagnostic attributes."""
        data = self.coordinator.data or {}
        return {
            "sensor_weights": data.get("weights", {}),  # {entity_id: weight}
            "recent_anomalies": data.get("recent_anomalies", []),
            "decoded_messages": data.get("decoded_messages", []),
            "total_samples": data.get("total_samples", 0),
            "anomaly_score": data.get("current_score", 0.0),
            "last_anomaly_time": data.get("last_anomaly"),
            "learning_notes": data.get("notes", "No patterns detected yet."),
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()