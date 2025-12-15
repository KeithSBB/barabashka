"""The Grok xAI Conversation integration."""

import logging
from typing import Any, Dict

import xai_sdk
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, BARABASHKA_COLLECTOR
from .llm_api import GrokCustomLLMApi

from .barabashka_sensor_collector import BarabashkaSensorCollector

_LOGGER = logging.getLogger(__name__)
PLATFORMS: list[Platform] = [Platform.CONVERSATION]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grokzilla from a config entry."""
    
    # 1. Register Custom LLM API
    try:
        await GrokCustomLLMApi.async_setup_api(hass, entry)
    except Exception as err:
        raise ConfigEntryNotReady(f"Failed to set up LLM API: {err}") from err

    # 2. Initialize Data
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {}
    collector = BarabashkaSensorCollector(hass, entry)
    hass.data[DOMAIN][entry.entry_id][BARABASHKA_COLLECTOR] = collector
    await collector.async_start()

    # Store cleanup for unload
    hass.data[DOMAIN][entry.entry_id]["cleanup_unsub"] = collector.async_stop

    # 3. Set up Platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    
    # 1. Unload Platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # 2. Clean up Data
        data = hass.data[DOMAIN].pop(entry.entry_id, None)
        
        if data and "cleanup_unsub" in data:
            data["cleanup_unsub"]()

        # 3. Remove Domain if empty
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)

    return unload_ok

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
