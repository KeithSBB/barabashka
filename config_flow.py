"""Config flow for Barabashka (formerly Grokzilla)."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import selector
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    TimeSelector,
)

from .const import (
    CONF_BARABASHKA_ENABLED,
    CONF_SENSOR_ENTITIES,
    CONF_SENSITIVITY,
    CONF_SPIRIT_HOURS_END,
    CONF_SPIRIT_HOURS_START,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class GrokConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Barabashka integration."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step – API key entry."""
        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title="Barabashka", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle reconfiguration (e.g., change API key)."""
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            return self.async_update_reload_and_abort(entry, data_updates=user_input)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_API_KEY): TextSelector(
                            TextSelectorConfig(type=TextSelectorType.PASSWORD)
                        ),
                    }
                ),
                entry.data,
            ),
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the Barabashka options flow handler."""
        return BarabashkaOptionsFlow()


class BarabashkaOptionsFlow(config_entries.OptionsFlow):
    """Options flow with full Barabashka controls."""
    
    # def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
    #     """Initialize options flow."""
    #     super().__init__(config_entry)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=await self._async_get_options_schema(self.hass),
            description_placeholders={"title": self.config_entry.title},
        )

    async def _async_get_options_schema(self, hass: HomeAssistant) -> vol.Schema:
        """Build the complete options schema including Barabashka features."""
        # Load available LLM APIs for the original selector
        try:
            apis: list[SelectOptionDict] = [
                SelectOptionDict(label=api.name, value=api.id)
                for api in hass.data.get("llm", {}).get("apis", [])
            ]
        except Exception as err:
            _LOGGER.warning("Failed to load LLM APIs for options: %s", err)
            apis = []

        # Entity selector limited to ambient sensors
        entity_selector = EntitySelector(
            EntitySelectorConfig(
                domain=["sensor", "binary_sensor"],
                device_class=[
                    "temperature",
                    "humidity",
                    "pressure",
                    "illuminance",
                    "sound",
                    "power",
                    "voltage",
                    "current",
                    "signal_strength",
                ],
                multiple=True,
            )
        )

        return vol.Schema(
            {
                # --- Original Grokzilla options ---
                vol.Optional("model", default=DEFAULT_MODEL): str,
                vol.Optional("prompt", default=DEFAULT_PROMPT): TextSelector(
                    TextSelectorConfig(multiline=True, type=TextSelectorType.TEXT)
                ),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(
                        options=apis,
                        multiple=True,
                        mode=SelectSelectorMode.LIST,
                    )
                ),

                # --- Barabashka supernatural controls ---
                vol.Optional(CONF_BARABASHKA_ENABLED, default=True): selector.BooleanSelector(),

                vol.Optional(
                    CONF_SENSOR_ENTITIES,
                    description=(
                        "Choose specific sensors for Barabashka to listen to. "
                        "Leave empty to auto-discover all ambient sensors."
                    ),
                ): entity_selector,

                vol.Optional(
                    CONF_SENSITIVITY,
                    default=5,
                    description="How sensitive Barabashka is to subtle changes (1 = calm, 10 = highly attuned)",
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=10,
                        step=1,
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),

                vol.Optional(
                    CONF_SPIRIT_HOURS_START,
                    default="02:00:00",
                    description="Start of the witching hours when the veil is thinnest",
                ): TimeSelector(),

                vol.Optional(
                    CONF_SPIRIT_HOURS_END,
                    default="04:00:00",
                    description="End of the witching hours",
                ): TimeSelector(),
            }
        )
