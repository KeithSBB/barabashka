"""Config flow for Barabashka (formerly Grokzilla)."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import selector, llm
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
            return self.async_update_reload_and_abort(entry, 
                                                      data_updates=user_input)

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

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        base_schema = await self._async_get_options_schema(self.hass)
        schema_with_suggested = self.add_suggested_values_to_schema(
            base_schema, self.config_entry.options
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema_with_suggested,
            description_placeholders={"title": self.config_entry.title},
        )

    async def _async_get_options_schema(self, hass: HomeAssistant) -> vol.Schema:
        """Build the complete options schema including Barabashka features."""
        # Canonical way to get registered LLM APIs → list[llm.API] with .name and .id
        apis = llm.async_get_apis(hass)
        api_options: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id) for api in apis
        ]

        if not api_options:
            _LOGGER.warning("No LLM APIs registered yet – selector will be empty until reload")

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
                        options=api_options,
                        multiple=True,
                        mode=SelectSelectorMode.LIST,
                    )
                ),

                # --- Barabashka supernatural controls ---
                vol.Optional(
                    CONF_BARABASHKA_ENABLED,
                    default=True,
                    description={"suggested_value": True},  # Optional: helps pre-check if saved
                ): selector.BooleanSelector(),

                vol.Optional(
                    CONF_SENSOR_ENTITIES,
                    description={
                        "suggested_value": None,  # Placeholder – actual entities pre-filled via add_suggested_values
                        "description": "Choose specific sensors for Barabashka to listen to. Leave empty to auto-discover all ambient sensors."
                    },
                ): entity_selector,

                vol.Optional(
                    CONF_SENSITIVITY,
                    default=5,
                    description={
                        "description": "How sensitive Barabashka is to subtle changes (1 = calm, 10 = highly attuned)"
                    },
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
                    description={
                        "description": "Start of the witching hours when the veil is thinnest"
                    },
                ): TimeSelector(),

                vol.Optional(
                    CONF_SPIRIT_HOURS_END,
                    default="04:00:00",
                    description={
                        "description": "End of the witching hours"
                    },
                ): TimeSelector(),
            }
        )