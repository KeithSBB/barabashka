"""Custom LLM API for Grok xAI Conversation."""

import fnmatch
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    label_registry as lr,
    llm,
    selector,
)
from homeassistant.util.json import JsonObjectType

from .const import MAX_SEARCH_RESULTS

_LOGGER = logging.getLogger(__name__)

def make_json_serializable(obj: Any) -> Any:
    """Recursively convert an object to be JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "as_dict"):
        return make_json_serializable(obj.as_dict())
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)

@dataclass(slots=True, kw_only=True)
class GrokCustomLLMApi(llm.API):
    """Expose custom Grok tools as an llm.API."""
    
    hass: HomeAssistant
    id: str
    name: str

    @staticmethod
    async def async_setup_api(hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Register the API with Home Assistant."""
        api = GrokCustomLLMApi(
            hass=hass, 
            id=f"grokzilla_unique_key-{entry.entry_id}", 
            name=entry.title
        )
        unreg = llm.async_register_api(hass, api)
        entry.async_on_unload(unreg)

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return the instance of the API."""
        return llm.APIInstance(
            api=self,
            api_prompt="Call the tools to fetch data from Home Assistant.",
            llm_context=llm_context,
            tools=[ GrokSearch(), 
                    GetEntStateAndAttrs(), 
                    CallService()
                   ]
        )

class GetEntStateAndAttrs(llm.Tool):
    """tool to get the current state of an entity"""
    
    name = "GetEntStateAndAttrs"
    description = "Returns the state of an entity defined by it entity id and any attributes"
    parameters = vol.Schema({
        vol.Required(
            "entity_id",
            description="The entity_id whos state is requested"):str
    })
    
    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
        ) -> JsonObjectType:
        """Call the tool."""
        args = tool_input.tool_args
        entity_id = args.get("entity_id") 
        state_obj = hass.states.get(entity_id)
        state = state_obj.state if state_obj else "unavailable"
        attributes = dict(state_obj.attributes) if state_obj else {}
        data = {
            "entity_id": entity_id,
            "state": state_obj.state,
            "attributes": make_json_serializable(dict(state_obj.attributes)),
            }
        #_LOGGER.debug("%s", data)
        return data


class GrokSearch(llm.Tool):
    """Tool to search for entities."""

    name = "GrokSearch"
    description = ("Search for entities by name. Optional add area, floor," 
                    " device or label to narrow search. ")

    parameters = vol.Schema({
        vol.Optional(
            "name",
            description="The name of a specific entity that you are searching for"
        ): str,
    
        vol.Optional(
            "entity_globs",
            description="Comma-separated list of entity ID glob patterns (e.g. light.kitchen_*, sensor.temperature*)"
        ): str,
    
        vol.Optional(
            "device_globs",
            description="Comma-separated device name or ID globs to filter by device"
        ): str,
    
        vol.Optional(
            "area_globs",
            description="Comma-separated area name globs (e.g. 'Kitchen', 'Living Room')"
        ): str,
    
        vol.Optional(
            "floor_globs",
            description="Comma-separated floor name globs (e.g. 'upper', 'main')"
        ): str,
    
        vol.Optional(
            "label_globs",
            description="Comma-separated label globs (e.g. 'presence', 'climate')"
        ): str,
    
        vol.Optional(
            "limit",
            default=MAX_SEARCH_RESULTS,
            description=f"Maximum number of results to return (1–500, default: {MAX_SEARCH_RESULTS})"
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=1, max=500, mode=selector.NumberSelectorMode.BOX)
        ),
    
        vol.Optional(
            "offset",
            default=0,
            description="Number of results to skip (for pagination)"
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, mode=selector.NumberSelectorMode.BOX)
        ),
    
        vol.Optional(
            "simple_mode",
            default=False,
            description="If true, searches only for name (faster, lighter response)"
        ): selector.BooleanSelector(),
    })

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        # Extract args safely
        args = tool_input.tool_args
        
        try:
            ent_reg = er.async_get(hass)
            area_reg = ar.async_get(hass)
            floor_reg = fr.async_get(hass)
            dev_reg = dr.async_get(hass)
            label_reg = lr.async_get(hass)

            matching = []
            
            # Prefetch registries for performance
            area_map = {a.id: a for a in area_reg.async_list_areas()}
            floor_map = {f.floor_id: f for f in floor_reg.async_list_floors()}
            dev_map = {d.id: d for d in dev_reg.devices.values()}
            label_map = {l.label_id: l for l in label_reg.async_list_labels()}

            # Helper to split comma-strings
            def parse_globs(val): 
                return [x.strip().lower() for x in (val or "").split(",") if x.strip()]

            name_query = (args.get("name") or "").lower()
            e_globs = parse_globs(args.get("entity_globs"))
            d_globs = parse_globs(args.get("device_globs"))
            a_globs = parse_globs(args.get("area_globs"))
            f_globs = parse_globs(args.get("floor_globs"))
            l_globs = parse_globs(args.get("label_globs"))
            simple_mode = args.get("simple_mode", False)

            for entry in ent_reg.entities.values():
                if entry.disabled_by:
                    continue
                
                entity_id = entry.entity_id
                state_obj = hass.states.get(entity_id)
                if not state_obj:
                    continue

                # Prepare metadata
                area = area_map.get(entry.area_id)
                device = dev_map.get(entry.device_id)
                
                # Fallback: Device area
                if not area and device and device.area_id:
                    area = area_map.get(device.area_id)

                area_name = area.name if area else ""
                floor_name = ""
                if area and area.floor_id:
                    floor = floor_map.get(area.floor_id)
                    if floor:
                        floor_name = floor.name

                device_name = device.name if device else ""
                entity_name = entry.name or entry.original_name or entity_id

                # -- Filtering Logic --

                # 1. Simple Mode
                if simple_mode and name_query:
                    search_text = f"{entity_name} {entity_id} {area_name}".lower()
                    # AND logic: all words must appear
                    if not all(w in search_text for w in name_query.split()):
                        continue

                # 2. Entity Globs
                if e_globs and not any(fnmatch.fnmatch(entity_id.lower(), g) for g in e_globs):
                    continue

                # 3. Area Globs
                if a_globs and not any(fnmatch.fnmatch(area_name.lower(), g) for g in a_globs):
                    continue

                # 4. Floor Globs
                if f_globs and not any(fnmatch.fnmatch(floor_name.lower(), g) for g in f_globs):
                    continue

                # 5. Device Globs
                if d_globs and not any(fnmatch.fnmatch(device_name.lower(), g) for g in d_globs):
                    continue
                
                # 6. Label Globs
                if l_globs:
                    ent_labels = {label_map[lid].name.lower() for lid in entry.labels if lid in label_map}
                    if not any(any(fnmatch.fnmatch(lbl, g) for lbl in ent_labels) for g in l_globs):
                        continue
                matching.append({
                    "entity_id": entity_id,
                    "name": entity_name,
                    "state": state_obj.state,
                    "area": area_name,
                    "floor": floor_name,
                    "device": device_name,
                    "attributes": make_json_serializable(dict(state_obj.attributes)),
                })

                if len(matching) >= args.get("limit", MAX_SEARCH_RESULTS) + args.get("offset", 0):
                    break

            # Apply offset/limit
            offset = args.get("offset", 0)
            limit = args.get("limit", MAX_SEARCH_RESULTS)
            result_slice = matching[offset : offset + limit]

            if not result_slice:
                return {"message": "No entities matched the criteria."}

            return result_slice

        except Exception as e:
            _LOGGER.exception("Search failed")
            return {"error": str(e)}
            
class CallService(llm.Tool):
    """Tool to call any Home Assistant service with optional target and data."""

    name = "CallService"
    description = (
        "Call any Home Assistant service (e.g., turn on a light, trigger an automation, "
        "set a thermostat temperature, send a notification, etc.). "
        "You must provide the full service name in the format 'domain.service_name'."
    )

    parameters = vol.Schema({
        vol.Required(
            "service",
            description="The service to call in the format 'domain.service_name' (e.g., 'light.turn_on', 'notify.send_message')"
        ): str,

        vol.Optional(
            "entity_id",
            description="Single entity_id or list of entity_ids to target (optional if service supports it)"
        ): vol.Any(str, [str]),

        vol.Optional(
            "target",
            description="Advanced targeting using entity_id, device_id, or area_id (preferred for many services)"
        ): vol.All(
            dict,
            vol.Schema({
                vol.Optional("entity_id"): vol.Any(str, [str]),
                vol.Optional("device_id"): vol.Any(str, [str]),
                vol.Optional("area_id"): vol.Any(str, [str]),
            })
        ),

        vol.Optional(
            "data",
            default={},
            description="Optional additional data payload as a JSON object (e.g., brightness, message, title)"
        ): dict,
    })

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        """Execute the service call."""
        args = tool_input.tool_args
        service_str = args.get("service", "").strip()
        
        if not service_str:
            return {"error": "Missing required parameter: service"}

        if "." not in service_str:
            return {"error": f"Invalid service format: '{service_str}'. Must be 'domain.service_name'"}

        domain, service_name = service_str.split(".", 1)
        full_service = f"{domain}.{service_name}"

        # Prepare service data
        service_data = dict(args.get("data", {}))

        # Handle legacy entity_id
        if "entity_id" in args:
            entity_ids = args["entity_id"]
            if isinstance(entity_ids, str):
                entity_ids = [entity_ids] if entity_ids else []
            service_data["entity_id"] = entity_ids

        # Handle modern target dict (takes precedence)
        if "target" in args and args["target"]:
            service_data["target"] = args["target"]

        # Optional: add who called it (useful for notifications/auditing)
        if llm_context.platform:
            service_data.setdefault("data", {}).update({
                "called_by": f"llm:{llm_context.platform}"
            })

        try:
            _LOGGER.info(
                "LLM calling service %s with data: %s (requested by user %s)",
                full_service,
                service_data,
                llm_context.user_id or "unknown",
            )

            await hass.services.async_call(
                domain,
                service_name,
                service_data=service_data,
                blocking=True,  # Wait for completion when possible
                context=llm_context.create_context(),  # Proper HA context propagation
            )

            return {
                "success": True,
                "service": full_service,
                "service_data": make_json_serializable(service_data),
                "message": f"Successfully called {full_service}",
            }

        except Exception as err:
            _LOGGER.exception("Failed to call service %s", full_service)
            return {
                "success": False,
                "service": full_service,
                "error": str(err),
                "message": f"Failed to call service {full_service}: {err}",
            }