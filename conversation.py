"""Grok xAI Conversation Agent."""

import asyncio
import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import voluptuous as vol
import grpc

from homeassistant.const import __version__ as HA_VERSION
from homeassistant.components.conversation import (
    async_set_agent,
    async_unset_agent,
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    ConverseError,
)
from homeassistant.components import persistent_notification, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

from homeassistant.const import (
    CONF_LLM_HASS_API, 
    EVENT_HOMEASSISTANT_START
    )
from homeassistant.core import (
    HomeAssistant, 
    ServiceCall, 
    Context,
    callback
    )
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    entity_registry as er,
    area_registry as ar,
    device_registry as dr,
    floor_registry as fr,
    label_registry as lr,
    intent,
    llm,
    )
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    )

from xai_sdk import AsyncClient, __version__ as xai__version__
from xai_sdk.aio.chat import Chat, Response
from xai_sdk.chat import assistant, system, tool_result, user, file as gfile
from xai_sdk.chat import tool as xai_tool_helper
from xai_sdk.tools import code_execution, web_search, get_tool_call_type

from .const import (
    AREA_CONTEXT_TEMPLATE,
    CLEANUP_INTERVAL,
    CONF_API_KEY,
    DEFAULT_API_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_CLEANUP_PREFIX,
    DOMAIN,
    MAX_CONCURRENT_API_CALLS,
    MAX_RETRIES,
    MAX_TOOL_LOOPS,
    FILE_TYPE_PROMPT,
    FILE_TYPE_CONTEXT,
    FILE_TYPE_TOOLS,
    FILE_TYPE_CONFIGS,
    DEFAULT_FILE_RETENTION_HOURS,
    DEFAULT_MAX_VERSIONS,
    SERVICE_REFRESH_PREFIX,
    )

_LOGGER = logging.getLogger(__name__)

def _to_naive_utc(dt):
    if dt is None:
        return None
    if hasattr(dt, "ToDatetime"):  # protobuf Timestamp
        return dt.ToDatetime().replace(tzinfo=None)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt

def get_descriptions(schema):
    descriptions = {}
    for key in schema.schema:
        if hasattr(key, 'description') and key.description:
            descriptions[key.schema] = key.description
    return descriptions
    
async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
    ) -> None:
    """Set up the Grok conversation agent."""
    _LOGGER.info("Seting up Grokzilla with xai_sdk version = %s", xai__version__)
    config = {**entry.data, **entry.options}

    api_key = config[CONF_API_KEY]
    model = config.get("model", DEFAULT_MODEL)
    prompt = config.get("prompt", DEFAULT_PROMPT)

    agent = GrokConversationAgent(hass, entry, api_key, model, prompt)

    # Store agent
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {"agent": agent}
    
    # Initialize only prompt + context at startup (tools will come after pipeline loads)
    await agent.file_manager.async_ensure_file(FILE_TYPE_PROMPT)
    await agent.file_manager.async_ensure_file(FILE_TYPE_CONTEXT)

    # Register as the default conversation agent
    async_set_agent(hass, entry, agent)
    async_add_entities([agent])
    
    # ----------------------------------------------------------------------
    # CRITICAL FIX: Defer tool listing until HA is running (STATE_RUNNING)
    # ----------------------------------------------------------------------

    # Use the @callback decorator to mark this as an HA callback function.
    @callback
    async def _schedule_get_tools_task(event):
        """Schedules the async_get_tools coroutine to run on the main event loop."""
        
        # We use async_add_job (or hass.async_run_job) here, but the key is
        # ensuring the listener *itself* is correctly marked and structured.
        # Since the traceback shows async_add_job is internally calling async_create_task,
        # we stick with the standard pattern but ensure the listener is registered correctly.
        
        # NOTE: If you are hitting this issue, it suggests the event handler is
        # executing outside of a dedicated "safe" job runner. We use async_add_job 
        # again, but this time ensure the function is marked with @callback.
        
        # If your version is highly strict, you might need to import 
        # async_run_job from homeassistant.core.
        # The documentation suggests using hass.async_add_job, so let's try 
        # enforcing the callback context first.
        
        task = hass.async_create_task(agent.async_get_tools())

        # 2. Await the Task to get the result
        # This line will pause until mycoroutine() is finished
        result = await task

    if hass.state == "running":
        task = hass.async_create_task(agent.async_get_tools())
        result = await task
    else:
        # Pass the @callback decorated function to the listener
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _schedule_get_tools_task)
        #_LOGGER.info("Deferred tool listing until Home Assistant is fully started.")

    # Register refresh services
    async_register_file_services(hass, entry)  

#=========================new service====================================
async def async_generate_smart_home_file(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """
    Generates a comprehensive JSON dump of the entire smart home
    and uploads it to xAI. Returns the file_path.
    Can be called from startup, service, or automation.
    """

    # 1. Collect registries
    frl = fr.async_get(hass)
    arl = ar.async_get(hass)
    drl = dr.async_get(hass)
    erl = er.async_get(hass)
    lrl = lr.async_get(hass)

    
    #schema
    output = {
        "meta": {
            "generated": hass.states.get("sensor.date_time_iso").state + "Z",
            "source": "Home Assistant registries",
            "ha_version": HA_VERSION,
            "description": "Use this file to understand the smart home and find entities and devices in areas or floors"
        },
        "floors by id": {},
        "areas by id": {},
        "devices by id": {},
        "entities by id": {},
    }    
    
    #=============
    # Floors (if you use them)
    for floor in frl.floors.values():
        output["floors by id"][floor.floor_id] = {
                                "name": floor.name,
                                "level": floor.level,
                                "aliases": list(floor.aliases) or [],
                                "area ids": []
                                # floors do not have labels
        }

        
    # Areas
    for area in arl.async_list_areas():
        area_id = area.id
        floor_id = area.floor_id
        output["floors by id"][floor_id]["area ids"].append(area_id)
        #floor_name = frl.floors[floor_id].name if floor_id and floor_id in frl.floors else None
        output["areas by id"][area_id] = {
            "name": area.name,
            "floor_id": floor_id or None,
            "aliases": list(area.aliases) or [],
            "labels": list(area.labels) or [],
            "device ids":[]
        }

    # Devices → collect per-area lists
    for device in drl.devices.values():
        # if device.area_id:
        #     output["idx"]["area_devices"][device.area_id].append(device.id)
        
        if not device.area_id:
            area_id = None
        else:
            area_id = device.area_id
            output["areas by id"][area_id]["device ids"].append(device.id)

        output["devices by id"][device.id] = {
             "name": device.name or device.name_by_user or "Unnamed device",
             "name_by_user": device.name_by_user,
          #  "model": device.model,
          #  "manufacturer": device.manufacturer,
           # "sw_version": device.sw_version,
             "area_id": device.area_id,
             "labels": list(device.labels),
           # "identifiers": list(device.identifiers),
             "entity ids":[]
        }

        
    # Entities — the most important part
    for ent_reg in erl.entities.values():
        if not async_should_expose(hass, conversation.DOMAIN, ent_reg.entity_id):
            _LOGGER.debug("Skipping non-exposed entity; %s", ent_reg.entity_id)
            continue
        if ent_reg.disabled_by or ent_reg.entity_id.startswith("sensor.") and "hidden" in ent_reg.entity_id:
            _LOGGER.debug("Skipping disabled or hidden entity: %s", ent_reg.entity_id)
            continue
    
        entity_id = ent_reg.entity_id
        area_id = ent_reg.area_id
        device_id = ent_reg.device_id
        
        device = drl.async_get(device_id) if device_id else None
        area = arl.async_get_area(area_id) if area_id else None
        floor = None
        # Fallback: Device area
        if not area and device and device.area_id:
            area = arl.async_get_area(device.area_id) if device.area_id else None
            area_id = device.area_id
        if device_id:
            output["devices by id"][device_id]["entity ids"].append(entity_id)
        # Basic entity info (only what Grok really needs)
        output["entities by id"][entity_id] = {
            "name": ent_reg.original_name or ent_reg.entity_id.split(".")[-1].replace("_", " ").title(),
            "class": ent_reg.device_class or ent_reg.entity_category,
            "unit": ent_reg.unit_of_measurement,
            #"area": area_id,
            #"device": device_id,
            "aliases": list(ent_reg.aliases) or [],
            "labels": list(ent_reg.labels) or []
            }

    # ------------------------------------------------------------------
    # 3. Write the file (readable by Grok via file read tool)
    # ------------------------------------------------------------------
    path = Path("/config/grok_smart_home_index.json")
    path.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    
    # Also print size for logs
    size_kb = round(path.stat().st_size / 1024)
    hass.states.async_set(
        "sensor.grok_index_size",
        size_kb,
        {
            "unit_of_measurement": "kB",
            "friendly_name": "Grok Index Size",
            "icon": "mdi:file-document-outline"
        }
    )    
    _LOGGER.info(f"Grok smart-home index generated → {path} ({size_kb} kB)")
    return path

# For prompt (simple example—customize to your DEFAULT_PROMPT + metadata)
async def async_generate_prompt_file(hass: HomeAssistant, entry: ConfigEntry) -> Path | None:
    """Generates a JSON file with the base system prompt."""
    config = {**entry.data, **entry.options}
    prompt_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "home_assistant_version": HA_VERSION,
        "prompt": config.get("prompt", DEFAULT_PROMPT),
        "model": config.get("model", DEFAULT_MODEL),
        "instructions": "This is the base system prompt for Grok. Use it to guide all smart home interactions.",
    }
    dump_path = Path(hass.config.path("grok_prompt.json"))
    dump_path.write_text(json.dumps(prompt_data, indent=2, default=str), encoding="utf-8")
    _LOGGER.info("Prompt JSON written to %s", dump_path)
    return dump_path
    
async def async_generate_tools_file(hass: HomeAssistant, entry: ConfigEntry) -> Path | None:
    """Generate a JSON file containing all available HA LLM tools for Grok."""
    # agent = hass.data[DOMAIN][entry.entry_id]["agent"]
    # ha_tools = await agent.ensure_tools()  # Make sure tools are loaded
    agent = hass.data[DOMAIN][entry.entry_id]["agent"]
    task = hass.async_create_task(agent.async_get_tools())

    # 2. Await the Task to get the result
    # This line will pause until mycoroutine() is finished
    success  = await task

    catalog = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "home_assistant_version": HA_VERSION,
        "total_tools": len(agent._grok_tools),
        "tools": [{"name":t.function.name, "description": t.function.description, "parameters":t.function.parameters} for t in agent._grok_tools]
    }
    
    # _LOGGER.debug("\n Tool attrs::: %s", dir(agent._grok_tools[-1]))
    # _LOGGER.debug("\n Tool object::: %s", agent._grok_tools[-1].function.parameters)
    

    dump_path = Path(hass.config.path("grok_tools_catalog.json"))
    dump_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    #_LOGGER.info("Tools catalog written to %s (%d tools)", dump_path, len(agent._grok_tools))
    return dump_path

def async_register_file_services(hass: HomeAssistant, entry: ConfigEntry) -> None:
    async def handle_refresh(call: ServiceCall) -> None:
        type_key = call.data.get("file_type")
        if type_key not in FILE_TYPE_CONFIGS:
            _LOGGER.error("Invalid file_type: %s", type_key)
            return

        agent_data = hass.data[DOMAIN].get(entry.entry_id)
        if not agent_data or "agent" not in agent_data:
            return

        agent = agent_data["agent"]
        #_LOGGER.info("Refresh requested for %s file", type_key)

        success = await agent.file_manager.async_refresh_file(type_key)
        if success:
            _LOGGER.info("Successfully refreshed %s file", type_key)

    hass.services.async_register(
        domain=DOMAIN,
        service="refresh_file",
        service_func=handle_refresh,
        schema=vol.Schema({
            vol.Required("file_type"): vol.In(["prompt", "context", "tools"]),
        }),
    )
#===============================new service=========================

@dataclass
class FileTypeConfig:
    """Dataclass for file type configs (from const.FILE_TYPE_CONFIGS)."""
    filename: str
    max_versions: int
    generator: str  # Name of async func to call
    key_in_hass_data: str
    retention_hours: int = DEFAULT_FILE_RETENTION_HOURS

class GrokFileManager:
    """Generic manager for Grok file uploads, reuse, and cleanup."""

    def __init__(self, hass: HomeAssistant, agent):
        self.hass = hass
        self.agent = agent  # For upload_file, etc.
        self._configs = {}
        self._load_configs()

    def _load_configs(self):
        """Load file type configs from const."""
        for type_key, cfg_dict in FILE_TYPE_CONFIGS.items():
            cfg = FileTypeConfig(
                filename=cfg_dict["filename"],
                max_versions=cfg_dict["max_versions"],
                generator=cfg_dict["generator"],  # We'll resolve to func later
                key_in_hass_data=cfg_dict["key_in_hass_data"],
                retention_hours=cfg_dict.get("retention_hours", DEFAULT_FILE_RETENTION_HOURS),
            )
            self._configs[type_key] = cfg

    def _get_generator(self, type_key: str) -> callable:
        """Resolve generator func name to actual async func."""
        cfg = self._configs[type_key]
        # Dynamic import/resolution—assumes funcs are in this module
        # For simplicity: use globals() or importlib; here, assume module-level
        generator_name = cfg.generator
        if generator_name in globals():
            gen_func = globals()[generator_name]
            if asyncio.iscoroutinefunction(gen_func):
                return gen_func
        raise ValueError(f"Generator '{generator_name}' not found for type '{type_key}'")

    async def async_get_file_list(self) -> List[Any]:
        """List all files (cached or fresh)."""
        return await self.agent.async_get_grok_file_list()

    async def async_ensure_file(self, type_key: str, force_new: bool = False) -> str | None:
        """
        Get existing file_id for type (reuse if recent) or generate/upload new.
        Stores cur_id in hass.data[DOMAIN][cfg.key_in_hass_data].
        """
        if type_key not in self._configs:
            _LOGGER.warning("Unknown file type: %s", type_key)
            return None

        cfg = self._configs[type_key]
        entry = self.agent.entry  # Assume agent has entry

        # 1. Check stored cur_id
        cur_id = self.hass.data[DOMAIN].get(cfg.key_in_hass_data)
        if cur_id:
            # Verify it exists and is recent
            file_data = await self.async_get_file_list()
            existing = next((f for f in file_data if f.id == cur_id), None)
            if existing and existing.created_at:
                created_dt = _to_naive_utc(existing.created_at)
                age_seconds = (datetime.utcnow() - created_dt).total_seconds()
                if (created_dt and age_seconds < cfg.retention_hours * 3600):                    
                    _LOGGER.debug("Reusing %s file %s (%s old)", 
                                    type_key, 
                                    cur_id, 
                                    str(age_seconds))
                    return cur_id
    
        # 2. Fallback: search by filename — BUT ONLY IF WE DON'T WANT TO FORCE A NEW ONE
        file_data = await self.async_get_file_list()
        if not force_new:  # ← ADD THIS PARAMETER
            existing = next((f for f in file_data if f.filename == cfg.filename), None)
            if existing:
                created_dt = _to_naive_utc(existing.created_at)
                if created_dt and (datetime.utcnow() - created_dt).total_seconds() < cfg.retention_hours * 3600:
                    file_id = existing.id
                    self.hass.data[DOMAIN][cfg.key_in_hass_data] = file_id
                    _LOGGER.info("Reusing existing %s file %s", type_key, file_id)
                    return file_id
                
        # 3. Generate & upload new
        _LOGGER.info("Generating and uploading new %s file", type_key)
        gen_func = self._get_generator(type_key)
        tmp_path = await gen_func(self.hass, entry)
        if not tmp_path:
            _LOGGER.error("Failed to generate %s file", type_key)
            return None

        file_id = await self.agent.async_upload_file(str(tmp_path), cfg.filename)
        if file_id:
            self.hass.data[DOMAIN][cfg.key_in_hass_data] = file_id
            await self._async_cleanup_old_files(type_key, file_data)
            _LOGGER.debug("New %s file uploaded: %s", type_key, file_id)
            return file_id
        return None

    async def _async_cleanup_old_files(self, type_key: str, file_data: List[Any]) -> None:
        """Delete old files for this type (keep max_versions)."""
        cfg = self._configs[type_key]
        _LOGGER.debug("FILE CLEANUP: cfg: %s", cfg)
 
        files = [f for f in file_data if f.filename.startswith(cfg.filename)]  # Or exact match
        files.sort(
                key=lambda f: _to_naive_utc(f.created_at),
                reverse=True  # Newest first
                )
        for old in files[cfg.max_versions:]:
            await self.agent.async_del_grok_file(old.id)
            _LOGGER.info("Deleted old %s file %s", type_key, old.id)

    async def async_refresh_file(self, type_key: str) -> bool:
        if type_key not in self._configs:
            return False
    
        cfg = self._configs[type_key]
        self.hass.data[DOMAIN].pop(cfg.key_in_hass_data, None)
        _LOGGER.info("Forcing refresh of %s file", type_key)
    
        # CRITICAL: Force tool loading before generating file
        if type_key == FILE_TYPE_TOOLS:
            if not self.agent._grok_tools:
                _LOGGER.error("No tools loaded — cannot create tools file")
                return False
    
        new_id = await self.async_ensure_file(type_key, force_new=True)
        return new_id is not None
        
    def get_file_id(self, type_key: str) -> str | None:
        """Get current file_id from hass.data (sync getter)."""
        cfg = self._configs[type_key]
        return self.hass.data[DOMAIN].get(cfg.key_in_hass_data)
            
class GrokConversationState:
    """ keeps track of communication device, the latest response_id
    and the time the respone_id was updated.  Used to continue conversations
    for a particular conversation input device"""
    
    def __init__(self,
                 device_id,
                 ):
        self._device_id = device_id
        self._grok_response_id = None
        self._last_ts = datetime.now()
    
    @property 
    def device_id(self):
        return self._device_id
        
    @property
    def grok_response_id(self):
        return self._grok_response_id
        
    @grok_response_id.setter
    def grok_response_id(self, rsp_id):
        self._grok_response_id = rsp_id
        self._last_ts = datetime.now()
        
    @property
    def last_useage_time(self):
        return self._last_ts
    

class GrokConversationAgent(ConversationEntity):
    """Grok xAI conversation agent (LLM API + Stateful History)."""

    _attr_has_entity_name = True
    _attr_name = None  # Use device/entity name from config
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_key: str,
        model: str,
        prompt: Optional[str] = None,
    ) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self.entry = entry
        self.api_key = api_key
        self.model = model
        self.base_prompt = prompt or DEFAULT_PROMPT
        
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="xAI",
            model=self.model,
        )
        
        self.device_conversation_states = {}

        self.client = AsyncClient(api_key=self.api_key)
        # Simple in-memory conversation history storage
        # Note: In production, consider limiting size or using LRU cache
        self.conversations: Dict[str, ConversationContext] = {}
        self._lock = asyncio.Lock()
        self.rate_limiter = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
        
        self.file_manager = GrokFileManager(hass, self)
        
        self.grok_context_file_id = None
        
        self._grok_tools: List[Any] | None = None
        self._ha_tools: List[Any] | None = None
        self._tools_initialized = asyncio.Event()

        # Cache registries
        self.area_reg = ar.async_get(hass)
        self.floor_reg = fr.async_get(hass)
        self.dev_reg = dr.async_get(hass)

    async def async_added_to_hass(self) -> None:
        """Run when entity is added."""
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed."""
        # Unregister only if we are the active agent
        # logic handled in init/unload mostly, but good to have checks
        await super().async_will_remove_from_hass()
        
    async def _ensure_pipeline_loaded(self) -> None:
        """Guarantee that assist_pipeline has loaded at least once."""
        if "assist_pipeline" not in self.hass.config.components:
            from homeassistant.components import assist_pipeline
            await assist_pipeline.async_setup_pipeline(self.hass)

    async def async_get_tools(self):
        # 1. Create a "dummy" LLMContext required to fetch the API
        # Since this is setup, we don't have a user command yet, so we use defaults.
        context = Context() # Standard HA Context
        llm_context = llm.LLMContext(
            platform=DOMAIN, # Your integration's domain
            context=context,
            language="en",
            assistant='barabashka', # Default assistant name
            device_id=None
        )
    
        # 2. Get the specific LLM API you want to inspect.
        # 'assist' is the default built-in API that contains HA control tools.
        # You can also use llm.async_get_apis(hass) to list all available API IDs.
       # _LOGGER.info("========== TOOL LISTING ==========")
        
        # ----------------------------------------------------------------------
        # FIX: Polling to wait for the API to transition from STARTING to RUNNING
        # ----------------------------------------------------------------------
        MAX_RETRIES = 5
        DELAY_SECONDS = 7
        tool_data = []
        for attempt in range(MAX_RETRIES):
            apis = llm.async_get_apis(self.hass)
            
            if not apis:
                _LOGGER.warning("No LLM APIs found on attempt %s", attempt + 1)
                await asyncio.sleep(DELAY_SECONDS)
                continue
    
            # Check if all APIs are past the 'STARTING' phase
            all_running = True
            for api in apis:
                # We assume 'RUNNING' means it's ready. If the attribute is not set, 
                # we rely on the internal API method calls to succeed.
                llm_api = await llm.async_get_api(self.hass, api.id, llm_context)
                if len(llm_api.tools) == 0:
                    all_running = False
                    _LOGGER.debug("API %s state is %s, waiting...", 
                                    api.name, 
                                    api.hass.state)
                    break
                #_LOGGER.debug("Tools: %s", llm_api.tools)
            if all_running:
                #_LOGGER.info("LLM APIs are ready after %s attempts.", attempt + 1)
                break
            
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(DELAY_SECONDS)
            else:
                _LOGGER.error("LLM APIs did not reach 'RUNNING' state"
                                " after %s attempts.", MAX_RETRIES)
                return "Error" # Exit if we failed to get them running
        tools = []       
        for api in apis:
            api_id = api.id
            #_LOGGER.debug("getting tools for %s", api_id)
            try:
                # Get the API instance (This might be a MergedAPI 
                # if multiple sources are combined)
                llm_api = await llm.async_get_api(self.hass, api_id, llm_context)
                
                #_LOGGER.info("Listing tools for %s", api.name)
                
                # 3. Get the list of tools
                # The tools are dynamic and may depend on the exposed 
                # entities in the context
                tools.extend(llm_api.tools)
                
                # Alternatively, if 'tools' is not a property of
                # the specific API class version:
                if not tools and hasattr(llm_api, "async_get_tools"):
                    tools.extend(await llm_api.async_get_tools())
            except Exception as e:
                _LOGGER.error(f"Error fetching LLM tools: {e}")
                return False
                
        #_LOGGER.info("DONE GETTING TOOLS")
                    
        if len(tools) > 0:
            self._ha_tools = tools
            self._grok_tools = self._format_tools_for_grok(self._ha_tools)
            self._grok_tools.append(code_execution())
            return True
        else:
            _LOGGER.error(f"Error no tools found")
            return False

    async def async_upload_file(self, file_path: str, file_name: str ) -> str | None:
        """
        Upload a file to xAI/Grok for use with assistants, fine-tuning, etc.
        
        Args:
            file_path: Path to the file on disk
            purpose: Must be "assistants" (or "fine-tune" if supported later)
    
        Returns:
            File ID (str) or None on failure
        """
        try:
            with open(file_path, "rb") as f:
                file_object = await self.client.files.upload(
                    file=f,
                    filename=file_name
                )
            _LOGGER.info("Uploaded file %s → ID: %s", file_path, file_object.id)
            return file_object.id
    
        except Exception as err:
            _LOGGER.error("Failed to upload file to xAI: %s", err)
            return None
            
    async def async_get_grok_file_list(self):
        response = await self.client.files.list(
                                            limit=100,
                                            order="desc",
                                            sort_by="created_at"
                                            )
        for file in response.data:
            _LOGGER.debug(f"File: {file.filename} (ID: {file.id}, Size: {file.size} bytes)")
        _LOGGER.debug("there are %s files on grok server", len(response.data))
        return response.data
        
    async def async_del_grok_file(self, file_id):
        await self.client.files.delete(file_id)
        
    async def async_initialize_files(self) -> Dict[str, str]:
        """Ensure all files at startup (returns {type: file_id})."""
        file_ids = {}
        for type_key in FILE_TYPE_CONFIGS:
            file_id = await self.file_manager.async_ensure_file(type_key)
            if file_id:
                file_ids[type_key] = file_id
        return file_ids

    async def async_cleanup_conversations(self, now: Optional[datetime] = None) -> None:
        """Clean up inactive conversations."""
        _LOGGER.debug("Running periodic conversation cleanup...")
        #TODO: Delete conversation_id on grok server
        current_time = datetime.now()
        async with self._lock:
            to_remove = [
                conv_id
                for conv_id, ctx in self.conversations.items()
                if current_time - ctx.last_update_time > timedelta(minutes=10)
            ]
            for conv_id in to_remove:
                self.conversations.pop(conv_id, None)

    @property
    def supported_languages(self) -> list[str] | None:
        """Return a list of supported languages."""
        return ["en"]  # or ["en", "de", "fr", ...] as needed

    async def _get_area_context(self, user_input: ConversationInput) -> str:
        """Return area-specific context if the user is speaking from a known area."""
        if not user_input.device_id:
            return ""
        
        # Registry lookups are synchronous, but wrapped in async usually safe or fast
        device = self.dev_reg.async_get_device(user_input.device_id)
        if not device or not device.area_id:
            #_LOGGER.warning("failed to get user_input device (id: %s) or area_id", user_input.device_id)
            return ""
        area_id = device.area_id
        if not area_id:
            return ""

        area = self.area_reg.async_get_area(area_id)
        _LOGGER.info("User is in %s", area.name)
        return AREA_CONTEXT_TEMPLATE.format(area=area.name) if area else ""

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log: ChatLog
    ) -> ConversationResult:
        """Process a user message.
        
        This method handles the core logic for processing user input, building prompts,
        collecting tools, and interacting with the xAI SDK. It now enables message storage
        on the xAI backend to minimize token traffic in repeated conversations.
        """
        
        #  Prerequisites: quasi-static Prompt and context info is to be uploaded
        #       to Grok as files.  It consists of a base prompt file and the
        #       HA context file.  Its is updated when:
        #          A. Upon Setup of this integration
        #          B. When triggered (by user, automation, etc.)
        
        #  Basic sequence of code:
        #       1. check if this is an onoing conversation or a new one
        #          NOTE: conversations are tracked by the input device they
        #                come from.  At most an input device can have one
        #                ongoing conversation, or none (then create new)
        
        if user_input.device_id not in  self.device_conversation_states:
            gcs = GrokConversationState(user_input.device_id)
            self.device_conversation_states[user_input.device_id] = gcs
        else:
            gcs =  self.device_conversation_states[user_input.device_id]
        
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                self.entry.options.get(CONF_LLM_HASS_API, []),  # e.g., ["assist_pipeline"]
                None,
                None,
            )
            # Now chat_log.llm_api is populated — use it as fallback if global is empty
            if chat_log.llm_api and hasattr(chat_log.llm_api, "tools"):
                # Cache these tools in your agent for the file manager
                if not self._grok_tools:
                    ha_tools = chat_log.llm_api.tools
                    # _LOGGER.debug("HA_TOOLS: %s", ha_tools)
                    self._grok_tools = self._format_tools_for_grok(ha_tools)
                    #_LOGGER.info("Loaded %d tools from chat_log fallback", len(ha_tools))
                    # _LOGGER.debug("GROK_TOOLS: %s", self._grok_tools)
        except ConverseError as err:
            _LOGGER.debug("LLM data load skipped: %s", err)
        
        #await self.ensure_tools()
        
        #        1.A.  If ongoing, tell grok to reference conversation_id
        #                   chat = client.chat.create(
        #                           model="grok-4",
        #                           previous_response_id=response.id,
        #                           store_messages=True,)
        if gcs.grok_response_id is not None:
            gchat = self.client.chat.create(
                                    model=self.model,
                                    previous_response_id=gcs.grok_response_id,
                                    store_messages=True,
                                    messages=[user(user_input.text)]
                                    )
                                    
        #        1.B.  If new, 
        #               I. Create new chat with  store_messages=True
        #               II. send Grok file_id of quasit-staic context
        #               III. Get and provide grok with fall back area context
        #               IV.  tools=self.grok_tools, tool_choice="auto"
        #               V.   user's message 
        else:
            # get area that input device is located in 
            area_context = await self._get_area_context(user_input)
            
            # Make absolutely sure tools are available
            # await self._ensure_pipeline_loaded()          # ← new
            # await self.ensure_tools()        # ← now works
            # await self.file_manager.async_ensure_file(FILE_TYPE_TOOLS)  # creates full catalog
            
            # Get file_ids generically
            prompt_id = self.file_manager.get_file_id(FILE_TYPE_PROMPT)
            context_id = self.file_manager.get_file_id(FILE_TYPE_CONTEXT)
            tools_id = self.file_manager.get_file_id(FILE_TYPE_TOOLS)
            
            config = {**self.entry.data, **self.entry.options}
            prompt =  config.get("prompt", DEFAULT_PROMPT)
            
            sm_context_instr = "For area-based queries, ONLY search the grok_smart_home_context.json file for entities where 'areas' includes the target (e.g., 'Bedroom'). Extract names, domains; ignore missing fields like entity_id."

            path = Path("/config/grok_smart_home_index.json")
            file_sys_context = path.read_text()

            # NEW: Barabashka influence
            collector = self.hass.data[DOMAIN][self.entry.entry_id].get("barabashka_collector")
            if collector:
                spirit_msg = collector.get_current_spirit_message()
                barabashka_msg = system(
                                        f"""Barabashka speaks from the house itself: {spirit_msg}
                                        You are channeling the ancient house spirit Barabashka. 
                                        Weave this message subtly into your tone and answers. 
                                        If asked directly about the Barabashka, reveal this message fully."""
                                    )

            
            messages = [
                #system(f"You are Barabashka the house spirit. Use the grok_prompt.json file for your Base prompt: "),  # Fallback inline
                
                #system("Use the grok_tools_catalog.json file for tool_call Tools catalog" ),
                system(prompt),
                barabashka_msg,
                #system(sm_context_instr),
                system(file_sys_context),
                system(area_context) if area_context else None,
                user(user_input.text) #, gfile(context_id))
                ]
            messages = [m for m in messages if m is not None]  # Filter Nones
            
            # Create new chat with context and tools in first message
            _LOGGER.debug("New Conversation started with: %s", messages)
            gchat = self.client.chat.create(
                                    model=self.model,
                                    store_messages=True,
                                    tools=self._grok_tools or [],
                                    tool_choice='auto',
                                    messages=messages,
                                    )
            
        #   3. chat.sample and Wait for response (_async_sample_with_retries)
        response = await self._async_sample_with_retries(gchat)
        gcs.grok_response_id = response.id  #update

        #   4. Start Tool loop while response has tool_calls
        for loop_count in range(MAX_TOOL_LOOPS):
            if not response.tool_calls:
                break
            # if response.finish_reason == "REASON_STOP":
            #     break
            _LOGGER.debug("\n ==== In Tool Call Loop %s ==== \n", loop_count + 1)
            # Execute tools and get results
            raw_tool_results = await self._async_execute_tool_calls(response.tool_calls)

            # Create list of tool call results) 
            tool_results_list = [assistant(response.content or "")]
            for result in raw_tool_results:
                _LOGGER.debug("Tool result: %s", result["content"])
                gchat.append(tool_result(result["content"]))

            
            # Resample with updated history
            response = await self._async_sample_with_retries(gchat)
            gcs.grok_response_id = response.id  #update
            
        #   5. If MAX_TOOL_LOOPS exceeded throw exception, catch and respond
        else:
            return self._create_error_response(
                user_input.language or "en",
                "Sorry, reached max tool loops without resolution. \n" + response.content
            )
            

        #    6. After tool_calls while loop.  Take last response as
        #       input to HA:
        #       intent_response.async_set_speech(response)
        assistant_text = response.content or ""

        intent_response = intent.IntentResponse(language=user_input.language or "en")
        intent_response.async_set_speech(assistant_text.strip())

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
        )

    async def _async_execute_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute HA tools and return xAI compatible results."""
        results = []
        
        for call in tool_calls:
            call_id = call.id
            tool_name = call.function.name
            args_str = call.function.arguments
            tool_call_type = get_tool_call_type(call)
            if tool_call_type  != "client_side_tool":
                _LOGGER.info("Server side tool call: type: %s, Name: %s, ars: %s", 
                                tool_call_type, 
                                tool_name,
                                args_str)
                continue
            
            try:
                tool_args = json.loads(args_str)
            except json.JSONDecodeError:
                results.append({
                    "id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": "Invalid JSON arguments"})
                })
                continue
    
            _LOGGER.debug(">>>>>> Executing tool %s with args %s", tool_name, tool_args)
    
            # FIRST: Try to find tool in our own registered tools (this catches custom ones!)
            found_tool = None
            if self._ha_tools:
                for t in self._ha_tools:
                    # xai_tool_helper stores the name directly
                    # _LOGGER.debug("TTTTTTTTT: %s", getattr(t, "name", None))
                    if getattr(t, "name", None) == tool_name:
                        found_tool = t
                        break
    
            # SECOND: Fallback to global LLM registry (for built-in assist__ tools)
            if not found_tool:
                for api in llm.async_get_apis(self.hass):
                    if not hasattr(api, "async_get_tools"):
                        continue
                    try:
                        tools = await api.async_get_tools()
                        for t in tools:
                            if t.name == tool_name:
                                found_tool = t
                                break
                    except Exception as e:
                        _LOGGER.debug("API failed during tool lookup: %s", e)
                    if found_tool:
                        break
    
            if not found_tool:
                results.append({
                    "id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": f"Tool {tool_name} not found"})
                })
                continue
    
            # Execute the tool
            try:
                tool_input = llm.ToolInput(
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                
                response_data = await found_tool.async_call(
                    self.hass,
                    tool_input,
                    llm.LLMContext(
                        platform=DOMAIN,
                        context=None,
                        language="en",
                        assistant=DOMAIN,
                        device_id=None,
                    ),
                )
                
                content_str = json.dumps(response_data) if not isinstance(response_data, str) else response_data
                results.append({
                    "id": call_id,
                    "name": tool_name,
                    "content": content_str
                })
                
            except Exception as e:
                _LOGGER.exception("Tool %s execution failed", tool_name)
                results.append({
                    "id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)})
                })
        
        return results

    def _create_error_response(self, language: str, error_message: str) -> ConversationResult:
        """Create an error response."""
        intent_response = intent.IntentResponse(language=language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            error_message
        )
        return ConversationResult(response=intent_response)
    
    def _schema_to_json_schema(self, schema: Any) -> dict:
        """Robust schema converter."""
        try:
            return self._vol_to_json_schema(schema)
        except Exception:
            return {"type": "string"}  # Fallback

    async def _async_sample_with_retries(self, chat: Chat) -> Response:
        """Sample chat with retries."""
        async with self.rate_limiter:
            for attempt in range(MAX_RETRIES):
                try:
                    result = await asyncio.wait_for(chat.sample(), timeout=DEFAULT_API_TIMEOUT)
                    _LOGGER.debug("Id: %s", result.id)
                    _LOGGER.debug("system_fingerprint: %s", result.system_fingerprint)
                    _LOGGER.debug("Created: %s", result.created)
                    _LOGGER.debug("Citations: %s", result.citations)
                    _LOGGER.debug("Content: %s", result.content)
                    _LOGGER.debug("Finish reason: %s", result.finish_reason)
                    _LOGGER.debug("Debug output: %s", result.debug_output)
                    _LOGGER.debug("Encrypted content: %s", result.encrypted_content)
                    
                  
                    _LOGGER.debug("Logprobs: %s", result.logprobs)
                    _LOGGER.debug("Process chunk: %s", result.process_chunk)
                    _LOGGER.debug("Protobuffer: %s", result.proto)
                    _LOGGER.debug("Reasoning content: %s", result.reasoning_content)
                    _LOGGER.debug("request settings: %s", result.request_settings)
                    _LOGGER.debug("role: %s", result.role)
                    _LOGGER.debug("server_side_tool_usage: %s", result.server_side_tool_usage)
                    _LOGGER.debug("tool_calls: %s", result.tool_calls)
                    #_LOGGER.debug("result dir: %s", dir(result))
                    self.process_usage(result)
                    return result
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(1)
            raise HomeAssistantError("Unreachable code")
            
    def process_usage(self, response):
        """ 'cached_prompt_text_tokens', 'completion_tokens', 
        'num_sources_used', 'prompt_image_tokens', 'prompt_text_tokens', 
        'prompt_tokens', 'reasoning_tokens', 'server_side_tools_used', 
        'total_tokens'"""
        _LOGGER.debug("cached_prompt_text_tokens: %s", response.usage.cached_prompt_text_tokens)
        _LOGGER.debug("completion_tokens: %s", response.usage.completion_tokens)
        _LOGGER.debug("num_sources_used: %s", response.usage.num_sources_used)
        _LOGGER.debug("prompt_image_tokens': %s", response.usage.prompt_image_tokens)
        _LOGGER.debug("prompt_text_tokens: %s", response.usage.prompt_text_tokens)
        _LOGGER.debug("prompt_tokens: %s", response.usage.prompt_tokens)
        _LOGGER.debug("reasoning_tokens: %s", response.usage.reasoning_tokens)
        _LOGGER.debug("server_side_tools_used: %s", response.usage.server_side_tools_used)
        _LOGGER.debug("total_tokens: %s", response.usage.total_tokens)


    def _vol_to_json_schema(self, vs: Any) -> dict:
        """Convert Voluptuous schema with HA selectors → valid xAI JSON schema.

        100% safe for HA 2025.11+ where NumberSelector.config can be None
        and attributes may be missing.
        """
        if isinstance(vs, dict):
            schema_dict = vs
        elif isinstance(vs, vol.Schema):
            schema_dict = vs.schema
        else:
            schema_dict = vs

        # _LOGGER.debug("Raw schema for conversion: %s", schema_dict)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for key, validator in schema_dict.items():
            # Handle Required / Optional
            if isinstance(key, (vol.Required, vol.Optional)):
                name = str(key.schema)
                is_required = isinstance(key, vol.Required)
                description = key.description or ""
            else:
                name = str(key)
                is_required = False
                description = ""

            if is_required:
                required.append(name)

            jtype = "string"
            extra: dict[str, Any] = {}

            # --- HA Selectors (fully safe) ---
            if isinstance(validator, TextSelector):
                jtype = "string"
                description = description or "Text input"

            elif isinstance(validator, NumberSelector):
                # config can be None in recent HA versions
                cfg = getattr(validator, "config", None)
                step = getattr(cfg, "step", None) if cfg else None
                mode = getattr(cfg, "mode", None) if cfg else None
                min_val = getattr(cfg, "min", None) if cfg else None
                max_val = getattr(cfg, "max", None) if cfg else None

                if mode == NumberSelectorMode.BOX and step == 1:
                    jtype = "integer"
                else:
                    jtype = "number"

                if min_val is not None:
                    extra["minimum"] = min_val
                if max_val is not None:
                    extra["maximum"] = max_val
                description = description or "Numeric value"
                
            elif isinstance(key, vol.Any):
                # We need to pick one representative name for the JSON schema
                # and put the possible choices in the description for the LLM.
                choices = [str(k) for k in key.validators]
                name = choices[0]  # Use the first one as the key name
                is_required = False # vol.Any is not explicitly Required/Optional
                # Use a specific description to guide the LLM
                description = f"One of the following identifiers must be used: {', '.join(choices)}"

            elif isinstance(validator, BooleanSelector):
                jtype = "boolean"
                description = description or "True or false"

            # --- Basic Voluptuous types ---
            elif validator in (str, vol.Coerce(str)):
                jtype = "string"
            elif validator in (int, vol.Coerce(int)):
                jtype = "integer"
            elif validator in (float, vol.Coerce(float)):
                jtype = "number"
            elif validator in (bool, vol.Coerce(bool)):
                jtype = "boolean"

            properties[name] = {
                "type": jtype,
                "description": description or "",
                **extra,
            }

        result = {
            "type": "object",
            "properties": properties,
            "required": required if required else [],
            "additionalProperties": False,
        }

        # _LOGGER.debug("Successfully converted schema → %s", result)
        return result

    def _format_tools_for_grok(self, ha_tools: Sequence[llm.Tool]) -> List[Any]:
        """Safely format every HA tool for the xAI SDK."""
        grok_tools = []
        for tool in ha_tools:
            try:
                params = self._vol_to_json_schema(tool.parameters)
                grok_tools.append(
                    xai_tool_helper(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=params,
                    )
                )
                #_LOGGER.info("Successfully registered tool: %s", tool.name)
            except Exception as exc:
                _LOGGER.warning("Skipping tool %s – schema conversion failed: %s", tool.name, exc)
        return grok_tools
