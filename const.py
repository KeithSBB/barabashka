"""Constants for the Grok xAI Conversation integration.

This file centralizes all static values used across the integration,
such as domain, default models, prompts, and API configurations.
"""

# Integration domain identifier
DOMAIN = "barabashka"

# Default xAI model (fallback for config)
DEFAULT_MODEL = "grok-beta"

# List of supported models for options flow
MODEL_OPTIONS = [
    "grok-beta",
    "grok-4",
    "grok-code-fast-1",
    "grok-4-fast-non-reasoning",
]

# Default system prompt for Grok conversation agent
# Tool instructions and HA Context are injected dynamically.
DEFAULT_PROMPT = (
    """You are Barabashka the house spirit and a helpful home automation assistant. 
     You are integrated into a Home Assistant instance. 
    For device control or reading sensor values, ALWAYS use the provided tools. 
    NEVER hallucinate entity states, sensor values, or temperatures. 
    If a tool returns no results or you cannot find an entity, honestly say so
    and ask the user for clarification. Example: For lights, use 
    'barabashka_CallService' with domain 'light', service 'turn_on', 
    target as a JSON object with entity_id set to 'light.bedroom', 
    data as a JSON object with brightness_pct set to 50. 
    For general queries, provide concise, accurate answers."""
).strip()

# Default API timeout in seconds
DEFAULT_API_TIMEOUT = 30

# Maximum concurrent API calls (for asyncio.Semaphore)
MAX_CONCURRENT_API_CALLS = 10
MAX_RETRIES = 3
# Maximum loops for tool-calling to prevent infinite loops
MAX_TOOL_LOOPS = 5

# Maximum number of results to return with search tool
MAX_SEARCH_RESULTS = 10

# Interval for conversation cleanup task in seconds
CLEANUP_INTERVAL = 60  # Note: This is 60 seconds (1 minute)

# Template for area context (used in conversation.py)
AREA_CONTEXT_TEMPLATE = (
    "The user is speaking from the {area} area."
    "Use this to disambiguate entities if needed."
)

# Grok server side file management constants
# File type keys (used as dict keys and service suffixes)
FILE_TYPE_PROMPT = "prompt"
FILE_TYPE_CONTEXT = "context"
FILE_TYPE_TOOLS = "tools"

# Global defaults (can be overridden per-type)
DEFAULT_FILE_RETENTION_HOURS = 24
DEFAULT_MAX_VERSIONS = 5
DEFAULT_CLEANUP_PREFIX = "grok_"  # For filename matching during cleanup

# Per-type configs: {type_key: {"filename": str, "max_versions": int, "generator": callable, "key_in_hass_data": str}}
# - filename: Base name (e.g., "grok_prompt.json")
# - max_versions: How many old versions to keep
# - generator: Async func: async def gen(hass, entry) -> Path (e.g., async_generate_prompt_file)
# - key_in_hass_data: For storing cur_file_id, e.g., "cur_grok_prompt_file_id"
# - retention_hours: Optional override (default 24)
FILE_TYPE_CONFIGS = {
    FILE_TYPE_PROMPT: {
        "filename": "grok_prompt.json",
        "max_versions": 1,  # Prompts change rarely
        "generator": "async_generate_prompt_file",  # You'll define this func
        "key_in_hass_data": "cur_grok_prompt_file_id",
        "retention_hours": 48,  # Longer for prompts
    },
    FILE_TYPE_CONTEXT: {
        "filename": "grok_smart_home_context.json",  # Matches your existing
        "max_versions": 1,  # Contexts update more often
        "generator": "async_generate_smart_home_file",  # Your existing func
        "key_in_hass_data": "cur_grok_context_file_id",
    },
    FILE_TYPE_TOOLS: {
        "filename": "grok_tools_catalog.json",
        "max_versions": 1,
        "generator": "async_generate_tools_file",  # Your new func from previous
        "key_in_hass_data": "cur_grok_tools_file_id",
    },
    # Future example: FILE_TYPE_EXAMPLES: { ... "generator": "async_generate_examples_file" }
}

# Service prefixes
SERVICE_REFRESH_PREFIX = "refresh"

# Config keys
GRPC_API_KEY_HEADER = "x-api-key"
CONF_API_KEY = "api_key"
MAX_PROMPT_LENGTH = 2000  # For validation
CONF_MODEL_OPTIONS = MODEL_OPTIONS
