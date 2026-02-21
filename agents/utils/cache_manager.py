"""
Gemini Context Caching Manager.
Handles creation, retrieval, and lifecycle of cached content.
"""

import logging
import hashlib
import datetime
from typing import Optional, List, Dict
from google.genai import types, Client
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages Gemini Context Caching.
    Docs: https://ai.google.dev/gemini-api/docs/caching
    """
    
    def __init__(self, client: Client):
        self.client = client
        
    def _compute_hash(self, content: str, tools: Optional[List[types.Tool]] = None, tool_config: Optional[types.ToolConfig] = None) -> str:
        """Computes a deterministic hash of the content and tools."""
        hash_content = content
        if tools:
            # We need a stable string representation of tools.
            # types.Tool objects might not have a stable str(), but their function_declarations should.
            # Ensure sorting/stability if possible. For now, str() is a decent proxy if deterministic.
            for tool in tools:
                if tool.function_declarations:
                    for fd in tool.function_declarations:
                        hash_content += str(fd)
        if tool_config:
             hash_content += str(tool_config)
        
        return hashlib.md5(hash_content.encode("utf-8")).hexdigest()

    async def get_or_create(
        self, 
        model: str,
        base_name: str, 
        system_instruction: str, 
        tools: Optional[List[types.Tool]] = None,
        tool_config: Optional[types.ToolConfig] = None,
        ttl_minutes: int = 60
    ) -> Optional[types.CachedContent]:
        """
        Retrieves an existing valid cache or creates a new one.
        
        Args:
            model: The model to create cache for (e.g. 'gemini-1.5-pro').
            base_name: Prefix for the cache name (e.g. "supervisor").
            system_instruction: The static system prompt text.
            tools: List of generic Tool objects.
            tool_config: Optional ToolConfig (e.g. mode='AUTO').
            ttl_minutes: Time-to-live in minutes.
            
        Returns:
            CachedContent object name (str) or None if caching failed/disabled.
        """
        # 1. Compute Hash for Versioning
        content_hash = self._compute_hash(system_instruction, tools, tool_config)
        cache_name = f"{base_name}-{content_hash}"
        
        # 3. List/Find existing cache
        # The caching API doesn't support "get by custom alias", we have to list or store ID.
        # Since we don't have persistent storage for IDs across restarts in this simple agent,
        # we will try to create. 
        # BUT: creating a duplicate cache with same content *might* be allowed or defined behavior.
        # Actually, Gemini Caching requires managing the `name` (resource ID) returned by creation.
        
        # STRATEGY: 
        # A. We cannot easily "find" a cache by hash without iterating all caches.
        # B. We will just Create a new one. 
        #    - If we create too many, we hit quotas.
        #    - Efficient agents usually have ONE active cache per "Role".
        
        # Improved Strategy:
        # Since this is a "Session" based agent, we can create it once at startup and reuse.
        # But `supervisor_node` runs every turn. We don't want to create 10 caches for 10 turns.
        
        # Solution: Use an in-memory map for the *current process*.
        # (See `_local_cache` below)
        
        return await self._create_new_cache(model, cache_name, system_instruction, tools, tool_config, ttl_minutes)

    async def _create_new_cache(
        self, 
        model: str,
        display_name: str, 
        system_instruction: str, 
        tools: Optional[List[types.Tool]], 
        tool_config: Optional[types.ToolConfig],
        ttl_minutes: int
    ) -> Optional[types.CachedContent]:
        
        try:
            # Construct Content
            # contents = [types.Content(role="user", parts=[types.Part.from_text(text=system_instruction)])] 
            # WAIT: System instruction in cache is passed via `system_instruction` arg, NOT content list usually.
            # Only "Shared Context" (User/Model turns) go in contents.
            # But specific SDK implementation might vary.
            
            # According to latest SDK:
            # We pass `system_instruction` to the config of the cache.
            
            config = types.CreateCachedContentConfig(
                system_instruction=types.Content(parts=[types.Part.from_text(text=system_instruction)]),
                tools=tools,
                tool_config=tool_config,
                display_name=display_name,
                ttl=f"{ttl_minutes * 60}s"
            )
            
            cached_content = await self.client.aio.caches.create(model=model, config=config)
            logger.info("[%s] Created Context Cache: %s (Exp: %s)", display_name, cached_content.name, cached_content.expire_time)
            return cached_content

        except Exception as e:
            if "contents must be at least" in str(e).lower():
                logger.warning("[%s] Context Caching skipped: Content too short. (%s)", display_name, str(e))
            elif "quota" in str(e).lower():
                logger.warning("[%s] Context Caching skipped: Quota exceeded.", display_name)
            else:
                logger.exception("[%s] Failed to create cache", display_name)
            return None

# Global/Process-level storage for cache reuse within a running process
# (This prevents re-creating cache on every turn of the same process)
_RUNTIME_CACHE_STORE: Dict[str, types.CachedContent] = {}

async def get_process_level_cache(
    client: Client,
    model: str,
    base_name: str,
    system_instruction: str,
    tools: Optional[List[types.Tool]] = None,
    tool_config: Optional[types.ToolConfig] = None,
    ttl_minutes: int = 60
) -> Optional[str]:
    """
    Helper to get a cache name, reusing in-memory reference if valid.
    """
    mgr = CacheManager(client)
    # We include tools in hash if possible
    # Note: _compute_hash handles the tools list -> str conversion
    
    hash_key = mgr._compute_hash(system_instruction, tools, tool_config)
    
    if hash_key in _RUNTIME_CACHE_STORE:
        existing = _RUNTIME_CACHE_STORE[hash_key]
        logger.info("[%s] Reusing cached content: %s", base_name, existing.name)
        return existing.name

    # Create new
    cache_obj = await mgr.get_or_create(model, base_name, system_instruction, tools, tool_config, ttl_minutes)
    
    if cache_obj:
        _RUNTIME_CACHE_STORE[hash_key] = cache_obj
        return cache_obj.name
    
    return None
