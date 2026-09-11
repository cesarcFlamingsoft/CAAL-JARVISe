"""Provider-agnostic LLM Node for CAAL.

This module provides a custom llm_node implementation that works with any
LLMProvider (Ollama, Groq, etc.) while maintaining full tool calling support.

Key Features:
- Provider-agnostic LLM calls via LLMProvider interface
- Tool discovery from @function_tool methods and MCP servers
- Tool execution routing (agent methods, n8n workflows, MCP tools)
- Streaming responses for best UX

Usage:
    class MyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            async for chunk in llm_node(
                self, chat_ctx, provider=self._provider
            ):
                yield chunk
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from caal import settings as settings_module
from caal.tools import create_default_registry
from caal.tools.errors import SafeToolError
from caal.tools.knowledge_tools import session_unavailable_result
from caal.user_scope import (
    memory_unavailable_result,
    scheduling_unavailable_result,
    scoped_tool_arguments,
)

from ..integrations.n8n import execute_n8n_workflow
from ..utils.formatting import strip_markdown_for_tts
from .agent_tools import resolve_agent_method_tool
from .context_barrier import (
    TOOL_DATA_HEADER,
    is_knowledge_tool,
    record_private_answer,
    sanitize_for_escalation,
)
from .providers import LLMProvider

if TYPE_CHECKING:
    from .providers import ToolCall

logger = logging.getLogger(__name__)

__all__ = ["llm_node", "ToolDataCache"]


class ToolDataCache:
    """Caches recent tool response data for context injection.

    Tool responses often carry structured data (ids, arrays) that the LLM needs
    for a follow-up call. That data is preserved apart from the chat history and
    injected into context on each LLM call.

    Connected-account data is refused outright. Cached, it would be injected
    into the context of every later turn -- including one that goes to the
    Hermes runtime, which must never see it. The local model still gets that
    result where it actually needs it: in the tool message of the turn that read
    it, which is what it composes its answer from.
    """

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._cache: list[dict] = []

    def add(self, tool_name: str, data: Any) -> bool:
        """Add tool response data to the cache; report whether it was kept."""
        if is_knowledge_tool(tool_name):
            logger.debug("Connected-account data is not cached past its own turn")
            return False
        entry = dict(tool=tool_name, data=data, timestamp=time.time())
        self._cache.append(entry)
        if len(self._cache) > self.max_entries:
            self._cache.pop(0)  # Remove oldest
        return True

    def get_context_message(self) -> str | None:
        """Format cached data as context string for LLM injection.

        One entry per line: the redaction barrier reads this block line by line
        if it ever meets one it did not build.
        """
        if not self._cache:
            return None
        parts = [TOOL_DATA_HEADER]
        for entry in self._cache:
            parts.append(entry["tool"] + ": " + json.dumps(entry["data"]))
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


async def llm_node(
    agent,
    chat_ctx,
    provider: LLMProvider,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> AsyncIterable[str]:
    """Provider-agnostic LLM node with tool calling support.

    This function should be called from an Agent's llm_node method override.

    Args:
        agent: The Agent instance (self)
        chat_ctx: Chat context from LiveKit
        provider: LLMProvider instance (OllamaProvider, GroqProvider, etc.)
        tool_data_cache: Cache for structured tool response data
        max_turns: Max conversation turns to keep in sliding window

    Yields:
        String chunks for TTS output

    Example:
        class MyAgent(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                async for chunk in llm_node(
                    self, chat_ctx, provider=self._provider
                ):
                    yield chunk
    """
    try:
        # Build messages from chat context with sliding window
        messages = _build_messages_from_context(
            chat_ctx,
            tool_data_cache=tool_data_cache,
            max_turns=max_turns,
        )

        # A provider that runs its own tool loop is a separate agent runtime
        # with its own model: everything it is sent crosses the redaction
        # barrier first, so no connected-account result, account label or
        # answer composed from either travels with the turn.
        if provider.manages_own_tools:
            messages = sanitize_for_escalation(messages)

        # Discover tools from agent and MCP servers. Providers that run their
        # own tool loop (Hermes) never receive these schemas, so building the
        # catalog would only add latency to every turn.
        if provider.manages_own_tools:
            tools = None
        else:
            tools = await _discover_tools(agent)

        # If tools available, check for tool calls first (non-streaming)
        if tools:
            response = await provider.chat(messages=messages, tools=tools)

            if response.tool_calls:
                logger.info(
                    f"LLM returned {len(response.tool_calls)} tool call(s): "
                    f"{[tc.name for tc in response.tool_calls]}"
                )

                # Track tool usage for frontend indicator
                tool_names = [tc.name for tc in response.tool_calls]
                tool_params = [tc.arguments for tc in response.tool_calls]

                # Publish tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    import asyncio

                    asyncio.create_task(agent._on_tool_status(True, tool_names, tool_params))

                # Execute tools and get results (cache structured data)
                # Also track hass_assist results for direct speech
                hass_results = []
                messages, hass_results = await _execute_tool_calls(
                    agent,
                    messages,
                    response.tool_calls,
                    response.content,
                    provider=provider,
                    tool_data_cache=tool_data_cache,
                )

                # If hass_assist was called, speak its response directly
                # (bypasses LLM follow-up which tends to summarize/paraphrase)
                if hass_results:
                    combined = " ".join(hass_results)
                    logger.info(f"Speaking hass_assist response directly: {combined[:100]}...")
                    yield strip_markdown_for_tts(combined)
                    return

                # Stream follow-up response with tool results
                # Pass tools so Ollama can validate tool_calls in message history
                logger.info("Streaming follow-up response from LLM...")
                chunk_count = 0
                # An answer composed from a connected-account result is that
                # same data in sentences, and it returns on later turns as
                # ordinary assistant transcript. It is recorded as a salted
                # hash so the barrier can recognise it then; the words
                # themselves are neither stored nor logged.
                private_turn = any(
                    _keeps_contents_private(call.name) for call in response.tool_calls
                )
                spoken: list[str] = []
                async for chunk in provider.chat_stream(messages=messages, tools=tools):
                    chunk_count += 1
                    text = strip_markdown_for_tts(chunk)
                    if private_turn:
                        spoken.append(text)
                    yield text
                if private_turn:
                    record_private_answer("".join(spoken))
                # The answer itself is never logged: a follow-up to a knowledge
                # tool is the user own mail and calendar, spoken back.
                logger.info(f"Follow-up complete: {chunk_count} chunks")
                return

            # No tool calls - return content directly
            elif response.content:
                # Publish no-tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    import asyncio

                    asyncio.create_task(agent._on_tool_status(False, [], []))
                yield strip_markdown_for_tts(response.content)
                return

        # No tools or no tool calls - stream directly
        # Publish no-tool status immediately
        if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
            import asyncio

            asyncio.create_task(agent._on_tool_status(False, [], []))

        async for chunk in provider.chat_stream(messages=messages):
            yield strip_markdown_for_tts(chunk)

    except Exception as e:
        logger.error(f"Error in llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _build_messages_from_context(
    chat_ctx,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> list[dict]:
    """Build messages with sliding window and tool data context.

    Message order:
    1. System prompt (always first, never trimmed)
    2. Tool data context (injected from cache)
    3. Chat history (sliding window applied)

    Args:
        chat_ctx: LiveKit chat context
        tool_data_cache: Cache of recent tool response data
        max_turns: Max conversation turns to keep (1 turn = user + assistant)
    """
    # Every system message is kept, in order, and merged into the single
    # leading system prompt. A later system message (e.g. the private handoff
    # continuation preamble) must add to the agent prompt, never replace it.
    system_parts: list[str] = []
    chat_messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                if item.text_content:
                    system_parts.append(item.text_content)
            else:
                chat_messages.append(msg)
        elif item_type == "FunctionCall":
            try:
                # Arguments must be JSON string for Groq compatibility
                args = getattr(item, "arguments", {}) or {}
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": args_str,
                                },
                            }
                        ],
                    }
                )
            except AttributeError:
                pass
        elif item_type == "FunctionCallOutput":
            try:
                chat_messages.append(
                    {
                        "role": "tool",
                        "content": str(item.content),
                        "tool_call_id": item.tool_call_id,
                    }
                )
            except AttributeError:
                pass

    # Build final message list
    messages = []

    # 1. System prompt always first
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # 2. Inject tool data context
    if tool_data_cache:
        context = tool_data_cache.get_context_message()
        if context:
            messages.append({"role": "system", "content": context})

    # 3. Apply sliding window to chat history
    # max_turns * 2 accounts for user + assistant pairs
    max_messages = max_turns * 2
    if len(chat_messages) > max_messages:
        trimmed = len(chat_messages) - max_messages
        chat_messages = chat_messages[-max_messages:]
        logger.debug(f"Sliding window: trimmed {trimmed} old messages")

    messages.extend(chat_messages)
    return messages


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools from agent methods and MCP servers.

    Tools are cached on the agent instance after first discovery to avoid
    redundant MCP API calls on every user utterance.
    """
    # Return cached tools if available
    if hasattr(agent, "_llm_tools_cache") and agent._llm_tools_cache is not None:
        return agent._llm_tools_cache

    tools = []

    if settings_module.get_setting("native_tools_enabled", True):
        native_registry = create_default_registry()
        tools.extend(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in native_registry.list()
        )
        agent._native_tool_registry = native_registry
        logger.info(f"Added {len(native_registry.names())} native assistant tools")

    # Get @function_tool decorated methods from agent (bound methods on class)
    if hasattr(agent, "_tools") and agent._tools:
        for tool in agent._tools:
            if hasattr(tool, "__func__"):
                func = tool.__func__
                name = func.__name__
                description = func.__doc__ or ""
                sig = inspect.signature(func)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_type = "string"
                    if param.annotation is not inspect.Parameter.empty:
                        if param.annotation is str:
                            param_type = "string"
                        elif param.annotation is int:
                            param_type = "integer"
                        elif param.annotation is float:
                            param_type = "number"
                        elif param.annotation is bool:
                            param_type = "boolean"
                    properties[param_name] = {"type": param_type}
                    if param.default is inspect.Parameter.empty and param_name != "self":
                        required.append(param_name)

                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            },
                        },
                    }
                )

    # Get MCP tools from all configured servers (except n8n and home_assistant)
    # n8n uses webhook-based workflow discovery, not direct MCP tools
    # home_assistant uses wrapper tools (hass_control, hass_get_state) for simpler LLM interface
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        for server_name, server in agent._caal_mcp_servers.items():
            # Skip servers that use wrapper tools instead of raw MCP tools
            if server_name in ("n8n", "home_assistant"):
                continue

            mcp_tools = await _get_mcp_tools(server)
            # Prefix tools with server name to avoid collisions
            for tool in mcp_tools:
                original_name = tool["function"]["name"]
                tool["function"]["name"] = f"{server_name}__{original_name}"
            tools.extend(mcp_tools)
            if mcp_tools:
                logger.info(f"Added {len(mcp_tools)} tools from MCP server: {server_name}")

    # Add n8n workflow tools (webhook-based execution, separate from MCP)
    if hasattr(agent, "_n8n_workflow_tools") and agent._n8n_workflow_tools:
        tools.extend(agent._n8n_workflow_tools)

    # Add Home Assistant tools (only if HASS is connected)
    if hasattr(agent, "_hass_tool_definitions") and agent._hass_tool_definitions:
        tools.extend(agent._hass_tool_definitions)
        logger.info(f"Added {len(agent._hass_tool_definitions)} HASS tools")
    else:
        logger.debug(
            f"No HASS tools: has_attr={hasattr(agent, '_hass_tool_definitions')}, "
            f"value={getattr(agent, '_hass_tool_definitions', None)}"
        )

    # Add Friday assistant tools (Clawdbot)
    if hasattr(agent, "_friday_tool_definitions") and agent._friday_tool_definitions:
        tools.extend(agent._friday_tool_definitions)
        logger.info(f"Added {len(agent._friday_tool_definitions)} Friday tools")

    # Log all discovered tools
    if tools:
        tool_names = [t["function"]["name"] for t in tools]
        logger.info(f"Discovered {len(tools)} tools: {tool_names}")

    # Cache tools on agent and return
    result = tools if tools else None
    agent._llm_tools_cache = result

    return result


async def _get_mcp_tools(mcp_server) -> list[dict]:
    """Get tools from an MCP server in OpenAI format."""
    tools = []

    if not mcp_server or not hasattr(mcp_server, "_client") or not mcp_server._client:
        return tools

    try:
        tools_result = await mcp_server._client.list_tools()
        if hasattr(tools_result, "tools"):
            for mcp_tool in tools_result.tools:
                # Convert MCP schema to OpenAI format
                parameters = {"type": "object", "properties": {}, "required": []}
                if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                    schema = mcp_tool.inputSchema
                    if isinstance(schema, dict):
                        parameters = schema.copy()
                    elif hasattr(schema, "properties"):
                        parameters["properties"] = schema.properties or {}
                        parameters["required"] = getattr(schema, "required", []) or []

                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mcp_tool.name,
                            "description": getattr(mcp_tool, "description", "") or "",
                            "parameters": parameters,
                        },
                    }
                )

        # Don't log here - caller logs the summary

    except Exception as e:
        logger.warning(f"Error getting MCP tools: {e}")

    return tools


async def _execute_tool_calls(
    agent,
    messages: list[dict],
    tool_calls: list["ToolCall"],
    response_content: str | None,
    provider: LLMProvider,
    tool_data_cache: ToolDataCache | None = None,
) -> tuple[list[dict], list[str]]:
    """Execute tool calls and append results to messages.

    Args:
        agent: The agent instance
        messages: Current message list to append to
        tool_calls: List of normalized ToolCall objects
        response_content: Original LLM response content (if any)
        provider: LLM provider (for formatting tool results)
        tool_data_cache: Optional cache to store structured tool response data

    Returns:
        tuple: (updated messages, list of hass_assist results for direct speech)
    """
    logger.info(f"_execute_tool_calls: Starting with {len(tool_calls)} tool(s)")
    hass_results: list[str] = []

    # Add assistant message with tool calls
    tool_call_message = provider.format_tool_call_message(
        content=response_content,
        tool_calls=tool_calls,
    )
    messages.append(tool_call_message)

    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.name
        arguments = tool_call.arguments
        # A model-chosen argument is the words of the user in another shape: a
        # search phrase, a reminder title, an address. The log names the tool
        # and its argument *names*, never a value, for every tool.
        logger.info(f"Executing tool: {tool_name} (argument names: {sorted(arguments)})")

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)
            if _keeps_contents_private(tool_name) or isinstance(tool_result, dict):
                status = tool_result.get("status") if isinstance(tool_result, dict) else None
                logger.info(f"Tool {tool_name} returned status={status}")
            else:
                logger.info(f"Tool {tool_name} returned {type(tool_result).__name__}")

            # Capture hass_assist results for direct speech
            if tool_name == "hass_assist" and isinstance(tool_result, str):
                hass_results.append(tool_result)

            # Cache structured data if present
            if tool_data_cache and isinstance(tool_result, dict):
                # Look for common data fields, otherwise cache the whole result
                data = tool_result.get("data") or tool_result.get("results") or tool_result
                tool_data_cache.add(tool_name, data)
                logger.debug(f"Cached tool data for {tool_name}")

            # Format tool result - preserve JSON structure for LLM
            if isinstance(tool_result, dict):
                result_content = json.dumps(tool_result)
            else:
                result_content = str(tool_result)

            result_message = provider.format_tool_result(
                content=result_content,
                tool_call_id=tool_call.id,
                tool_name=tool_name,
            )
            messages.append(result_message)

        except Exception as e:
            # The model is told what to say, not what went wrong: a traceback, a
            # provider detail or an echo of a private argument must never become
            # part of a spoken reply.
            if isinstance(e, SafeToolError):
                logger.info(f"Tool {tool_name} refused the request")
                safe = {"status": "invalid_request", "message": str(e), "data": {}}
            else:
                logger.error(f"Tool {tool_name} failed: {type(e).__name__}", exc_info=True)
                safe = {
                    "status": "error",
                    "message": (
                        "That did not go through on my end. Tell the user it failed, say "
                        "nothing about why, and offer to try again."
                    ),
                    "data": {},
                }
            result_message = provider.format_tool_result(
                content=json.dumps(safe),
                tool_call_id=tool_call.id,
                tool_name=tool_name,
            )
            messages.append(result_message)

    return messages, hass_results


# Local tools whose arguments and results are the private words of the user: a
# reminder title, an alarm label, a note. They are not connected-account
# knowledge, so they do not cross the Hermes barrier, but they do not belong in
# the log either, and neither do the row ids they carry.
PRIVATE_LOCAL_TOOLS = frozenset(
    {"alarms.set", "reminders.create", "reminders.list", "reminders.set_delivery"}
)


def _keeps_contents_private(tool_name: str) -> bool:
    """Whether a tool arguments and results stay out of the log.

    Read from the tool catalog rather than from the agent, so a session that
    never built its own registry still treats connected-account data as private.
    """
    return is_knowledge_tool(tool_name) or tool_name in PRIVATE_LOCAL_TOOLS


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call.

    Routing priority:
    1. Home Assistant tools (callable dict)
    2. Agent methods (@function_tool decorated on class)
    3. n8n workflows (webhook-based execution)
    4. MCP servers (with server_name__tool_name prefix parsing)
    """
    logger.debug(
        f"Looking up tool '{tool_name}': "
        f"hass_callables={list(getattr(agent, '_hass_tool_callables', {}).keys())}"
    )

    # Check native assistant tools first (email/calendar/reminders/alarms registry)
    native_registry = getattr(agent, "_native_tool_registry", None)
    if native_registry is None and settings_module.get_setting("native_tools_enabled", True):
        native_registry = create_default_registry()
        agent._native_tool_registry = native_registry
    if native_registry is not None and tool_name in native_registry.names():
        logger.info(f"Calling native tool: {tool_name}")
        tool = native_registry.get(tool_name)
        # The model never chooses whose data a tool touches: user-scoped tools
        # are bound to the session's verified scope, and an unidentified
        # session under multi-user is refused before any store is opened.
        bound = scoped_tool_arguments(tool, arguments, getattr(agent, "_user_scope", None))
        if bound is None:
            logger.info(f"Refused user-scoped tool {tool_name} for an unidentified session")
            if tool.category == "knowledge":
                return session_unavailable_result()
            if tool.category in ("alarms", "reminders"):
                return scheduling_unavailable_result()
            return memory_unavailable_result()
        result = tool.handler(**bound)
        if inspect.isawaitable(result):
            # Knowledge tools read a bounded index and may refresh it first.
            result = await result
        logger.info(f"Native tool {tool_name} completed")
        return result

    # Check Home Assistant tools (callable functions stored in dict)
    if hasattr(agent, "_hass_tool_callables") and tool_name in agent._hass_tool_callables:
        logger.info(f"Calling HASS tool: {tool_name}")
        result = await agent._hass_tool_callables[tool_name](**arguments)
        logger.info(f"HASS tool {tool_name} completed")
        return result

    # Check Friday assistant tools (Clawdbot)
    if hasattr(agent, "_friday_tool_callables") and tool_name in agent._friday_tool_callables:
        logger.info(f"Calling Friday tool: {tool_name}")
        result = await agent._friday_tool_callables[tool_name](**arguments)
        logger.info(f"Friday tool {tool_name} completed")
        return result

    # Check if it's an allowlisted agent method (decorated on class)
    agent_tool = resolve_agent_method_tool(agent, tool_name)
    if agent_tool is not None:
        logger.info(f"Calling agent tool: {tool_name}")
        result = await agent_tool(**arguments)
        logger.info(f"Agent tool {tool_name} completed")
        return result

    # Check if it's an n8n workflow
    if (
        hasattr(agent, "_n8n_workflow_name_map")
        and tool_name in agent._n8n_workflow_name_map
        and hasattr(agent, "_n8n_base_url")
        and agent._n8n_base_url
    ):
        logger.info(f"Calling n8n workflow: {tool_name}")
        workflow_name = agent._n8n_workflow_name_map[tool_name]
        result = await execute_n8n_workflow(agent._n8n_base_url, workflow_name, arguments)
        logger.info(f"n8n workflow {tool_name} completed")
        return result

    # Check MCP servers (with multi-server routing)
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        # Parse server name from prefixed tool name
        # Format: server_name__actual_tool (double underscore separator)
        if "__" in tool_name:
            server_name, actual_tool = tool_name.split("__", 1)
        else:
            # Unprefixed tools default to n8n server
            server_name, actual_tool = "n8n", tool_name

        if server_name in agent._caal_mcp_servers:
            server = agent._caal_mcp_servers[server_name]
            result = await _call_mcp_tool(server, actual_tool, arguments)
            if result is not None:
                return result

    raise ValueError(f"Tool {tool_name} not found")


async def _call_mcp_tool(mcp_server, tool_name: str, arguments: dict) -> Any | None:
    """Call a tool on an MCP server.

    Calls the tool directly without checking if it exists first - the MCP
    server will return an error if the tool doesn't exist.
    """
    if not mcp_server or not hasattr(mcp_server, "_client"):
        return None

    try:
        logger.info(f"Calling MCP tool: {tool_name}")
        result = await mcp_server._client.call_tool(tool_name, arguments)

        # Check for errors
        if result.isError:
            text_contents = []
            for content in result.content:
                if hasattr(content, "text") and content.text:
                    text_contents.append(content.text)
            error_msg = f"MCP tool {tool_name} error: {text_contents}"
            logger.error(error_msg)
            return error_msg

        # Extract text content
        text_contents = []
        for content in result.content:
            if hasattr(content, "text") and content.text:
                text_contents.append(content.text)

        return "\n".join(text_contents) if text_contents else "Tool executed successfully"

    except Exception as e:
        logger.warning(f"Error calling MCP tool {tool_name}: {e}")

    return None
