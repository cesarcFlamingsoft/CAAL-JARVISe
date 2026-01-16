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

from ..integrations.n8n import execute_n8n_workflow
from ..utils.formatting import strip_markdown_for_tts
from .providers import LLMProvider

if TYPE_CHECKING:
    from .providers import ToolCall

logger = logging.getLogger(__name__)

__all__ = ["llm_node", "ToolDataCache"]


class ToolDataCache:
    """Caches recent tool response data for context injection.

    Tool responses often contain structured data (IDs, arrays) that the LLM
    needs for follow-up calls. This cache preserves that data separately
    from chat history and injects it into context on each LLM call.
    """

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._cache: list[dict] = []

    def add(self, tool_name: str, data: Any) -> None:
        """Add tool response data to cache."""
        entry = {"tool": tool_name, "data": data, "timestamp": time.time()}
        self._cache.append(entry)
        if len(self._cache) > self.max_entries:
            self._cache.pop(0)  # Remove oldest

    def get_context_message(self) -> str | None:
        """Format cached data as context string for LLM injection."""
        if not self._cache:
            return None
        parts = ["Recent tool response data for reference:"]
        for entry in self._cache:
            parts.append(f"\n{entry['tool']}: {json.dumps(entry['data'])}")
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

        # Discover tools from agent and MCP servers
        tools = await _discover_tools(agent)

        # If tools available, check for tool calls first (non-streaming)
        if tools:
            response = await provider.chat(messages=messages, tools=tools)

            if response.tool_calls:
                logger.info(f"LLM returned {len(response.tool_calls)} tool call(s)")

                # Track tool usage for frontend indicator
                tool_names = [tc.name for tc in response.tool_calls]
                tool_params = [tc.arguments for tc in response.tool_calls]

                # Publish tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    import asyncio

                    asyncio.create_task(
                        agent._on_tool_status(True, tool_names, tool_params)
                    )

                # Execute tools and get results (cache structured data)
                messages = await _execute_tool_calls(
                    agent,
                    messages,
                    response.tool_calls,
                    response.content,
                    provider=provider,
                    tool_data_cache=tool_data_cache,
                )

                # Stream follow-up response with tool results
                # Pass tools so Ollama can validate tool_calls in message history
                async for chunk in provider.chat_stream(messages=messages, tools=tools):
                    yield strip_markdown_for_tts(chunk)
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
    system_prompt = None
    chat_messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                system_prompt = msg
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
    if system_prompt:
        messages.append(system_prompt)

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
) -> list[dict]:
    """Execute tool calls and append results to messages.

    Args:
        agent: The agent instance
        messages: Current message list to append to
        tool_calls: List of normalized ToolCall objects
        response_content: Original LLM response content (if any)
        provider: LLM provider (for formatting tool results)
        tool_data_cache: Optional cache to store structured tool response data
    """
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
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)

            # Cache structured data if present
            if tool_data_cache and isinstance(tool_result, dict):
                # Look for common data fields, otherwise cache the whole result
                data = (
                    tool_result.get("data")
                    or tool_result.get("results")
                    or tool_result
                )
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
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            result_message = provider.format_tool_result(
                content=error_msg,
                tool_call_id=tool_call.id,
                tool_name=tool_name,
            )
            messages.append(result_message)

    return messages


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call.

    Routing priority:
    1. Home Assistant tools (callable dict)
    2. Agent methods (@function_tool decorated on class)
    3. n8n workflows (webhook-based execution)
    4. MCP servers (with server_name__tool_name prefix parsing)
    """
    # Check Home Assistant tools (callable functions stored in dict)
    if hasattr(agent, "_hass_tool_callables") and tool_name in agent._hass_tool_callables:
        logger.info(f"Calling HASS tool: {tool_name}")
        result = await agent._hass_tool_callables[tool_name](**arguments)
        logger.info(f"HASS tool {tool_name} completed")
        return result

    # Check if it's an agent method (decorated on class)
    if hasattr(agent, tool_name) and callable(getattr(agent, tool_name)):
        logger.info(f"Calling agent tool: {tool_name}")
        result = await getattr(agent, tool_name)(**arguments)
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
        result = await execute_n8n_workflow(
            agent._n8n_base_url, workflow_name, arguments
        )
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
