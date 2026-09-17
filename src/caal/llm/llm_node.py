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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from caal import company_privacy, scheduled_events
from caal import settings as settings_module
from caal.tools import create_default_registry, natural_schedule
from caal.tools.arguments import validate_tool_arguments
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
from .scheduled_reply import spoken_outcome

if TYPE_CHECKING:
    from .providers import ToolCall

logger = logging.getLogger(__name__)

__all__ = ["llm_node", "ToolDataCache", "ToolOutcomes"]


@dataclass
class ToolOutcomes:
    """What the executed tools of one turn can say without a model.

    Two kinds, kept apart because they are answered differently. ``hass`` is
    the spoken response Home Assistant already produced. ``scheduled`` is the
    outcome of a reminder or alarm call, in the order the model asked for them;
    see :mod:`caal.llm.scheduled_reply` for why it is spoken rather than
    narrated.
    """

    hass: list[str] = field(default_factory=list)
    scheduled: list[str] = field(default_factory=list)


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
        if tool_name in PRIVATE_LOCAL_TOOLS:
            # A reminder title, an alarm label and the row ids they carry are
            # the private words of this person. They are answered directly in
            # their own turn and are not injected into any later one.
            logger.debug("Scheduled-item data is not cached past its own turn")
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


def _with_directive(messages: list[dict], directive: str) -> list[dict]:
    """Add one session instruction to the prompt without replacing any of it."""
    updated = [dict(message) for message in messages]
    if updated and updated[0].get("role") == "system":
        updated[0]["content"] = f"{updated[0].get('content', '')}\n\n{directive}".strip()
    else:
        updated.insert(0, {"role": "system", "content": directive})
    return updated


def _language_directive(agent) -> str:
    """The reply-language instruction for this turn, empty for English."""
    session = getattr(agent, "_language_session", None)
    return session.directive() if session is not None else ""


async def llm_node(
    agent,
    chat_ctx,
    provider: LLMProvider,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
    reasoning: bool | None = None,
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
    request_options = {"think": reasoning} if type(reasoning) is bool else {}
    try:
        # Build messages from chat context with sliding window
        messages = _build_messages_from_context(
            chat_ctx,
            tool_data_cache=tool_data_cache,
            max_turns=max_turns,
        )

        # Before anything is routed, discovered or escalated: decide whether
        # this session may leave the machine at all. A company-private session
        # is decided from the signed-in user and the configured policy, never
        # from the words of the turn, and it never lifts once engaged.
        company_privacy.begin_turn(agent, _latest_user_text(messages))

        # A company-private session tells the model that the owner's library is
        # open to it. Only the model that chooses the tools is told: a provider
        # with its own tool loop is the runtime a private session never
        # reaches, and it is not offered the company tools either.
        if not provider.manages_own_tools:
            directive = company_privacy.private_session_directive(agent)
            if directive:
                messages = _with_directive(messages, directive)

        # Which language to answer this turn in. Resolved from the verified
        # user's own preference and the speech server's detection before the
        # turn reached here (see caal.language_policy); English resolves to an
        # empty directive, so an English session's prompt is unchanged.
        from caal.language_policy import reply_directive
        from caal.speech_request import turn_language

        language = turn_language(agent, chat_ctx)
        language_directive = reply_directive(language)
        if language_directive:
            messages = _with_directive(messages, language_directive)

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
            if any(t["function"]["name"] == "time.current" for t in tools or []):
                messages = _with_directive(
                    messages,
                    "For every current time, current date, or time follow-up question, call "
                    "time.current before answering. It reads the clock at this exact turn in the "
                    "configured local timezone. Never answer from a time mentioned earlier in "
                    "the session, including a system-prompt session-start timestamp.",
                )
            if any(t["function"]["name"] == "weather.current" for t in tools or []):
                messages = _with_directive(
                    messages,
                    "For current weather questions and weather follow-up questions, call "
                    "weather.current before answering. The tool includes rain, gusts, humidity "
                    "and feels-like readings even when an earlier spoken answer omitted them. "
                    "Read it again for follow-ups; do not claim a measurement is unavailable "
                    "without checking. Use its freshness state and conditions faithfully. "
                    "It has current observations only, not future forecasts. Answer in the "
                    "conversation language. Never infer a location or supply one to this tool.",
                )

        if getattr(agent, "_satellite_restricted", False):
            # The next model response may resolve a read into another tool call.
            # SatelliteModel's text stream deliberately refuses tool frames.
            corrected = False
            light_result = None
            for _ in range(6):
                response = await provider.chat(messages=messages, tools=tools, **request_options)
                if _invalid_tool_batch(agent, response.tool_calls, tools):
                    if corrected:
                        break
                    corrected = True
                    messages.append({
                        "role": "system",
                        "content": "The proposed batch was rejected before execution. "
                        "Use only the advertised tools and valid arguments. "
                        "Do not replay any earlier successful action.",
                    })
                    continue
                if response.tool_calls:
                    messages, _ = await _execute_tool_calls(
                        agent, messages, response.tool_calls, response.content, provider,
                    )
                    results = messages[-len(response.tool_calls):]
                    for call, result in zip(response.tool_calls, results):
                        if call.name == "home.light":
                            light_result = json.loads(result["content"])
                    continue
                if response.content and response.content.strip():
                    # A model must not turn a refusal or an accepted request into
                    # an unsupported claim about the physical lamp.
                    content = (
                        light_result.get("spoken_message", light_result["message"])
                        if light_result else response.content
                    )
                    yield strip_markdown_for_tts(content)
                    return
                break
            yield (
                light_result.get("spoken_message", light_result["message"])
                if light_result else
                "I couldn't complete that request. No further light requests will be sent."
            )
            return

        # An external tool's model-written arguments cross the same boundary
        # as a Hermes prompt. Remove prior private results before that model
        # can paraphrase them into arguments. Current native tool results still
        # go to the local answer stream below.
        if tools and any(_is_unscoped_tool(agent, tool["function"]["name"]) for tool in tools):
            messages = sanitize_for_escalation(messages)

        # If tools available, check for tool calls first (non-streaming)
        if tools:
            response = await provider.chat(messages=messages, tools=tools, **request_options)

            # Validate the entire batch before running any member. A correction
            # must never replay a mutation that ran beside an unknown tool.
            if _invalid_tool_batch(agent, response.tool_calls, tools):
                # No call ran, so there is no tool result to narrate. Inserting
                # a rejected call/result here makes some local models apologize
                # instead of selecting a supported tool. Re-evaluate the intact
                # conversation once, keeping the failed batch out of history.
                correction = (
                    "The previous proposed tool batch was rejected before execution. "
                    "No tools have run in this turn. Re-evaluate the user's request "
                    "using only exact tool names and arguments from the available schemas. "
                    "Preserve the requested account and time scope, including on a retry. "
                    "Use a supported tool when needed; do not claim a lookup succeeded "
                    "without its result."
                )
                messages = [dict(message) for message in messages]
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += "\n\n" + correction
                else:
                    messages.insert(0, {"role": "system", "content": correction})
                response = await provider.chat(messages=messages, tools=tools, **request_options)
                if _invalid_tool_batch(agent, response.tool_calls, tools):
                    yield "I couldn't complete that lookup. Please try again in a moment."
                    return

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

                # A real speed test runs the site's own full measurement and can
                # take a couple of minutes. Say so once, before the wait starts,
                # so the silence is not mistaken for a hang. One line only.
                notice = speedtest_notice(response.tool_calls)
                if notice:
                    yield notice

                # Execute tools and get results (cache structured data), plus
                # whatever those tools can already say for themselves.
                # What this person just said is in scope for the length of
                # their own tool calls and no longer. A native scheduling call
                # that dropped the time can read it back from there; nothing
                # durable, nothing logged, and nothing that leaves the process.
                with natural_schedule.user_turn(_latest_user_text(messages)):
                    messages, outcomes = await _execute_tool_calls(
                        agent,
                        messages,
                        response.tool_calls,
                        response.content,
                        provider=provider,
                        tool_data_cache=tool_data_cache,
                    )

                # A single weather failure has an actionable fixed reply. Do not
                # depend on a second model call to explain missing location/access.
                if (
                    len(response.tool_calls) == 1
                    and response.tool_calls[0].name == "weather.current"
                ):
                    from caal.tools.weather_tools import spoken_failure

                    failure = spoken_failure(json.loads(messages[-1]["content"]), language)
                    if failure:
                        yield failure
                        return

                # A reminder or an alarm that was just written already has its
                # one sentence, question included, and the local model returns
                # nothing at all for a continuation of this shape. Speaking the
                # handler message ends the turn here: no follow-up stream, no
                # escalation behind it, and no scheduled item of this person
                # handed to a second runtime to be described.
                #
                # Bounded deliberately: a turn that mixed a scheduled call with
                # another tool is answered from the scheduled outcomes alone,
                # in call order. Streaming instead would mean handing the model
                # a history that contains the reminder, which is the thing this
                # path exists to avoid. The Home Assistant reply, which is
                # already spoken verbatim, is kept after them.
                if outcomes.scheduled:
                    combined = " ".join([*outcomes.scheduled, *outcomes.hass])
                    logger.info(f"Speaking {len(outcomes.scheduled)} scheduled outcome(s) directly")
                    spoken = strip_markdown_for_tts(combined)
                    if outcomes.hass:
                        record_private_answer(spoken)
                    yield spoken
                    return

                # If hass_assist was called, speak its response directly
                # (bypasses LLM follow-up which tends to summarize/paraphrase)
                if outcomes.hass:
                    combined = " ".join(outcomes.hass)
                    logger.info("Speaking the private Home Assistant result")
                    spoken = strip_markdown_for_tts(combined)
                    record_private_answer(spoken)
                    yield spoken
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
                async for chunk in provider.chat_stream(
                    messages=messages, tools=tools, **request_options
                ):
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

        async for chunk in provider.chat_stream(messages=messages, **request_options):
            yield strip_markdown_for_tts(chunk)

    except Exception as e:
        if getattr(agent, "_satellite_restricted", False):
            raise RuntimeError("satellite_model_unavailable") from None
        logger.error(f"Error in llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _latest_user_text(messages: list[dict]) -> str:
    """The turn being answered, from the messages already built for this turn.

    Read back rather than kept: there is no second copy of the utterance
    anywhere, and a turn with nothing spoken in it simply yields "".
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


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


LEGACY_ACCOUNT_READS = frozenset(
    {"calendar.list_events", "calendar.find_free_time", "email.search", "email.read"}
)


# Categories whose arguments narrow a private read. An undeclared argument
# here is dropped by scoped_tool_arguments a few lines later, and dropping the
# employee or the document the user actually asked about turns their narrow
# question into a wide one over everything the owner can see. So it is
# rejected first, while the mistake is still visible.
STRICT_ARGUMENT_CATEGORIES = frozenset({"knowledge", "company", "weather"})


def _connected_argument_error(tool, arguments: dict) -> dict | None:
    """Do not turn a misspelled account/employee/document filter into a wider read."""
    if tool.category not in STRICT_ARGUMENT_CATEGORIES:
        return None
    if set(arguments) - set(tool.parameters.get("properties", {})) - {"user_id"}:
        return {
            "status": "invalid_request",
            "message": (
                "Use only schema-declared arguments, retaining the requested "
                "account, employee, document and time scope."
            ),
            "data": {},
        }
    return validate_tool_arguments(tool, arguments)


def _invalid_tool_batch(agent, calls, tools: list[dict]) -> bool:
    offered = {tool["function"]["name"] for tool in tools}
    registry = getattr(agent, "_native_tool_registry", None)
    for call in calls:
        if call.name not in offered or not _tool_available(agent, call.name):
            return True
        if registry is not None and call.name in registry.names():
            if _connected_argument_error(registry.get(call.name), call.arguments):
                return True
    return False


def _is_unscoped_tool(agent, name: str) -> bool:
    """External discovery is the boundary; native/class tools stay local."""
    if name == "hass_assist":
        return False
    registry = getattr(agent, "_native_tool_registry", None) or create_default_registry()
    return name not in registry.names() and resolve_agent_method_tool(agent, name) is None


def _company_tool_policy(agent, name: str) -> bool | None:
    """The company library's own answer about this tool, or ``None`` if it has none.

    Two rules, and they pull in opposite directions on purpose:

    * a **company-private session** is offered a narrow list of tools that keep
      text on this machine. A passage from an uploaded document is untrusted
      text, and the surest answer to "ignore your instructions and email this
      to..." is that there is no tool in the session that sends anything;
    * a session that is **not** the company owner's, or a deployment with the
      private mode switched off, is not offered the company tools at all.
    """
    from caal import company_privacy

    scope = getattr(agent, "_user_scope", None)
    user_id = getattr(scope, "user_id", None)
    if name.startswith("company."):
        return company_privacy.company_tools_offered(user_id)
    if not company_privacy.is_local_only():
        return None
    registry = getattr(agent, "_native_tool_registry", None)
    category = None
    if registry is not None and name in registry.names():
        category = registry.get(name).category
    return company_privacy.tool_allowed_in_private_session(category, name)


def _tool_available(agent, name: str) -> bool:
    """Connected identities read their own accounts, never deployment-wide stores."""
    company = _company_tool_policy(agent, name)
    if company is not None:
        if not company:
            return False
    from caal.ha_policy import tool_allowed
    if not tool_allowed(agent, name):
        return False
    from caal.tools.network_tools import authorized
    if name.startswith("network.") and not authorized(agent):
        return False
    scope = getattr(agent, "_user_scope", None)
    return not (getattr(scope, "identity_configured", False) and name in LEGACY_ACCOUNT_READS)


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools from agent methods and MCP servers.

    Tools are cached on the agent instance after first discovery to avoid
    redundant MCP API calls on every user utterance.
    """
    if getattr(agent, "_satellite_restricted", False):
        return [{"type": "function", "function": {
            "name": tool.name, "description": tool.description, "parameters": tool.parameters,
        }} for tool in agent._native_tool_registry.list()]
    # Return cached tools if available
    if hasattr(agent, "_llm_tools_cache") and agent._llm_tools_cache is not None:
        return [t for t in agent._llm_tools_cache if _tool_available(agent, t["function"]["name"])]

    tools = []

    if settings_module.get_setting("native_tools_enabled", True):
        native_registry = create_default_registry()
        # Bound before the filter runs: _tool_available asks the registry what
        # category a tool is in, and a company-private session decides from it.
        agent._native_tool_registry = native_registry
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
            if _tool_available(agent, tool.name)
        )
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
    tools = [t for t in tools if _tool_available(agent, t["function"]["name"])]
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
) -> tuple[list[dict], ToolOutcomes]:
    """Execute tool calls and append results to messages.

    Args:
        agent: The agent instance
        messages: Current message list to append to
        tool_calls: List of normalized ToolCall objects
        response_content: Original LLM response content (if any)
        provider: LLM provider (for formatting tool results)
        tool_data_cache: Optional cache to store structured tool response data

    Returns:
        tuple: (updated messages, :class:`ToolOutcomes` for direct speech)
    """
    logger.info(f"_execute_tool_calls: Starting with {len(tool_calls)} tool(s)")
    outcomes = ToolOutcomes()

    # Add assistant message with tool calls
    tool_call_message = provider.format_tool_call_message(
        content=response_content,
        tool_calls=tool_calls,
    )
    messages.append(tool_call_message)

    # This flag lasts one batch only. A private result must not be followed by
    # an unscoped dispatch in that batch; a fresh sanitized turn stays usable.
    private_result_seen = False
    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.name
        arguments = tool_call.arguments
        # A model-chosen argument is the words of the user in another shape: a
        # search phrase, a reminder title, an address. The log names the tool
        # and its argument *names*, never a value, for every tool.
        logger.info(f"Executing tool: {tool_name} (argument names: {sorted(arguments)})")

        try:
            if private_result_seen and _is_unscoped_tool(agent, tool_name):
                tool_result = {
                    "status": "unsupported_tool",
                    "message": "Private account results cannot be forwarded to external tools.",
                    "data": {},
                }
            else:
                tool_result = await _execute_single_tool(agent, tool_name, arguments)
            private_result_seen = private_result_seen or _keeps_contents_private(tool_name)
            if _keeps_contents_private(tool_name) or isinstance(tool_result, dict):
                status = tool_result.get("status") if isinstance(tool_result, dict) else None
                logger.info(f"Tool {tool_name} returned status={status}")
            else:
                logger.info(f"Tool {tool_name} returned {type(tool_result).__name__}")

            # Capture hass_assist results for direct speech
            if tool_name == "hass_assist" and isinstance(tool_result, str):
                outcomes.hass.append(tool_result)

            # A scheduled item speaks for itself, in the order it was asked
            # for. Only the sentence is kept here; the result it came from is
            # not logged, not cached and not repeated.
            spoken = spoken_outcome(tool_name, tool_result)
            if spoken is not None:
                outcomes.scheduled.append(spoken)

            # A scheduled item that now exists has to reach the dashboard of
            # its own owner without waiting out a polling interval. What is
            # published is a constant; see caal.scheduled_events.
            if scheduled_events.is_scheduled_mutation(tool_name, tool_result):
                await scheduled_events.announce(agent)

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
            # A scheduled call that refused or failed is answered here too: the
            # refusal in the words the handler chose, a failure as one fixed
            # internal-error line that never claims anything was scheduled.
            spoken = spoken_outcome(tool_name, safe)
            if spoken is not None:
                outcomes.scheduled.append(spoken)
            result_message = provider.format_tool_result(
                content=json.dumps(safe),
                tool_call_id=tool_call.id,
                tool_name=tool_name,
            )
            messages.append(result_message)

    return messages, outcomes


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
    return (is_knowledge_tool(tool_name) or tool_name in PRIVATE_LOCAL_TOOLS
            or tool_name.startswith("network."))


def speedtest_notice(tool_calls) -> str | None:
    """One sentence before a measurement that genuinely takes minutes, or nothing.

    Deliberately not routine chatter: every other tool stays silent, and a batch
    that asks for the speed test twice still gets a single line.
    """
    if any(getattr(call, "name", None) == "network.speedtest" for call in tool_calls or ()):
        return "Running a full speed test now. That takes up to a couple of minutes."
    return None


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call.

    Routing priority:
    1. Home Assistant tools (callable dict)
    2. Agent methods (@function_tool decorated on class)
    3. n8n workflows (webhook-based execution)
    4. MCP servers (with server_name__tool_name prefix parsing)
    """
    from caal.tools.network_tools import authorized, denied
    if tool_name.startswith("network.") and not authorized(agent):
        return denied()
    if getattr(agent, "_satellite_restricted", False):
        registry = agent._native_tool_registry
        if tool_name not in registry.names() or not isinstance(arguments, dict):
            return {
                'status': 'unauthorized',
                'message': 'Unavailable on this satellite.',
                'data': {},
            }
        tool = registry.get(tool_name)
        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        if set(arguments) - set(properties) or not set(required) <= set(arguments):
            return {
                'status': 'unauthorized',
                'message': 'Invalid satellite tool arguments.',
                'data': {},
            }
        try:
            result = tool.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except (PermissionError, ValueError, TypeError) as error:
            if tool_name == "home.light" and error.args == ("light_not_available",):
                logger.info("Satellite tool refused reason_code=light_not_available")
                return {
                    'status': 'target_unavailable',
                    'reason_code': 'light_not_available',
                    'spoken_message': "I couldn't find that light in Home Assistant. "
                    "Which lamp do you mean?",
                    'message': 'That light ID is not in the current Home Assistant states. '
                    'No light request was sent. Read home.states, following next_offset '
                    'if needed, and use an exact returned light ID. Ask for clarification '
                    'if the requested lamp is ambiguous. Never invent an ID.',
                    'data': {},
                }
            permission_reasons = {
                "ha_connection_required", "ha_configuration_changed", "ha_identity_changed",
                "ha_connection_changed", "ha_credentials_changed", "ha_reconnect_required",
                "ha_permission_denied", "ha_registry_unavailable", "ha_access_denied",
                "satellite_home_denied", "satellite_registry_changed", "satellite_grant_changed",
                "satellite_revoked", "satellite_unauthorized", "satellite_binding_mismatch",
            }
            if isinstance(error, PermissionError):
                reason = (
                    str(error) if str(error) in permission_reasons
                    else "satellite_permission_denied"
                )
                status = "unauthorized"
                message = (
                    "This speaker's Home Assistant authorization could not be validated. "
                    "No successful light action was confirmed."
                )
                if reason == "ha_reconnect_required":
                    message = (
                        "Home Assistant rejected the connection credentials. "
                        "The connected account needs to reconnect. "
                        "No successful light action was confirmed."
                    )
                elif reason == "ha_permission_denied":
                    message = (
                        "Home Assistant denied this request for the connected account. "
                        "No successful light action was confirmed."
                    )
            elif error.args == ("invalid_arguments",) or isinstance(error, TypeError):
                reason, status = "invalid_arguments", "invalid_request"
                message = (
                    "The tool arguments are invalid. Use the advertised schema. "
                    "Only exact light IDs and turn_on or turn_off are supported; "
                    "switch actuation and generic services are unavailable."
                )
            else:
                reason, status = "ha_unavailable", "unavailable"
                message = "Home Assistant is unavailable. No successful action was confirmed."
            logger.info("Satellite tool refused reason_code=%s", reason)
            return {
                'status': status,
                'reason_code': reason,
                'message': message,
                'data': {},
            }
        except Exception:
            return {
                'status': 'unavailable',
                'message': 'Home Assistant is unavailable. No successful action was confirmed.',
                'data': {},
            }
    if not _tool_available(agent, tool_name):
        return {
            "status": "unsupported_tool",
            "message": (
                "This tool is unavailable for this account. Use the available scoped tools; "
                "unscoped delegation is not authorized."
            ),
            "data": {},
        }
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
        invalid = _connected_argument_error(tool, arguments)
        if invalid is not None:
            return invalid
        # The model never chooses whose data a tool touches: user-scoped tools
        # are bound to the session's verified scope, and an unidentified
        # session under multi-user is refused before any store is opened.
        bound = scoped_tool_arguments(tool, arguments, getattr(agent, "_user_scope", None))
        if bound is None:
            logger.info(f"Refused user-scoped tool {tool_name} for an unidentified session")
            if tool.category == "weather":
                from caal.tools.weather_tools import (
                    session_unavailable_result as weather_unavailable,
                )

                return weather_unavailable()
            if tool.category == "knowledge":
                return session_unavailable_result()
            if tool.category == "company":
                from caal.tools import company_tools

                return company_tools.session_unavailable_result()
            if tool.category in ("alarms", "reminders"):
                return scheduling_unavailable_result()
            return memory_unavailable_result()
        # The model writes these arguments, so the schema is enforced here
        # rather than left to Python: an omitted or mistyped argument becomes a
        # sentence the model can act on, never a TypeError inside the turn. Only
        # argument names are reported; a value carries what the user said.
        # A natural request carries its own schedule even when the model
        # drops it. Recovered from this turn alone, from words the user said,
        # and only when they name one unambiguous relative time.
        bound = natural_schedule.recovered_arguments(tool_name, bound)
        invalid = validate_tool_arguments(tool, bound)
        if invalid is not None:
            logger.info(f"Refused an incomplete call to {tool_name} before its handler")
            return invalid
        if tool.category == "company":
            # However this call was arrived at, a passage from the library is
            # now part of this conversation. It stays on this machine from
            # here, including on every follow-up turn.
            company_privacy.note_company_tool_use()
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
