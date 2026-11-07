"""Bridge AstrBot's missing intermediate-response hook without editing core files."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from functools import wraps
from typing import TypeAlias
from weakref import WeakKeyDictionary

from astrbot.core.agent.response import AgentResponse, AgentResponseData
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import ProviderRequest

REQUEST_KEY = "astrbot_plugin_llm_result_final_only.request"
SUPPRESSED_RESPONSES = frozenset(
    {"llm_result", "streaming_delta", "tool_call", "tool_call_result"}
)
Runner: TypeAlias = ToolLoopAgentRunner[AstrAgentContext]
Step: TypeAlias = Callable[[Runner], AsyncGenerator[AgentResponse, None]]


class OutputGate:
    """Wrap agent steps while keeping request state on their existing event."""

    def __init__(self) -> None:
        self.active = False
        self.original_step: Step | None = None
        self.installed_step: Step | None = None
        self.requests: WeakKeyDictionary[Runner, ProviderRequest] = WeakKeyDictionary()

    def install(self) -> None:
        """Install one wrapper; repeated installation is a no-op."""
        if self.installed_step is not None:
            return

        original_step = ToolLoopAgentRunner.step
        self.original_step = original_step
        self.active = True

        @wraps(original_step)
        async def final_only_step(
            runner: Runner,
        ) -> AsyncGenerator[AgentResponse, None]:
            # The runner is generic: unrelated contexts need not have an event.
            event = getattr(runner.run_context.context, "event", None)
            marked = event is not None and event.get_extra(REQUEST_KEY) is runner.req
            enabled = (
                self.active
                and self.installed_step is final_only_step
                and (marked or self.requests.get(runner) is runner.req)
            )
            if enabled:
                self.requests[runner] = runner.req

            async with aclosing(original_step(runner)) as responses:
                async for response in responses:
                    if enabled and response.type in SUPPRESSED_RESPONSES:
                        continue
                    yield response

            if not enabled or not runner.done() or runner.was_aborted():
                return

            final_response = runner.get_final_llm_resp()
            if final_response is None or final_response.role != "assistant":
                return

            chain = final_response.result_chain
            if chain is not None:
                chain = chain.derive(chain=list(chain.chain))
            elif final_response.completion_text:
                chain = MessageChain().message(final_response.completion_text)

            if chain is None or not chain.chain:
                return

            # Do not let result_decorate reinsert reasoning from the final hook.
            # The provider response and model history remain intact.
            event.set_extra("_llm_reasoning_content", None)
            yield AgentResponse(
                type="llm_result",
                data=AgentResponseData(chain=chain),
            )
            # AstrBot selects one lane: ordinary delivery consumes llm_result;
            # streaming delivery consumes streaming_delta. Keep both protocols.
            if runner.streaming:
                yield AgentResponse(
                    type="streaming_delta",
                    data=AgentResponseData(chain=chain),
                )

        self.installed_step = final_only_step
        ToolLoopAgentRunner.step = final_only_step

    def uninstall(self) -> None:
        """Restore our patch without overwriting a later third-party wrapper."""
        self.active = False
        if (
            self.installed_step is not None
            and ToolLoopAgentRunner.step is self.installed_step
            and self.original_step is not None
        ):
            ToolLoopAgentRunner.step = self.original_step
        self.installed_step = None
        self.original_step = None
        self.requests.clear()
