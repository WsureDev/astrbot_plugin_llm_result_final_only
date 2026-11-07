"""Expose final-only delivery as a native AstrBot plugin setting."""

from __future__ import annotations

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .output_gate import REQUEST_KEY, OutputGate


class FinalOnlyPlugin(Star):
    """Discard intermediate output from marked built-in agent requests."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.gate = OutputGate()

    async def initialize(self) -> None:
        """Install the reversible output gate when AstrBot activates the plugin."""
        self.gate.install()

    @filter.on_llm_request()
    async def mark_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Mark the exact request covered by this plugin's enable switch.

        Args:
            event: Incoming event with AstrBot's plugin-scope filtering applied.
            req: Request that will initialize the built-in agent runner.
        """
        if self.config.get("enabled", True):
            event.set_extra(REQUEST_KEY, req)
        else:
            event.set_extra(REQUEST_KEY, None)

    async def terminate(self) -> None:
        """Restore the original runner method when disabled or reloaded."""
        self.gate.uninstall()
