"""Production asyncio concurrency gates for agent and evaluator work."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncSemaphoreResourceManager:
    def __init__(self, *, agent_concurrency: int, evaluation_concurrency: int) -> None:
        for name, value in (
            ("agent_concurrency", agent_concurrency),
            ("evaluation_concurrency", evaluation_concurrency),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._agents = asyncio.Semaphore(agent_concurrency)
        self._evaluations = asyncio.Semaphore(evaluation_concurrency)
        self.active_agents = 0
        self.active_evaluations = 0

    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        async with self._agents:
            self.active_agents += 1
            try:
                yield
            finally:
                self.active_agents -= 1

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        async with self._evaluations:
            self.active_evaluations += 1
            try:
                yield
            finally:
                self.active_evaluations -= 1


__all__ = ["AsyncSemaphoreResourceManager"]
