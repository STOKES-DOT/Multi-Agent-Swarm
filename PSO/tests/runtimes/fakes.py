from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from multi_agent_pso.runtimes.local_codex import CodexTurnResult, CodexUsage


class FakeCodexThread:
    def __init__(self, provider_id: str, client: "FakeCodexClient") -> None:
        self.id = provider_id
        self.client = client
        self.results: list[CodexTurnResult | BaseException] = []

    async def run(self, prompt, *, cwd, output_schema, sandbox):
        self.client.run_calls.append((self.id, prompt, cwd, output_schema, sandbox))
        self.client.active += 1
        self.client.max_active = max(self.client.max_active, self.client.active)
        self.client.active_by_thread[self.id] = self.client.active_by_thread.get(self.id, 0) + 1
        self.client.max_by_thread[self.id] = max(
            self.client.max_by_thread.get(self.id, 0),
            self.client.active_by_thread[self.id],
        )
        try:
            if self.client.delay:
                await asyncio.sleep(self.client.delay)
            result = self.results.pop(0) if self.results else self.client.default_result
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.client.active -= 1
            self.client.active_by_thread[self.id] -= 1


class FakeCodexClient:
    def __init__(self) -> None:
        self.started: list[tuple[str, Path, str]] = []
        self.resumed: list[tuple[str, str, Path, str]] = []
        self.run_calls: list[tuple[object, ...]] = []
        self.threads: dict[str, FakeCodexThread] = {}
        self.next_id = 0
        self.provider_ids: list[str] = []
        self.close_calls = 0
        self.close_results: list[BaseException | None] = []
        self.close_delay = 0.0
        self.close_started = asyncio.Event()
        self.delay = 0.0
        self.start_delay = 0.0
        self.start_active = 0
        self.max_start_active = 0
        self.resume_error: BaseException | None = None
        self.active = 0
        self.max_active = 0
        self.active_by_thread: dict[str, int] = {}
        self.max_by_thread: dict[str, int] = {}
        self.default_result = CodexTurnResult(
            "{\"ok\":true}", CodexUsage(3, 5, 1), "turn-1", "completed", 12
        )

    async def thread_start(self, *, model, cwd, sandbox):
        self.start_active += 1
        self.max_start_active = max(self.max_start_active, self.start_active)
        try:
            self.started.append((model, cwd, sandbox))
            if self.start_delay:
                await asyncio.sleep(self.start_delay)
            provider_id = (
                self.provider_ids.pop(0)
                if self.provider_ids
                else f"provider-{self.next_id}"
            )
            self.next_id += 1
            thread = FakeCodexThread(provider_id, self)
            self.threads[provider_id] = thread
            return thread
        finally:
            self.start_active -= 1

    async def thread_resume(self, thread_id, *, model, cwd, sandbox):
        self.resumed.append((thread_id, model, cwd, sandbox))
        if self.resume_error is not None:
            raise self.resume_error
        return self.threads.setdefault(thread_id, FakeCodexThread(thread_id, self))

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        if self.close_results:
            result = self.close_results.pop(0)
            if result is not None:
                raise result
