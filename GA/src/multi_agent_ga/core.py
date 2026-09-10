"""Deterministic generational GA; workers own variation and evaluation.

Genomes are opaque canonical strings (for example serialized edit programs). The core
never edits chemistry, invents a reward, or interprets an agent's explanation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class Individual:
    identity: str
    genome: str
    fitness: float
    feasible: bool
    evidence_json: str = '{}'

    def __post_init__(self):
        if not self.identity or not isinstance(self.identity, str) or not self.genome or not isinstance(self.genome, str):
            raise ValueError('identity and genome must be nonempty strings')
        if type(self.fitness) not in (int, float) or not math.isfinite(self.fitness):
            raise ValueError('fitness must be finite')
        if type(self.feasible) is not bool:
            raise ValueError('feasible must be boolean')
        evidence = json.loads(self.evidence_json, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite evidence')))
        if not isinstance(evidence, dict):
            raise ValueError('evidence must be a JSON object')
        object.__setattr__(self, 'evidence_json', _json(evidence))

    @property
    def evidence(self) -> dict:
        return json.loads(self.evidence_json)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 20
    elite_count: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.6
    mutation_probability: float = 0.8
    concurrency: int = 4
    seed: int = 42

    def __post_init__(self):
        for name in ('population_size', 'elite_count', 'tournament_size', 'concurrency', 'seed'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError('require 1 <= elite_count < population_size')
        if not 1 <= self.tournament_size <= self.population_size or self.concurrency < 1:
            raise ValueError('invalid tournament size or concurrency')
        for name in ('crossover_probability', 'mutation_probability'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a probability')


@dataclass(frozen=True)
class OffspringRequest:
    request_id: str
    generation: int
    slot: int
    parent: Individual
    donor: Individual | None
    mutate: bool
    seed: int


@dataclass(frozen=True)
class Generation:
    generation: int
    population: tuple[Individual, ...]
    attempted_requests: int = 0
    failed_requests: int = 0

    @property
    def best(self) -> Individual:
        return max(self.population, key=_rank)


def _rank(individual: Individual):
    # Maximization, feasibility first, deterministic tie break.
    return individual.feasible, individual.fitness, individual.identity


Worker = Callable[[OffspringRequest], Awaitable[Individual | None]]


class GeneticSearch:
    """Select parents, request offspring, preserve elites, save each generation.

An interrupted, uncommitted generation may reissue requests on resume. Workers
must deduplicate evaluations by request_id or chemical/protocol identity. The
checkpoint contract promises no repeat of *committed* generations, not exactly
once execution across a crash.
"""

    def __init__(self, config: GAConfig, worker: Worker, *, protocol_id: str):
        if not isinstance(protocol_id, str) or not protocol_id:
            raise ValueError('protocol_id must freeze the worker/evaluator configuration')
        self.config, self.worker, self.protocol_id = config, worker, protocol_id

    def _requests(self, state: Generation, identity: str):
        generation = state.generation + 1
        seed = int(hashlib.sha256(f'{identity}:{generation}'.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        def select():
            return max(rng.sample(state.population, self.config.tournament_size), key=_rank)
        requests = []
        for slot in range(self.config.population_size - self.config.elite_count):
            parent = select()
            donor = None
            if rng.random() < self.config.crossover_probability:
                alternatives = [item for item in state.population if item.genome != parent.genome]
                if alternatives:
                    donor = max(rng.sample(alternatives, min(len(alternatives), self.config.tournament_size)), key=_rank)
            # With identical founders, a two-parent crossover has no new material.
            mutate = donor is None or rng.random() < self.config.mutation_probability
            requests.append(OffspringRequest(f'{identity[:20]}-g{generation}-s{slot}',
                generation, slot, parent, donor, mutate, rng.getrandbits(63)))
        return requests

    async def run(self, founders: tuple[Individual, ...], *, generations: int, directory: Path) -> Generation:
        if type(generations) is not int or generations < 0:
            raise ValueError('generations must be nonnegative')
        if len(founders) != self.config.population_size or not all(isinstance(x, Individual) for x in founders):
            raise ValueError('founder population must match configuration')
        identity = hashlib.sha256(_json({'config': asdict(self.config), 'protocol': self.protocol_id,
                                       'founders': [asdict(x) for x in founders]}).encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(directory.glob('generation-*.json'))
        if paths:
            document = json.loads(paths[-1].read_text())
            if document['identity'] != identity or document['schema'] != 'multi-agent-ga:v1':
                raise ValueError('checkpoint identity does not match configuration/protocol/founders')
            state = Generation(document['generation'], tuple(Individual(**x) for x in document['population']),
                               document['attempted_requests'], document['failed_requests'])
            if len(state.population) != self.config.population_size:
                raise ValueError('checkpoint population size mismatch')
        else:
            state = Generation(0, tuple(founders))
            self._save(directory, identity, state, [])
        if state.generation > generations:
            raise ValueError('target is behind checkpoint')
        semaphore = asyncio.Semaphore(self.config.concurrency)
        while state.generation < generations:
            requests = self._requests(state, identity)
            async def evaluate(request):
                async with semaphore:
                    try:
                        child = await self.worker(request)
                        if child is not None and not isinstance(child, Individual):
                            raise TypeError('worker must return an evaluated Individual or None')
                        return child, None if child is not None else 'no_valid_offspring'
                    except Exception as error:
                        return None, f'{type(error).__name__}: {str(error)[:300]}'
            tasks = [asyncio.create_task(evaluate(request)) for request in requests]
            try:
                outcomes = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            ranked_parents = sorted(state.population, key=_rank, reverse=True)
            survivors = ranked_parents[:self.config.elite_count]
            seen = {item.genome for item in survivors}
            children = sorted((child for child, _ in outcomes if child is not None), key=_rank, reverse=True)
            for item in children + ranked_parents:
                if len(survivors) == self.config.population_size:
                    break
                if item.genome not in seen:
                    survivors.append(item)
                    seen.add(item.genome)
            # Lack of valid/unique children must not collapse the population.
            for item in ranked_parents:
                if len(survivors) == self.config.population_size:
                    break
                survivors.append(item)
            state = Generation(state.generation + 1, tuple(survivors),
                               state.attempted_requests + len(requests),
                               state.failed_requests + sum(child is None for child, _ in outcomes))
            events = [{'request': asdict(req), 'child': asdict(child) if child else None, 'error': error}
                      for req, (child, error) in zip(requests, outcomes, strict=True)]
            self._save(directory, identity, state, events)
        return state

    @staticmethod
    def _save(directory: Path, identity: str, state: Generation, events: list):
        document = {'schema': 'multi-agent-ga:v1', 'identity': identity, **asdict(state), 'events': events}
        target = directory / f'generation-{state.generation:08d}.json'
        # A complete file is published through a hard link; never overwrite history.
        import os
        import tempfile
        fd, name = tempfile.mkstemp(prefix='.generation-', dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(_json(document) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
