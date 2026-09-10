"""Generational GA with worker-lineage mortality and explicit context renewal."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random

from .core import GAConfig, Individual, OffspringRequest, _rank
from .persistence import digest, publish, read_record, run_lock


@dataclass(frozen=True)
class DeathPolicy:
    window: int = 3
    tolerance: float = 1.0
    failure_limit: int = 3

    def __post_init__(self):
        if type(self.window) is not int or self.window < 1 or type(self.failure_limit) is not int or self.failure_limit < 1:
            raise ValueError('death windows must be positive integers')
        if type(self.tolerance) not in (int, float) or not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('death tolerance must be finite and nonnegative')

    def reason(self, distances):
        if len(distances) < self.window + 1:
            return None
        values = distances[-self.window-1:]
        if any(type(d) not in (int, float) or not math.isfinite(d) or d < 0 for d in values):
            raise ValueError('invalid target distances')
        if values[-1] == 0:
            return None
        if all(b > a for a, b in zip(values, values[1:])) and values[-1]-values[0] > self.tolerance:
            return 'moving_away'
        if min(values[1:]) >= values[0]-self.tolerance:
            return 'no_progress'
        return None


@dataclass(frozen=True)
class Member:
    slot: int
    lineage_id: str
    individual: Individual
    distances: tuple[float, ...]
    valid_edits: int = 0
    failures: int = 0
    birth_generation: int = 0


@dataclass(frozen=True)
class PopulationState:
    generation: int
    members: tuple[Member, ...]
    best: Individual
    attempted_requests: int = 0
    failed_requests: int = 0
    death_count: int = 0
    events: tuple[dict, ...] = ()

    @property
    def population(self):
        return tuple(member.individual for member in self.members)


class LifecycleSearch:
    """Worker lineage is distinct from selected genetic-parent ancestry.

    Each active lineage tests one offspring per generation. Selection chooses
    genetic parents; it does not reset that worker's observed distance history.
    Elitism may keep an old genotype, but cannot erase failed trial history or
    exempt its worker from death. The best-ever archive is separate and immutable.
    """
    def __init__(self, config: GAConfig, worker, *, protocol_id: str, death_policy: DeathPolicy = DeathPolicy()):
        if not protocol_id:
            raise ValueError('protocol_id is required')
        self.config, self.worker = config, worker
        self.protocol_id, self.death_policy = protocol_id, death_policy

    @staticmethod
    def _distance(individual):
        if not isinstance(individual, Individual) or individual.target_distance is None:
            raise ValueError('lifecycle search requires an externally evaluated target_distance')
        return individual.target_distance

    async def _reconcile(self, state):
        callback = getattr(self.worker, 'reconcile', None)
        if callback is not None:
            await callback(tuple(m.lineage_id for m in state.members))

    def _requests(self, state, run_id):
        generation = state.generation + 1
        rng = random.Random(int(digest([run_id, generation]), 16))
        individuals = state.population
        requests = []
        for member in sorted(state.members, key=lambda m: m.slot):
            fresh = member.valid_edits == 0
            parent = member.individual if fresh else max(rng.sample(individuals, self.config.tournament_size), key=_rank)
            donor = None
            if not fresh and rng.random() < self.config.crossover_probability:
                choices = [item for item in individuals if item.genome != parent.genome]
                if choices:
                    donor = max(rng.sample(choices, min(len(choices), self.config.tournament_size)), key=_rank)
            requests.append(OffspringRequest(
                f'{run_id[:20]}-g{generation}-s{member.slot}', generation, member.slot,
                parent, donor, donor is None or rng.random() < self.config.mutation_probability,
                rng.getrandbits(63), member.lineage_id, member.individual))
        return requests

    async def run(self, founders, *, generations: int, directory: Path):
        if type(generations) is not int or generations < 0:
            raise ValueError('generations must be nonnegative')
        founders = tuple(founders)
        if len(founders) != self.config.population_size:
            raise ValueError('founder count must match population')
        for item in founders:
            self._distance(item)
        identity = digest({'schema': 'ga-lifecycle:v1', 'config': asdict(self.config),
                           'death_policy': asdict(self.death_policy), 'protocol': self.protocol_id,
                           'founders': [asdict(x) for x in founders]})
        with run_lock(directory):
            paths = sorted(directory.glob('population-*.json'))
            state = None
            previous = None
            for index, path in enumerate(paths):
                record = read_record(path)
                if (path.name != f'population-{index:08d}.json' or record['identity'] != identity
                        or record['previous'] != previous or record['state']['generation'] != index):
                    raise ValueError('checkpoint identity or chain mismatch')
                state = self._decode(record['state'])
                previous = digest(record)
            if state is None:
                members = tuple(Member(i, f'{identity[:20]}-birth0-s{i}', x, (self._distance(x),))
                                for i, x in enumerate(founders))
                state = PopulationState(0, members, max(founders, key=_rank))
                previous = self._save(directory, identity, state, None)
            if state.generation > generations:
                raise ValueError('target is behind checkpoint')
            await self._reconcile(state)
            semaphore = asyncio.Semaphore(self.config.concurrency)
            while state.generation < generations:
                requests = self._requests(state, identity)
                async def attempt(request):
                    path = directory / 'trials' / f'{request.request_id}.json'
                    request_hash = digest(asdict(request))
                    if path.exists():
                        saved = read_record(path)
                        if saved['request_hash'] != request_hash:
                            raise ValueError('trial identity mismatch')
                        return Individual(**saved['child']) if saved['child'] is not None else None, saved['error']
                    async with semaphore:
                        try:
                            child = await self.worker(request)
                            if child is not None:
                                self._distance(child)
                            error = None if child else 'no_valid_edit'
                        except Exception as exc:
                            child, error = None, f'{type(exc).__name__}: {str(exc)[:500]}'
                        publish(path, {'request_hash': request_hash, 'request': asdict(request),
                                       'child': asdict(child) if child else None, 'error': error})
                        return child, error
                tasks = [asyncio.create_task(attempt(req)) for req in requests]
                try:
                    outcomes = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                elite_ids = {m.lineage_id for m in sorted(state.members, key=lambda m: _rank(m.individual), reverse=True)[:self.config.elite_count]}
                current = {m.slot: m for m in state.members}
                members, events = [], []
                best = state.best
                for request, (child, error) in zip(requests, outcomes, strict=True):
                    old = current[request.slot]
                    if child is not None:
                        best = max((best, child), key=_rank)
                        distances = (*old.distances, self._distance(child))[-self.death_policy.window-1:]
                        individual = max((old.individual, child), key=_rank) if old.lineage_id in elite_ids else child
                        member = replace(old, individual=individual, distances=distances,
                                         valid_edits=old.valid_edits+1, failures=0)
                        reason = self.death_policy.reason(distances)
                    else:
                        member = replace(old, failures=old.failures+1)
                        reason = 'execution_failures' if member.failures >= self.death_policy.failure_limit else None
                    if reason:
                        new_id = f'{identity[:20]}-birth{request.generation}-s{old.slot}'
                        events.append({'kind': 'death', 'reason': reason, 'lineage_id': old.lineage_id,
                                       'replacement_lineage_id': new_id, 'slot': old.slot,
                                       'distances': list(member.distances), 'valid_edits': member.valid_edits,
                                       'failures': member.failures})
                        founder = founders[old.slot]
                        member = Member(old.slot, new_id, founder, (self._distance(founder),),
                                        birth_generation=request.generation)
                    members.append(member)
                state = PopulationState(state.generation+1, tuple(members), best,
                    state.attempted_requests+len(requests),
                    state.failed_requests+sum(child is None for child, _ in outcomes),
                    state.death_count+len(events), tuple(events))
                previous = self._save(directory, identity, state, previous)
                await self._reconcile(state)
            return state

    def _decode(self, raw):
        members = tuple(Member(**{**m, 'individual': Individual(**m['individual']),
                                  'distances': tuple(m['distances'])}) for m in raw['members'])
        if sorted(m.slot for m in members) != list(range(self.config.population_size)) or len({m.lineage_id for m in members}) != len(members):
            raise ValueError('invalid checkpoint members')
        for m in members:
            self._distance(m.individual)
            if not m.distances or any(not math.isfinite(d) or d < 0 for d in m.distances):
                raise ValueError('invalid checkpoint history')
        return PopulationState(**{**raw, 'members': members, 'best': Individual(**raw['best']),
                                   'events': tuple(raw['events'])})

    @staticmethod
    def _save(directory, identity, state, previous):
        record = {'schema': 'ga-lifecycle:v1', 'identity': identity, 'previous': previous, 'state': asdict(state)}
        publish(directory / f'population-{state.generation:08d}.json', record)
        return digest(record)
