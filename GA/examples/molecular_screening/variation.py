"""Wiki-guided gene variation followed by external phenotype evaluation.

Live Codex/Wiki/MoleculeEditor/FLAME services are injected. This contract layer
does not launch scientific jobs by itself.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import random

from multi_agent_ga.core import Individual, OffspringRequest
from .genes import EditProgram, crossover


@dataclass(frozen=True)
class WikiMutation:
    program: EditProgram
    hypothesis: str
    evidence_references: tuple[str, ...]


@dataclass(frozen=True)
class EvaluatedPhenotype:
    program_identity: str
    smiles: str
    chemical_identity: str
    fitness: float
    feasible: bool
    artifact_reference: str
    evaluation_reference: str


Mutator = Callable[[EditProgram, OffspringRequest], Awaitable[WikiMutation]]
Expression = Callable[[EditProgram, OffspringRequest], Awaitable[EvaluatedPhenotype | None]]


class MolecularGeneWorker:
    def __init__(self, *, wiki_mutator: Mutator, express_and_evaluate: Expression):
        self.wiki_mutator = wiki_mutator
        self.express_and_evaluate = express_and_evaluate

    async def __call__(self, request: OffspringRequest) -> Individual | None:
        program = EditProgram.decode(request.parent.genome)
        if request.donor is not None:
            donor = EditProgram.decode(request.donor.genome)
            rng = random.Random(request.seed)
            program = crossover(program, donor, left_cut=rng.choice(program.cuts),
                                right_cut=rng.choice(donor.cuts))
        mutation = None
        if request.mutate:
            mutation = await self.wiki_mutator(program, request)
            if not isinstance(mutation, WikiMutation) or not mutation.hypothesis.strip() or not mutation.evidence_references:
                raise ValueError('Wiki mutation requires a hypothesis and retrieved evidence references')
            program = mutation.program
        phenotype = await self.express_and_evaluate(program, request)
        if phenotype is None:
            return None
        if phenotype.program_identity != program.identity:
            raise ValueError('evaluated phenotype belongs to another edit program')
        if not phenotype.artifact_reference or not phenotype.evaluation_reference:
            raise ValueError('phenotype requires structure and evaluator provenance')
        evidence = {
            'phenotype_smiles': phenotype.smiles,
            'chemical_identity': phenotype.chemical_identity,
            'artifact_reference': phenotype.artifact_reference,
            'evaluation_reference': phenotype.evaluation_reference,
            'parent_genome_identity': request.parent.identity,
            'donor_genome_identity': request.donor.identity if request.donor else None,
            'request_id': request.request_id,
            'hypothesis': mutation.hypothesis if mutation else None,
            'wiki_evidence': list(mutation.evidence_references) if mutation else [],
        }
        return Individual(program.identity, program.encode(), phenotype.fitness,
                          phenotype.feasible, json.dumps(evidence))


__all__ = ['WikiMutation', 'EvaluatedPhenotype', 'MolecularGeneWorker']
