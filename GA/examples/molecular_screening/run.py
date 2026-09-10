"""Run the GA example in mock mode, or with explicitly authorized live services."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE.parents[1] / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_agent_ga import GAConfig, Individual, DeathPolicy, LifecycleSearch
from multi_agent_ga.persistence import digest, publish, run_lock
from molecular_screening.genes import EditGene, EditProgram
from molecular_screening.reward import SpectralObjective


def parameters(config):
    ga = GAConfig(**{k: config[k] for k in ('population_size','elite_count','tournament_size',
        'crossover_probability','mutation_probability','concurrency','seed')})
    death = DeathPolicy(config['death_window'], config['death_tolerance_nm'], config['execution_failure_limit'])
    objective = SpectralObjective(config['target_lower_nm'], config['target_upper_nm'])
    if type(config['generations']) is not int or config['generations'] < 1:
        raise ValueError('generations must be positive')
    return ga, death, objective


class MockWorker:
    async def __call__(self, request):
        # Deliberately produce three worsening generations, then improve after rebirth.
        distance = 180. + request.generation*10 if request.generation <= 3 else max(0.,180.-30*(request.generation-3))
        gene = EditGene('replace_atom', '{"atom":"seed.atom.0"}', '{"atomic_number":16}', f'mock-{request.slot}')
        program = EditProgram((gene,))
        return Individual(program.identity, program.encode(), -distance, False,
                          json.dumps({'mode':'mock', 'request':request.request_id}), distance)


async def execute(config_path, directory, *, live=False, confirmed=None, preflight_only=False):
    config = json.loads(config_path.read_text())
    ga, death, objective = parameters(config)
    budget = ga.population_size * config['generations']
    if live and confirmed != budget:
        raise ValueError(f'live run requires --confirm-max-new-evaluations {budget}; baseline adds one evaluation')
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in (HERE, HERE.parents[1]/'src') for p in sorted(folder.rglob('*.py'))}
    manifest = {'schema':'ga-molecular-run:v1','mode':'live' if live else 'mock',
                'config':config, 'source_hashes':sources}
    if not live:
        publish(directory/'manifest.json', manifest)
        founder = Individual('seed', EditProgram(()).encode(), -180., False, '{"mode":"mock"}', 180.)
        state = await LifecycleSearch(ga, MockWorker(), protocol_id=digest(manifest), death_policy=death).run(
            (founder,)*ga.population_size, generations=config['generations'], directory=directory/'population')
    else:
        from molecular_screening.support import REPOSITORY, MoleculeEditorProvider, LocalCodexRuntime, LocalWikiRetriever, plain
        from molecular_screening.compiler import ProgramCompiler
        from molecular_screening.agent import CodexGeneWorker
        from molecular_screening.services import FlameService
        from multi_agent_pso.runtimes import CodexTransportInterruptedError
        from openai_codex.errors import TransportClosedError
        from multi_agent_pso.configuration import load_run_inputs
        from multi_agent_pso.retrieval import snapshot_maintained_wiki
        from examples.red_absorption.flame_inputs import FlameRunInputs
        from examples.red_absorption.preflight import _default_auth_probe, _resolve_probe
        input_path = (config_path.parent / config['flame_inputs']).resolve()
        loaded = load_run_inputs(input_path, FlameRunInputs)
        inputs = loaded.value.model_copy(update={'flame_argv':(
            sys.executable, str(REPOSITORY/'PSO/examples/red_absorption/backends/flame_flsf.py'))})
        skill_root = Path(config['skill'])
        skill_files = [skill_root/'SKILL.md', *sorted((skill_root/'references').glob('*.md'))]
        skill_text = '\n'.join(p.read_text() for p in skill_files)
        manifest['skill_hashes'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in skill_files}
        manifest['wiki_snapshot'] = [asdict(entry) for entry in snapshot_maintained_wiki(Path(config['wiki']))]
        manifest['inputs_hash'] = loaded.semantic_sha256
        manifest['service_sources'] = {str(p.relative_to(REPOSITORY)):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in (REPOSITORY/'PSO/src', REPOSITORY/'PSO/examples/red_absorption') for p in sorted(base.rglob('*.py'))}
        protocol = digest(manifest)
        # This lock also covers preflight, preventing duplicate baseline/model calls.
        with run_lock(directory):
            publish(directory/'manifest.json', manifest)
            auth = await _resolve_probe(_default_auth_probe())
            if not auth.get('authenticated'):
                raise RuntimeError('local Codex is not authenticated')
            workspace = directory/'compiler'
            workspace.mkdir(exist_ok=True)
            async with MoleculeEditorProvider() as editor:
                seed = await editor.inspect({'kind':'smiles','value':config['seed_smiles']}, cwd=workspace)
                if not seed.processed or seed.chemical_status != 'VALID':
                    raise ValueError('seed molecule inspection failed')
                payload = plain(seed.payload)
                service = FlameService(inputs, directory, protocol_id=protocol,
                                       max_new_evaluations=budget+1, seed_smiles=payload['canonical_isomeric_smiles'])
                try:
                    prediction, reference = await service(payload)
                    metrics = objective.evaluate(prediction)
                    publish(directory/'preflight.json', {'prediction':prediction,'metrics':metrics,
                                                          'seed':payload,'protocol':protocol})
                    if preflight_only:
                        return {'mode':'live','status':'PREFLIGHT_PASSED','metrics':metrics}
                    evidence = {'phenotype_smiles':payload['canonical_isomeric_smiles'],
                                'chemical_identity':payload['chemical_identity_hash'],
                                'prediction':prediction,'evaluation_reference':reference}
                    empty = EditProgram(())
                    founder = Individual(empty.identity, empty.encode(), metrics['fitness'], metrics['feasible'],
                                         json.dumps(evidence), metrics['target_distance'])
                    for runtime_attempt in range(4):
                        try:
                            async with LocalCodexRuntime(model=config['model'], sandbox='read-only') as runtime:
                                worker = CodexGeneWorker(runtime=runtime, wiki=LocalWikiRetriever(Path(config['wiki'])),
                                    compiler=ProgramCompiler(editor,payload,workspace), evaluate=service, objective=objective,
                                    directory=directory, skill_text=skill_text)
                                state = await LifecycleSearch(ga,worker,protocol_id=protocol,death_policy=death).run(
                                    (founder,)*ga.population_size, generations=config['generations'], directory=directory/'population')
                            break
                        except (CodexTransportInterruptedError, TransportClosedError) as error:
                            print(f'Codex transport recovery {runtime_attempt+1}/4: {type(error).__name__}', file=sys.stderr)
                            if runtime_attempt == 3:
                                raise RuntimeError('Codex transport retries exhausted; resume with the same run directory') from error
                finally:
                    await service.close()
    summary = {'mode':'live' if live else 'mock', 'status':'COMPLETED', 'generations':state.generation,
               'population':len(state.members), 'deaths':state.death_count,
               'attempted_requests':state.attempted_requests, 'failed_requests':state.failed_requests,
               'best':asdict(state.best)}
    publish(directory/f'summary-{state.generation:08d}.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--runs-dir',type=Path,required=True)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--confirm-max-new-evaluations',type=int)
    args = parser.parse_args()
    try:
        result = asyncio.run(execute(args.config.resolve(),args.runs_dir,live=args.live,
            confirmed=args.confirm_max_new_evaluations,preflight_only=args.preflight_only))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'GA error: {error}\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
