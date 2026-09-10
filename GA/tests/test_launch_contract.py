import json
from pathlib import Path
import pytest

from molecular_screening.reward import SpectralObjective
from molecular_screening.services import evaluation_reference
from molecular_screening.run import parameters
from examples.red_absorption.flame_proxy import FlamePrediction, FlameProxyEvaluator


def test_twenty_by_nine_configuration():
    path = Path(__file__).resolve().parents[1]/'examples/molecular_screening/config-20x9.json'
    config = json.loads(path.read_text())
    ga, death, objective = parameters(config)
    assert (ga.population_size, config['generations'], config['model']) == (20,9,'gpt-5.6-luna')
    assert (objective.lower_nm,objective.upper_nm)==(620.,750.)
    assert death.window==3


def test_evaluation_reference_is_stable_across_restart(tmp_path):
    prediction = {'absorption_nm':393.3, 'model_hashes':{'abs':'a'*64}}
    first = evaluation_reference(tmp_path,prediction)
    assert evaluation_reference(tmp_path,json.loads(json.dumps(prediction))) == first
    assert evaluation_reference(tmp_path,{**prediction,'absorption_nm':400.}) != first


@pytest.mark.parametrize('wavelength', [393.3,619.,620.,700.,750.,800.])
def test_default_ga_reward_matches_pso(wavelength):
    prediction = FlamePrediction(dye_smiles='C',solvent_smiles='ClCCl',absorption_nm=wavelength,
        emission_nm=800.,plqy=.5,epsilon_m1_cm1=3e4,model_hashes={k:'a'*64 for k in ('abs','emi','plqy','e')})
    ga = SpectralObjective().evaluate(prediction.model_dump(mode='json'))
    pso = FlameProxyEvaluator().evaluate_prediction(prediction)
    assert ga['fitness'] == pytest.approx(pso.fitness,abs=1e-14)
    assert ga['feasible']==pso.feasible
