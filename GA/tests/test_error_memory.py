import pytest

from molecular_screening.error_memory import ErrorMemory
from molecular_screening.genes import EditGene, EditProgram
from multi_agent_ga import Individual, OffspringRequest


def request(lineage='L1', generation=1, parent='P1', slot=0):
    individual=Individual(parent,'genome',0.,False)
    return OffspringRequest(f'g{generation}-s{slot}',generation,slot,individual,None,True,42,lineage,individual)


def program():
    return EditProgram((EditGene('replace_atom','{"atom":"seed.atom.0"}','{"atomic_number":16}'),))


def test_schema_rules_shared_but_private_failure_stays_with_lineage(tmp_path):
    memory=ErrorMemory(tmp_path,'seed-1')
    memory.record_failure(request(),0,'ExpressionError: add_atom requires site roles []',program())
    own=memory.packet(request(generation=2))
    other=memory.packet(request(lineage='L2',generation=2,slot=1))
    assert len(own['private_failures'])==1
    assert not other['private_failures']
    assert own['coding_rules']['add_atom_site']['observed_count']==1
    assert other['coding_rules']['add_atom_site']['observed_count']==1


def test_structure_error_requires_matching_seed_and_genetic_context(tmp_path):
    memory=ErrorMemory(tmp_path,'seed-1')
    memory.record_failure(request(),0,'ATOM_VALENCE_ERROR: carbon valence 6',program())
    assert memory.packet(request(lineage='L2',generation=2))['structural_failures']
    assert not memory.packet(request(lineage='L2',generation=2,parent='unrelated'))['structural_failures']
    assert not ErrorMemory(tmp_path,'seed-2').packet(request(generation=2))['structural_failures']


def test_new_lineage_has_no_old_private_history_and_same_generation_is_frozen(tmp_path):
    memory=ErrorMemory(tmp_path,'seed-1')
    first=memory.packet(request())
    memory.record_failure(request(),0,'AROMATICITY_ERROR',program())
    assert memory.packet(request())==first
    assert memory.packet(request(generation=2))['private_failures']
    reborn=memory.packet(request(lineage='reborn',generation=2))
    assert not reborn['private_failures']
    assert reborn['structural_failures']  # Shared applicable cases remain available.


def test_refuted_prediction_not_promoted_to_causal_prohibition(tmp_path):
    memory=ErrorMemory(tmp_path,'seed-1')
    memory.record_outcome(request(),0,program(),{
        'hypothesis':'red shift expected',
        'hypothesis_outcome':{'status':'REFUTED','delta_absorption_nm':5.,'minimum_change_nm':100.},
        'evaluation_reference':{'sha256':'abc'},
        'reflection':{'reflection':'mechanism is uncertain'},
    })
    packet=memory.packet(request(generation=2))
    assert packet['model_outcomes'][0]['mechanism_proven'] is False
    assert packet['model_outcomes'][0]['hypothesis_outcome']['status']=='REFUTED'
    assert packet['private_reflections']


def test_packet_is_bounded_and_reloadable(tmp_path):
    memory=ErrorMemory(tmp_path,'seed-1')
    for i in range(15):
        memory.record_failure(request(generation=i+1),0,'AROMATICITY_ERROR',program())
    packet=ErrorMemory(tmp_path,'seed-1').packet(request(generation=20))
    assert len(packet['private_failures'])<=4
    assert len(packet['structural_failures'])<=4
