import pytest
from molecular_screening.agent import evidence_contract, resolve_evidence, PROPOSAL


def hits():
    return [
        {'relative_path':'sources/source-mr-tadf-315.md','line_start':20,'line_end':25},
        {'relative_path':'sources/source-mr-tadf-315.md','line_start':40,'line_end':45},
        {'relative_path':'sources/source-mr-tadf-240.md','line_start':10,'line_end':12},
    ]


def test_explicit_ids_and_schema_refer_to_exact_retrieved_passages():
    raw=hits()
    annotated,schema=evidence_contract(raw)
    assert [hit['evidence_id'] for hit in annotated]==['W0','W1','W2']
    assert schema['properties']['evidence_ids']['items']=={'type':'string','enum':['W0','W1','W2']}
    assert 'evidence_id' not in raw[0]
    assert resolve_evidence(['W1'],annotated)[0]['line_start']==40
    assert 'enum' not in PROPOSAL['properties']['evidence_ids']['items']
    smaller,other=evidence_contract(raw[:1])
    assert other['properties']['evidence_ids']['items']['enum']==['W0']
    assert schema['properties']['evidence_ids']['items']['enum']==['W0','W1','W2']


@pytest.mark.parametrize('refs',[[315,240,105],[84,85,87],[186],['315'],[0],['W9'],[],None,[True]])
def test_paper_ids_lines_and_unknown_ids_are_rejected_with_actionable_feedback(refs):
    annotated,_=evidence_contract(hits())
    with pytest.raises(ValueError,match='W0.*W1.*W2') as exc:
        resolve_evidence(refs,annotated)
    assert 'not paper numbers' in str(exc.value)
    assert 'Correct evidence_ids only' in str(exc.value)


def test_provider_cannot_override_controller_assigned_id():
    annotated,_=evidence_contract([{**hits()[0],'evidence_id':'315'}])
    assert annotated[0]['evidence_id']=='W0'
