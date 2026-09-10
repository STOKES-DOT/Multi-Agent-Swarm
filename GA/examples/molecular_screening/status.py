"""Read committed progress without starting a runtime or calculation."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from multi_agent_ga.persistence import read_record


def summarize(directory):
    paths=sorted((directory/'population').glob('population-*.json'))
    if not paths:
        return {'status':'NO_COMMITTED_POPULATION'}
    state=read_record(paths[-1])['state']
    trials=list((directory/'population'/'trials').glob('*.json'))
    return {'status':'CHECKPOINT_SNAPSHOT', 'generation':state['generation'],
            'population':len(state['members']), 'attempted_requests':state['attempted_requests'],
            'journaled_trials':len(trials), 'failed_requests':state['failed_requests'],
            'deaths':state['death_count'], 'best':state['best'],
            'members':[{'slot':m['slot'],'lineage_id':m['lineage_id'],
                        'fitness':m['individual']['fitness'], 'distances':m['distances'],
                        'valid_edits':m['valid_edits'], 'failures':m['failures']} for m in state['members']]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs_dir',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.runs_dir),ensure_ascii=False,indent=2))
