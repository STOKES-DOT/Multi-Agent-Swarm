"""Reuse the existing checked tool/runtime services from the PSO subproject."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[3]
for path in (REPOSITORY / 'PSO', REPOSITORY / 'PSO' / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_agent_pso.tools import MoleculeEditorProvider, canonicalize_commands
from multi_agent_pso.runtimes import LocalCodexRuntime
from multi_agent_pso.protocols import StageRequest, WikiQuery
from multi_agent_pso.core import AgentStage
from multi_agent_pso.retrieval import LocalWikiRetriever


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value
