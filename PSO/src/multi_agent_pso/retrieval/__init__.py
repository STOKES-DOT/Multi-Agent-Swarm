"""Read-only retrieval adapters."""

from multi_agent_pso.protocols import WikiHit, WikiQuery

from .local_wiki import (
    LocalWikiRetriever,
    WikiIndexLimits,
    WikiSnapshotEntry,
    snapshot_maintained_wiki,
)

__all__ = [
    "LocalWikiRetriever",
    "WikiHit",
    "WikiIndexLimits",
    "WikiQuery",
    "WikiSnapshotEntry",
    "snapshot_maintained_wiki",
]
