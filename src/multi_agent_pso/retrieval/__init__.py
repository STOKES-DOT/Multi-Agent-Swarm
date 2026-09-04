"""Read-only retrieval adapters."""

from multi_agent_pso.protocols import WikiHit, WikiQuery

from .local_wiki import LocalWikiRetriever, WikiIndexLimits

__all__ = ["LocalWikiRetriever", "WikiHit", "WikiIndexLimits", "WikiQuery"]
