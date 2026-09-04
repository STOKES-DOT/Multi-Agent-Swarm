"""Read-only retrieval adapters."""

from multi_agent_pso.protocols import WikiHit, WikiQuery

from .local_wiki import LocalWikiRetriever

__all__ = ["LocalWikiRetriever", "WikiHit", "WikiQuery"]
