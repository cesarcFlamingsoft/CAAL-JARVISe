"""The local company knowledge library.

An independent package: an owner-scoped, encrypted store of company documents
(policies, HR documents, contracts, general material), a bounded local
extractor, an encrypted full-text index, and a read-only MCP server that lets
the local model look a passage up and cite it.

Nothing in this package reaches a network, and nothing in it ever treats the
text of an uploaded document as an instruction. See
``reports/company-knowledge/DESIGN.md`` for the architecture and threat model.
"""

from __future__ import annotations

__all__ = ["config", "extraction", "service"]
