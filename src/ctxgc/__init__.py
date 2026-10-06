"""ctxgc: dependency-graph context compression.

Pipeline: ingest -> edges -> propagate -> allocate -> render.
See docs/design-notes.md for the design rationale.
"""

from .model import Edge, EdgeType, Graph, Node, Role, L0, L1, L2, L3
from .compress import compress, build_graph, Result

__all__ = [
    "Edge", "EdgeType", "Graph", "Node", "Role", "L0", "L1", "L2", "L3",
    "compress", "build_graph", "Result",
]
