"""Provenance Subgraph Jaccard Similarity (PSJS).

Measures overlap of node element-IDs bound during pattern matching.
"""
from __future__ import annotations

import re

from graph_eval_utils.neo4j_connector import Neo4jConnector


def _extract_node_ids(cypher: str, connector: Neo4jConnector, timeout: int) -> set:
    """Execute a query variant that returns internal node IDs from MATCH patterns."""
    var_pattern = re.findall(r'\((\w+)(?::\w+)', cypher)
    if not var_pattern:
        return set()

    target_var = var_pattern[0]
    id_query = re.sub(
        r'RETURN\s+.+$',
        f'RETURN DISTINCT id({target_var}) AS __node_id',
        cypher,
        flags=re.IGNORECASE | re.DOTALL,
    )
    id_query = re.sub(r'ORDER\s+BY\s+.+?(?=RETURN|$)', '', id_query, flags=re.IGNORECASE)
    id_query = re.sub(r'LIMIT\s+\d+', '', id_query, flags=re.IGNORECASE)

    try:
        results = connector.run_query(id_query, timeout=timeout)
        return {r["__node_id"] for r in results if "__node_id" in r}
    except Exception:
        return set()


def provenance_subgraph_jaccard_similarity(
    pred_cypher: str,
    gold_cypher: str,
    connector: Neo4jConnector,
    timeout: int = 120,
) -> float:
    """Compute Jaccard similarity of node-ID sets bound by pred and gold queries."""
    pred_ids = _extract_node_ids(pred_cypher, connector, timeout)
    gold_ids = _extract_node_ids(gold_cypher, connector, timeout)

    if not pred_ids and not gold_ids:
        return 0.0

    intersection = pred_ids & gold_ids
    union = pred_ids | gold_ids
    return len(intersection) / len(union) if union else 0.0
