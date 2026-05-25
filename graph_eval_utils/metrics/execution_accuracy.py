"""Execution Accuracy: compare predicted vs gold Cypher by executing both."""
from __future__ import annotations

from graph_eval_utils.neo4j_connector import Neo4jConnector


def execution_accuracy(
    pred_cypher: str,
    gold_cypher: str,
    connector: Neo4jConnector,
    timeout: int = 120,
) -> float:
    """Return 1.0 if pred and gold produce identical result sets, else 0.0."""
    try:
        pred_result = connector.run_query(pred_cypher, timeout=timeout)
        gold_result = connector.run_query(gold_cypher, timeout=timeout)
    except Exception:
        return 0.0

    if len(pred_result) != len(gold_result):
        return 0.0

    if not gold_result:
        return 1.0

    gold_keys = sorted(gold_result[0].keys())
    pred_keys = sorted(pred_result[0].keys()) if pred_result else []
    if gold_keys != pred_keys:
        return 0.0

    def normalize_row(row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in gold_keys)

    gold_set = sorted(normalize_row(r) for r in gold_result)
    pred_set = sorted(normalize_row(r) for r in pred_result)
    return 1.0 if gold_set == pred_set else 0.0
