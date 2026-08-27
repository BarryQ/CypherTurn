"""GraphTurn evaluation metrics.

Turn-level: EX, PSJS, CER
Session-level: SEM (Session Exact Match)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from graph_eval_utils.metrics.execution_accuracy import execution_accuracy
from graph_eval_utils.metrics.provenance_subgraph_jaccard_similarity import (
    provenance_subgraph_jaccard_similarity,
)
from graph_eval_utils.neo4j_connector import Neo4jConnector

from models import ActionType, Phenomenon, Session, Turn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Turn-level metrics
# ═══════════════════════════════════════════════════════════════════════════

def turn_ex(turn: Turn, connector: Neo4jConnector, timeout: int = 120) -> float:
    """Execution Accuracy: 1.0 if predicted and gold results match exactly."""
    pred_cypher = _get_submitted_cypher(turn)
    if not pred_cypher:
        return 0.0
    try:
        return execution_accuracy(pred_cypher, turn.gold_cypher, connector, timeout)
    except Exception as e:
        logger.debug(f"EX computation failed: {e}")
        return 0.0


def turn_psjs(turn: Turn, connector: Neo4jConnector, timeout: int = 120) -> float:
    """Provenance Subgraph Jaccard Similarity."""
    pred_cypher = _get_submitted_cypher(turn)
    if not pred_cypher:
        return 0.0
    try:
        ex = execution_accuracy(pred_cypher, turn.gold_cypher, connector, timeout)
        if ex == 1.0:
            return 1.0
    except Exception:
        pass
    try:
        return provenance_subgraph_jaccard_similarity(
            pred_cypher, turn.gold_cypher, connector, timeout
        )
    except Exception as e:
        logger.debug(f"PSJS computation failed: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Session-level metrics
# ═══════════════════════════════════════════════════════════════════════════

def session_sem(session: Session, connector: Neo4jConnector, timeout: int = 120) -> float:
    """Session Exact Match: 1.0 iff ALL turns have EX=1.0."""
    if not session.turns:
        return 0.0
    return 1.0 if all(
        turn_ex(t, connector, timeout) == 1.0 for t in session.turns
    ) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_turn_metrics(
    turn: Turn,
    prev_turn: Optional[Turn],
    connector: Neo4jConnector,
    timeout: int = 120,
) -> Dict[str, float]:
    """Compute all turn-level metrics for a single turn."""
    curr_ex = turn_ex(turn, connector, timeout)
    curr_psjs = turn_psjs(turn, connector, timeout)

    # CER: P(EX_t=0 | EX_{t-1}=0) for chain-dependent turns
    cer: Optional[float] = None
    if turn.depends_on_result_of is not None:
        if curr_ex == 1.0:
            cer = 0.0
        elif prev_turn is not None:
            prev_ex = turn_ex(prev_turn, connector, timeout)
            cer = 1.0 if prev_ex == 0.0 else 0.0
        else:
            cer = 0.0

    metrics: Dict[str, float] = {"EX": curr_ex, "PSJS": curr_psjs}
    if cer is not None:
        metrics["CER"] = cer
    return metrics


def compute_all_session_metrics(
    session: Session,
    connector: Neo4jConnector,
    timeout: int = 120,
) -> Dict[str, float]:
    """Compute session-level metrics."""
    return {"SEM": session_sem(session, connector, timeout)}


def aggregate_metrics(
    sessions: List[Session],
    connector_map: Dict[str, Neo4jConnector],
    timeout: int = 120,
) -> Dict[str, dict]:
    """Aggregate metrics across all sessions.

    Returns overall, by_graph, and by_phenomenon breakdowns.
    """
    all_turn_metrics = []
    turn_to_graph = []
    turn_to_phenomena = []

    for session in sessions:
        connector = connector_map.get(session.graph)
        if connector is None:
            logger.warning(f"No Neo4j connection for {session.graph}, skipping")
            continue

        for i, turn in enumerate(session.turns):
            prev_turn = session.turns[i - 1] if i > 0 else None
            metrics = compute_all_turn_metrics(turn, prev_turn, connector, timeout)
            turn.metrics = metrics
            all_turn_metrics.append(metrics)
            turn_to_graph.append(session.graph)
            turn_to_phenomena.append(turn.phenomena)

        session.session_metrics = compute_all_session_metrics(session, connector, timeout)

    result = {}

    # Overall
    result["overall"] = {}
    for m in ["EX", "PSJS"]:
        result["overall"][m] = _safe_avg([t.get(m, 0) for t in all_turn_metrics])
    cer_vals = [t["CER"] for t in all_turn_metrics if "CER" in t]
    result["overall"]["CER"] = _safe_avg(cer_vals) if cer_vals else None
    sem_vals = [s.session_metrics.get("SEM", 0) for s in sessions if s.session_metrics]
    result["overall"]["SEM"] = _safe_avg(sem_vals)

    # By graph
    result["by_graph"] = {}
    for graph_name in set(turn_to_graph):
        indices = [i for i, g in enumerate(turn_to_graph) if g == graph_name]
        gd: Dict[str, Any] = {}
        for m in ["EX", "PSJS"]:
            gd[m] = _safe_avg([all_turn_metrics[i].get(m, 0) for i in indices])
        vals = [all_turn_metrics[i]["CER"] for i in indices if "CER" in all_turn_metrics[i]]
        gd["CER"] = _safe_avg(vals) if vals else None
        graph_sessions = [s for s in sessions if s.graph == graph_name and s.session_metrics]
        gd["SEM"] = _safe_avg([s.session_metrics.get("SEM", 0) for s in graph_sessions])
        result["by_graph"][graph_name] = gd

    # By phenomenon
    result["by_phenomenon"] = {}
    for phen in Phenomenon:
        indices = [i for i, phens in enumerate(turn_to_phenomena) if phen in phens]
        if indices:
            pd: Dict[str, Any] = {}
            for m in ["EX", "PSJS"]:
                pd[m] = _safe_avg([all_turn_metrics[i].get(m, 0) for i in indices])
            vals = [all_turn_metrics[i]["CER"] for i in indices if "CER" in all_turn_metrics[i]]
            pd["CER"] = _safe_avg(vals) if vals else None
            result["by_phenomenon"][phen.value] = pd

    return result


def _safe_avg(values: List[float]) -> float:
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(valid) / len(valid) if valid else 0.0


def _get_submitted_cypher(turn: Turn) -> Optional[str]:
    """Extract the final submitted Cypher from a turn's action history."""
    if turn.pred_cypher:
        return turn.pred_cypher
    submit_actions = [
        a for a in turn.pred_actions if a.action_type == ActionType.SUBMIT_ANSWER
    ]
    if submit_actions:
        return submit_actions[-1].payload
    execute_actions = [
        a for a in turn.pred_actions if a.action_type == ActionType.EXECUTE_CYPHER
    ]
    if execute_actions:
        return execute_actions[-1].payload
    return None
