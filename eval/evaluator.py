"""GraphTurn evaluation orchestrator.

Usage:
  python -m eval.evaluator --model gpt-4o --protocol guided --output_dir output/guided_gpt4o
  python -m eval.evaluator --model gpt-4o --protocol agentic --budget 3 --output_dir output/agentic3_gpt4o
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api_client import BaseLLMClient, get_client
from eval.agentic_runner import run_agentic
from eval.guided_runner import run_guided
from eval.metrics import aggregate_metrics, compute_all_session_metrics, compute_all_turn_metrics
from models import Protocol, Session
from neo4j_utils import get_connector

logger = logging.getLogger(__name__)


def load_benchmark(scenarios_dir: str) -> list:
    """Load benchmark from per-graph scenario files."""
    sessions = []
    scenarios_path = Path(scenarios_dir)
    for f in sorted(scenarios_path.glob("*_sessions.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        for item in data:
            sessions.append(Session(**item))
    logger.info(f"Loaded {len(sessions)} sessions from {scenarios_dir}")
    return sessions


def evaluate_single_session(
    session: Session,
    model_client: BaseLLMClient,
    protocol: Protocol,
    sim_client: BaseLLMClient = None,
    budget_multiplier: int = 3,
) -> Session:
    """Evaluate a single session."""
    try:
        if protocol == Protocol.GUIDED:
            session = run_guided(session, model_client)
        else:
            if sim_client is None:
                raise ValueError("Agentic protocol requires sim_client")
            session = run_agentic(session, model_client, sim_client, budget_multiplier=budget_multiplier)

        connector = get_connector(session.graph)
        for i, turn in enumerate(session.turns):
            prev_turn = session.turns[i - 1] if i > 0 else None
            turn.metrics = compute_all_turn_metrics(turn, prev_turn, connector)
        session.session_metrics = compute_all_session_metrics(session, connector)
        return session

    except Exception as e:
        logger.error(f"Session {session.session_id} failed: {e}")
        session.session_metrics = {"error": str(e)}
        return session


def evaluate(
    sessions: list,
    model_client: BaseLLMClient,
    protocol: Protocol,
    num_workers: int = 8,
    sim_client: BaseLLMClient = None,
    budget_multiplier: int = 3,
) -> list:
    """Evaluate all sessions."""
    results = []
    start_time = time.time()
    logger.info(f"Starting evaluation: protocol={protocol.value}, sessions={len(sessions)}, workers={num_workers}")

    if num_workers <= 1:
        for i, session in enumerate(sessions):
            session.protocol = protocol
            result = evaluate_single_session(session, model_client, protocol, sim_client, budget_multiplier)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(sessions)}")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for session in sessions:
                session.protocol = protocol
                future = executor.submit(
                    evaluate_single_session, session, model_client, protocol, sim_client, budget_multiplier
                )
                futures[future] = session.session_id
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Session {futures[future]} exception: {e}")
                if len(results) % 10 == 0:
                    logger.info(f"Progress: {len(results)}/{len(sessions)}")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete: {len(results)} sessions in {elapsed:.1f}s")
    return results


def save_results(sessions: list, aggregated: dict, output_dir: Path):
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result_data = [s.model_dump() for s in sessions]
    result_path = output_dir / "result_with_metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)

    agg_path = output_dir / "aggregated_metrics.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    if "overall" in aggregated:
        print("\n[Overall Metrics]")
        for k, v in aggregated["overall"].items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
    if "by_graph" in aggregated:
        print("\n[By Graph - EX]")
        for graph, metrics in aggregated["by_graph"].items():
            print(f"  {graph}: {metrics.get('EX', 0):.4f}")
    if "by_phenomenon" in aggregated:
        print("\n[By Phenomenon - EX]")
        for phen, metrics in aggregated["by_phenomenon"].items():
            print(f"  {phen}: {metrics.get('EX', 0):.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GraphTurn Evaluation")
    parser.add_argument("--scenarios_dir", default="data/scenarios", help="Path to scenario files")
    parser.add_argument("--model", required=True, help="Model name (passed to API)")
    parser.add_argument("--base_url", default=None, help="LLM API base URL (or set LLM_BASE_URL)")
    parser.add_argument("--api_key", default=None, help="LLM API key (or set LLM_API_KEY)")
    parser.add_argument("--protocol", required=True, choices=["guided", "agentic"])
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3, help="Agentic budget multiplier (3/5/10)")
    parser.add_argument("--sim_model", default=None, help="User simulator model (agentic only)")
    parser.add_argument("--sim_base_url", default=None, help="Simulator API base URL")
    parser.add_argument("--sim_api_key", default=None, help="Simulator API key")
    parser.add_argument("--max_sessions", type=int, default=None, help="Limit sessions (for debugging)")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sessions = load_benchmark(args.scenarios_dir)
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]

    model_client = get_client(args.model, base_url=args.base_url, api_key=args.api_key)
    protocol = Protocol(args.protocol)

    sim_client = None
    if protocol == Protocol.AGENTIC:
        sim_model = args.sim_model or args.model
        sim_client = get_client(sim_model, base_url=args.sim_base_url, api_key=args.sim_api_key)

    results = evaluate(
        sessions=sessions,
        model_client=model_client,
        protocol=protocol,
        num_workers=args.num_workers,
        sim_client=sim_client,
        budget_multiplier=args.budget,
    )

    connector_map = {}
    for s in results:
        if s.graph not in connector_map:
            try:
                connector_map[s.graph] = get_connector(s.graph)
            except Exception:
                pass

    aggregated = aggregate_metrics(results, connector_map)
    save_results(results, aggregated, Path(args.output_dir))


if __name__ == "__main__":
    main()
