"""Agentic protocol evaluation runner.

The model operates autonomously with a shared action budget across all turns.
No oracle context is provided; the model must discover schema and manage its own history.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from api_client import BaseLLMClient
from eval.guided_runner import GuidedEnvironment, format_action_result, parse_action
from eval.user_sim_eval import UserSimulatorEval
from models import Action, ActionResult, ActionType, BudgetConfig, Session, Turn
from neo4j_utils import get_schema_summary
from prompts.eval_prompts import ACTION_SPACE_SYSTEM_PROMPT, AGENTIC_SYSTEM_PROMPT, AGENTIC_TURN_PROMPT

logger = logging.getLogger(__name__)


class AgenticEnvironment(GuidedEnvironment):
    """Agentic environment with live user simulator and total budget tracking."""

    def __init__(self, graph: str, budget: BudgetConfig, user_sim: UserSimulatorEval):
        super().__init__(graph, budget)
        self.user_sim = user_sim
        self.total_budget_remaining = budget.agentic_total

    def process_action(
        self, action: Action, turn: Optional[Turn] = None,
        conversation_context: Optional[List[dict]] = None,
    ) -> ActionResult:
        if action.action_type == ActionType.ASK_USER:
            last_model_result = None
            if turn and turn.pred_action_results:
                for ar in reversed(turn.pred_action_results):
                    if ar.action_type == ActionType.EXECUTE_CYPHER and ar.success and ar.result:
                        try:
                            last_model_result = json.dumps(ar.result[:5], ensure_ascii=False)
                        except Exception:
                            last_model_result = str(ar.result)[:300]
                        break
            reply = self.user_sim.respond(
                action.payload, conversation_context or [], last_model_result=last_model_result,
            )
            return ActionResult(action_type=action.action_type, success=True, result=reply)
        return super().process_action(action, turn)


def run_agentic(
    session: Session, model_client: BaseLLMClient, sim_client: BaseLLMClient,
    budget_multiplier: int = 3,
) -> Session:
    """Run agentic evaluation for a single session."""
    budget = BudgetConfig()
    budget.agentic_total = len(session.turns) * budget_multiplier

    user_sim = UserSimulatorEval(
        persona=session.persona, graph=session.graph,
        gold_turns=session.turns, client=sim_client,
    )
    env = AgenticEnvironment(session.graph, budget, user_sim)

    system_prompt = AGENTIC_SYSTEM_PROMPT.format(total_budget=budget.agentic_total)

    for turn_idx, turn in enumerate(session.turns):
        if turn_idx == 0:
            user_utterance = turn.user_utterance
        else:
            user_utterance = user_sim.get_next_utterance(turn_idx)

        interaction_history = _build_interaction_history(session.turns[:turn_idx])
        turn_prompt = AGENTIC_TURN_PROMPT.format(
            graph_name=session.graph, user_utterance=user_utterance,
            interaction_history=interaction_history or "(This is the first turn of the conversation)",
            budget_remaining=env.total_budget_remaining, total_budget=budget.agentic_total,
        )

        turn.pred_actions = []
        turn.pred_action_results = []
        turn.budget_used = 0

        conversation = [
            {"role": "system", "content": system_prompt + "\n\n" + ACTION_SPACE_SYSTEM_PROMPT},
            {"role": "user", "content": turn_prompt},
        ]

        while env.total_budget_remaining > 0:
            response, in_tokens, out_tokens = model_client.call(conversation, temperature=0.0, max_tokens=2048)
            action = parse_action(response)
            action.input_tokens = in_tokens
            action.output_tokens = out_tokens
            action.timestamp = datetime.now().isoformat()
            result = env.process_action(action, turn, conversation)

            turn.pred_actions.append(action)
            turn.pred_action_results.append(result)
            turn.budget_used += 1
            env.total_budget_remaining -= 1

            if action.action_type == ActionType.SUBMIT_ANSWER:
                turn.pred_cypher = action.payload
                break

            action_text = f"ACTION: {action.action_type.value}\nPAYLOAD: {action.payload}"
            result_text = format_action_result(result)
            conversation.append({"role": "assistant", "content": action_text})
            conversation.append({"role": "user", "content": result_text})

            if env.total_budget_remaining <= 0:
                logger.info(f"Session {session.session_id} budget exhausted (turn {turn_idx + 1})")
                break

        if not turn.pred_cypher:
            for a in reversed(turn.pred_actions):
                if a.action_type == ActionType.EXECUTE_CYPHER:
                    turn.pred_cypher = a.payload
                    break

        if env.total_budget_remaining <= 0:
            for remaining_turn in session.turns[turn_idx + 1:]:
                remaining_turn.pred_actions = []
                remaining_turn.pred_action_results = []
                remaining_turn.budget_used = 0
                remaining_turn.pred_cypher = None
            break

    return session


def _build_interaction_history(previous_turns: List[Turn]) -> str:
    if not previous_turns:
        return ""
    parts = []
    for turn in previous_turns:
        parts.append(f"--- Turn {turn.turn_id} ---")
        parts.append(f"User: {turn.user_utterance}")
        for action, result in zip(turn.pred_actions, turn.pred_action_results):
            parts.append(f"  Action: {action.action_type.value}")
            if action.action_type == ActionType.EXECUTE_CYPHER:
                if result.success:
                    parts.append(f"  Result: {str(result.result)[:200]}")
                else:
                    parts.append(f"  Error: {result.error_message[:200]}")
            elif action.action_type == ActionType.ASK_USER:
                parts.append(f"  Question: {action.payload}")
                parts.append(f"  User Reply: {result.result}")
            elif action.action_type == ActionType.SUBMIT_ANSWER:
                parts.append(f"  Submitted: {action.payload[:200]}")
    return "\n".join(parts)
