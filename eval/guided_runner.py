"""Guided protocol evaluation runner.

Each turn receives gold-annotated history (oracle context) and a budget of 3 actions.
"""
from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

from api_client import BaseLLMClient
from models import Action, ActionResult, ActionType, BudgetConfig, Session, Turn
from neo4j_utils import execute_cypher, get_connector, get_schema_text, inspect_schema, search_values, serialize_result
from prompts.eval_prompts import ACTION_SPACE_SYSTEM_PROMPT, GUIDED_TURN_PROMPT

logger = logging.getLogger(__name__)


class GuidedEnvironment:
    """Guided evaluation environment: processes model actions and returns results."""

    def __init__(self, graph: str, budget: BudgetConfig):
        self.graph = graph
        self.connector = get_connector(graph)
        self.budget = budget
        self.schema_text = get_schema_text(graph)

    def process_action(self, action: Action, turn: Optional[Turn] = None) -> ActionResult:
        start_time = time.time()

        if action.action_type == ActionType.EXECUTE_CYPHER:
            result, error = execute_cypher(self.connector, action.payload, self.budget.cypher_timeout_sec)
            elapsed = (time.time() - start_time) * 1000
            return ActionResult(
                action_type=action.action_type, success=error is None,
                result=result, error_message=error, execution_time_ms=elapsed,
            )
        elif action.action_type == ActionType.INSPECT_SCHEMA:
            schema = inspect_schema(self.graph)
            return ActionResult(action_type=action.action_type, success=True, result=schema)
        elif action.action_type == ActionType.SEARCH_VALUES:
            try:
                params = json.loads(action.payload)
                values = search_values(
                    self.connector, label=params.get("label", ""),
                    property_name=params.get("property", "name"), query=params.get("query", ""),
                )
                return ActionResult(action_type=action.action_type, success=True, result=values)
            except (json.JSONDecodeError, Exception) as e:
                return ActionResult(action_type=action.action_type, success=False, error_message=f"SEARCH_VALUES parameter parse error: {e}")
        elif action.action_type == ActionType.ASK_USER:
            reply = self._get_scripted_user_reply(action.payload, turn)
            return ActionResult(action_type=action.action_type, success=True, result=reply)
        elif action.action_type == ActionType.SUBMIT_ANSWER:
            return ActionResult(action_type=action.action_type, success=True, result="Answer submitted")

        return ActionResult(action_type=action.action_type, success=False, error_message=f"Unknown action type: {action.action_type}")

    def _get_scripted_user_reply(self, question: str, turn: Optional[Turn]) -> str:
        if turn and getattr(turn, "ambiguity_info", None):
            return turn.ambiguity_info.simulator_reply
        return "I think my question was clear enough. Please try to answer it directly."


def parse_action(raw_output: str) -> Action:
    """Parse model output into an Action."""
    if raw_output is None:
        return Action(action_type=ActionType.SUBMIT_ANSWER, payload="", raw_llm_output=raw_output)

    lines = raw_output.strip().split("\n")
    action_type = None
    payload_lines = []
    in_payload = False

    for line in lines:
        stripped = line.strip()
        action_match = re.match(r"^ACTION:\s*(.+)$", stripped, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip().upper()
            action_map = {
                "EXECUTE_CYPHER": ActionType.EXECUTE_CYPHER, "EXECUTE": ActionType.EXECUTE_CYPHER,
                "CYPHER": ActionType.EXECUTE_CYPHER, "ASK_USER": ActionType.ASK_USER,
                "ASK": ActionType.ASK_USER, "INSPECT_SCHEMA": ActionType.INSPECT_SCHEMA,
                "SCHEMA": ActionType.INSPECT_SCHEMA, "SEARCH_VALUES": ActionType.SEARCH_VALUES,
                "SEARCH": ActionType.SEARCH_VALUES, "SUBMIT_ANSWER": ActionType.SUBMIT_ANSWER,
                "SUBMIT": ActionType.SUBMIT_ANSWER,
            }
            action_type = action_map.get(action_str)
            in_payload = False
            continue
        payload_match = re.match(r"^PAYLOAD:\s*(.*)$", stripped, re.IGNORECASE)
        if payload_match:
            payload_lines = [payload_match.group(1)]
            in_payload = True
            continue
        if in_payload:
            payload_lines.append(line)

    if action_type is not None:
        payload = "\n".join(payload_lines).strip()
        return Action(action_type=action_type, payload=payload, raw_llm_output=raw_output)

    # Fallback: extract Cypher from markdown code blocks
    if "```cypher" in raw_output:
        cypher = raw_output.split("```cypher")[1].split("```")[0].strip()
        return Action(action_type=ActionType.SUBMIT_ANSWER, payload=cypher, raw_llm_output=raw_output)
    if "```" in raw_output:
        code = raw_output.split("```")[1].split("```")[0].strip()
        if code.upper().startswith(("MATCH", "CALL", "OPTIONAL", "RETURN", "WITH")):
            return Action(action_type=ActionType.SUBMIT_ANSWER, payload=code, raw_llm_output=raw_output)

    text = raw_output.strip()
    if text.upper().startswith(("MATCH", "CALL", "OPTIONAL")):
        return Action(action_type=ActionType.SUBMIT_ANSWER, payload=text, raw_llm_output=raw_output)

    return Action(action_type=ActionType.SUBMIT_ANSWER, payload=text, raw_llm_output=raw_output)


def format_action_result(result: ActionResult) -> str:
    """Format action result as text for conversation context."""
    if result.action_type == ActionType.EXECUTE_CYPHER:
        if result.success:
            return f"[Query Result]\n{serialize_result(result.result, max_rows=20)}"
        return f"[Query Failed] {result.error_message}"
    elif result.action_type == ActionType.INSPECT_SCHEMA:
        return f"[Schema]\n{result.result}"
    elif result.action_type == ActionType.SEARCH_VALUES:
        if result.success:
            return f"[Search Result] {json.dumps(result.result, ensure_ascii=False)}"
        return f"[Search Failed] {result.error_message}"
    elif result.action_type == ActionType.ASK_USER:
        return f"[User Reply] {result.result}"
    elif result.action_type == ActionType.SUBMIT_ANSWER:
        return "[Answer Submitted]"
    return str(result.result)


def build_conversation_history(
    previous_turns: List[Turn], gold_history: bool = True, no_truncate_last: bool = False,
) -> str:
    """Build conversation history text from previous turns."""
    if not previous_turns:
        return "(This is the first turn of the conversation)"
    parts = []
    for idx, turn in enumerate(previous_turns):
        parts.append(f"User: {turn.user_utterance}")
        if gold_history:
            is_last = (idx == len(previous_turns) - 1)
            max_rows = None if (is_last and no_truncate_last) else 10
            result_text = serialize_result(turn.gold_answer, max_rows=max_rows)
            parts.append(f"Assistant: [Query Result] {result_text}")
        else:
            if turn.pred_actions:
                last_action = turn.pred_actions[-1]
                parts.append(f"Assistant: {last_action.payload}")
    return "\n".join(parts)


def run_guided(session: Session, model_client: BaseLLMClient) -> Session:
    """Run guided evaluation for a single session."""
    budget = BudgetConfig()
    env = GuidedEnvironment(session.graph, budget)

    for i, turn in enumerate(session.turns):
        is_contrast_turn = any(str(p).upper() == "CONTRAST" for p in (turn.phenomena or []))
        history = build_conversation_history(session.turns[:i], gold_history=True, no_truncate_last=is_contrast_turn)

        dependency_hint = ""
        if turn.depends_on_result_of is not None:
            dep_turn_id = turn.depends_on_result_of
            dependency_hint = (
                f"\n[Note: This turn builds upon the result of Turn {dep_turn_id}. "
                f"Use the result shown for Turn {dep_turn_id} in the conversation history above.]"
            )

        turn_prompt = GUIDED_TURN_PROMPT.format(
            graph_name=session.graph, schema_info=f"Schema:\n{env.schema_text}",
            conversation_history=history, user_utterance=turn.user_utterance + dependency_hint,
            budget_remaining=budget.guided_per_turn,
        )

        turn.pred_actions = []
        turn.pred_action_results = []
        turn.budget_used = 0
        action_history = []

        for step in range(budget.guided_per_turn):
            messages = [{"role": "system", "content": ACTION_SPACE_SYSTEM_PROMPT}]
            messages.append({"role": "user", "content": turn_prompt})
            for ah in action_history:
                messages.append({"role": "assistant", "content": ah["action_text"]})
                messages.append({"role": "user", "content": ah["result_text"]})

            response, in_tokens, out_tokens = model_client.call(messages, temperature=0.0, max_tokens=2048)
            action = parse_action(response)
            action.input_tokens = in_tokens
            action.output_tokens = out_tokens
            action.timestamp = datetime.now().isoformat()
            result = env.process_action(action, turn)

            turn.pred_actions.append(action)
            turn.pred_action_results.append(result)
            turn.budget_used += 1

            if action.action_type == ActionType.SUBMIT_ANSWER:
                turn.pred_cypher = action.payload
                break

            action_text = f"ACTION: {action.action_type.value}\nPAYLOAD: {action.payload}"
            result_text = format_action_result(result)
            action_history.append({"action_text": action_text, "result_text": result_text})

        if not turn.pred_cypher:
            for a in reversed(turn.pred_actions):
                if a.action_type == ActionType.EXECUTE_CYPHER:
                    turn.pred_cypher = a.payload
                    break

    return session
