"""Agentic evaluation user simulator.

Responds to ASK_USER actions with persona-appropriate replies.
Includes two-stage anti-leakage: intent classification + schema term filtering.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from api_client import BaseLLMClient
from models import PERSONA_DESCRIPTIONS, PersonaType, Turn
from neo4j_utils import load_schema
from prompts.eval_prompts import EVAL_USER_SIM_PROMPT, EVAL_USER_SIM_SYSTEM

logger = logging.getLogger(__name__)

_SCHEMA_PROBE_KEYWORDS = [
    "schema", "label", "relationship", "property", "properties",
    "node type", "edge type", "cypher", "query", "syntax",
    "database structure", "graph structure", "what tables",
    "what entities", "what nodes", "what relationships",
    "column", "field", "attribute",
]

_REJECTION_REPLIES = [
    "I'm not sure about the technical details of the database. Can you just try to answer my question?",
    "I don't know about the database schema. I just want to find the information I asked about.",
    "Sorry, I'm not familiar with how the data is structured. Could you figure that out yourself?",
    "I wouldn't know about that. I'm just a regular user looking for information.",
]


class UserSimulatorEval:
    """User simulator for agentic evaluation with anti-leakage mechanisms."""

    def __init__(self, persona: PersonaType, graph: str, gold_turns: List[Turn], client: BaseLLMClient):
        self.persona = persona
        self.graph = graph
        self.gold_turns = gold_turns
        self.client = client
        self.current_turn_idx = 0
        self.ask_count = 0
        self._rejection_idx = 0

        persona_info = PERSONA_DESCRIPTIONS.get(persona, {})
        self.persona_name = persona_info.get("name", persona.value)
        self.persona_traits = persona_info.get("traits", "")
        self._schema_terms = self._load_schema_terms(graph)

    def _load_schema_terms(self, graph: str) -> Set[str]:
        terms = set()
        try:
            schema = load_schema(graph)
            for entity in schema.entities:
                terms.add(entity.label.lower())
                for prop_name in entity.properties:
                    if prop_name.lower() not in {"name", "id", "type", "value", "date", "year", "count"}:
                        terms.add(prop_name.lower())
            for relation in schema.relations:
                terms.add(relation.label.lower())
        except Exception as e:
            logger.warning(f"Failed to load schema terms ({graph}): {e}")
        return terms

    def _classify_question_intent(self, question: str, current_turn: Optional[Turn]) -> str:
        q_lower = question.lower()
        for keyword in _SCHEMA_PROBE_KEYWORDS:
            if keyword in q_lower:
                return "REJECT"
        if re.search(r"\b(cypher|sql|sparql|gremlin|match\s*\(|return\s+\w)", q_lower):
            return "REJECT"
        if current_turn and getattr(current_turn, "ambiguity_info", None):
            ambiguous_term = current_turn.ambiguity_info.ambiguous_term.lower()
            if ambiguous_term in q_lower:
                return "DISAMBIGUATE"
            if re.search(r"\b(mean|definition|clarif|specific|which\s+one|interpret)", q_lower):
                return "DISAMBIGUATE"
        return "GENERAL"

    def _get_rejection_reply(self) -> str:
        reply = _REJECTION_REPLIES[self._rejection_idx % len(_REJECTION_REPLIES)]
        self._rejection_idx += 1
        return reply

    def _get_disambiguation_reply(self, question: str, current_turn: Turn) -> str:
        ai = getattr(current_turn, "ambiguity_info", None)
        if ai and ai.simulator_reply:
            return ai.simulator_reply
        if ai:
            correct_desc = ai.options.get(ai.correct_option, "")
            return f"I mean {correct_desc}."
        return "Let me clarify - I think my original question was clear enough."

    def _safety_filter(self, reply: str) -> str:
        reply_lower = reply.lower()
        for term in self._schema_terms:
            if len(term) > 3 and term in reply_lower:
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, reply_lower):
                    return "I'm not sure about the specific details. Can you try looking into it yourself?"
        return reply

    def respond(
        self, question: str, conversation_context: List[dict],
        last_model_result: Optional[str] = None,
    ) -> str:
        """Respond to an ASK_USER action with two-stage anti-leakage."""
        self.ask_count += 1
        current_turn = (
            self.gold_turns[self.current_turn_idx]
            if self.current_turn_idx < len(self.gold_turns) else None
        )

        intent = self._classify_question_intent(question, current_turn)
        if intent == "REJECT":
            return self._get_rejection_reply()
        if intent == "DISAMBIGUATE" and current_turn:
            return self._get_disambiguation_reply(question, current_turn)

        disambiguation_hints = "No special disambiguation information"
        if current_turn and getattr(current_turn, "ambiguity_info", None):
            ai = current_turn.ambiguity_info
            disambiguation_hints = f"The term '{ai.ambiguous_term}' means: {ai.options.get(ai.correct_option, '')}"

        goal_desc = current_turn.user_utterance if current_turn else "Complete the previous information query"
        domain_knowledge = f"I am querying data about the {self.graph} domain"

        history_text = ""
        for msg in conversation_context[-10:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            history_text += f"{role}: {content}\n"

        if last_model_result:
            history_text += f"\n[The assistant's last query returned: {last_model_result[:300]}]\n"
        if self.ask_count >= 3:
            history_text += "\n(Note: The user has been asked multiple times and is becoming impatient)\n"

        prompt = EVAL_USER_SIM_PROMPT.format(
            persona_name=self.persona_name, persona_traits=self.persona_traits,
            goal_description=goal_desc, disambiguation_hints=disambiguation_hints,
            domain_knowledge=domain_knowledge,
            conversation_history=history_text or "(Conversation just started)",
            assistant_question=question,
        )

        response, _, _ = self.client.call(
            [{"role": "system", "content": EVAL_USER_SIM_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=256,
        )

        if response is None:
            return "Sorry, I'm not sure how to answer that."

        reply = response.strip().strip('"').strip("'")
        for prefix in ["User:", "Response:"]:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

        return self._safety_filter(reply)

    def get_next_utterance(self, turn_idx: int) -> str:
        """Get the next user utterance for the given turn index."""
        self.current_turn_idx = turn_idx
        self.ask_count = 0
        if turn_idx < len(self.gold_turns):
            return self.gold_turns[turn_idx].user_utterance
        return ""

    def reset(self):
        self.current_turn_idx = 0
        self.ask_count = 0
        self._rejection_idx = 0
