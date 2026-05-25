"""GraphTurn core data models."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class Phenomenon(str, Enum):
    """13 graph-native conversational phenomena."""
    EXPAND = "EXPAND"
    FILTER = "FILTER"
    PIVOT = "PIVOT"
    COUNT = "COUNT"
    FIRST = "FIRST"
    REFINE = "REFINE"
    CONTRAST = "CONTRAST"
    TOPIC_SHIFT = "TOPIC_SHIFT"
    VALUE_FILTER = "VALUE_FILTER"
    MULTI_CONDITION = "MULTI_CONDITION"
    AGG_SUM = "AGG_SUM"
    AGG_AVG = "AGG_AVG"
    AGG_MAX = "AGG_MAX"


class ActionType(str, Enum):
    """5 actions available to the evaluated model."""
    EXECUTE_CYPHER = "EXECUTE_CYPHER"
    ASK_USER = "ASK_USER"
    INSPECT_SCHEMA = "INSPECT_SCHEMA"
    SEARCH_VALUES = "SEARCH_VALUES"
    SUBMIT_ANSWER = "SUBMIT_ANSWER"


class PersonaType(str, Enum):
    """10 user personas governing utterance register."""
    JOURNALIST = "journalist"
    STUDENT = "student"
    PRODUCT_MANAGER = "product_manager"
    NOVICE_USER = "novice_user"
    DATA_ANALYST = "data_analyst"
    CASUAL_USER = "casual_user"
    IMPATIENT_EXEC = "impatient_exec"
    ADVERSARIAL = "adversarial"
    NON_NATIVE = "non_native"
    VERBOSE_USER = "verbose_user"


PERSONA_DESCRIPTIONS = {
    PersonaType.JOURNALIST: {
        "name": "Investigative Journalist",
        "language_style": "formal, detail-oriented, probing",
        "info_density": "medium",
        "tolerance": "low (demands precision)",
        "traits": "Asks pointed questions, probes for specific numbers, uses varied journalistic openers"
    },
    PersonaType.STUDENT: {
        "name": "Curious Student",
        "language_style": "simple, curious, direct",
        "info_density": "low",
        "tolerance": "high",
        "traits": "Simple language, curious tone, uses 'I wonder', 'can you show me', short sentences"
    },
    PersonaType.PRODUCT_MANAGER: {
        "name": "Product Manager",
        "language_style": "business-oriented, metric-focused",
        "info_density": "medium",
        "tolerance": "medium",
        "traits": "Asks from business angle, cares about KPIs and trends, uses business vocabulary"
    },
    PersonaType.NOVICE_USER: {
        "name": "Novice User",
        "language_style": "colloquial, vague, imprecise",
        "info_density": "low",
        "tolerance": "high",
        "traits": "Colloquial expressions, may use wrong terms, vague references, friendly tone"
    },
    PersonaType.DATA_ANALYST: {
        "name": "Data Analyst",
        "language_style": "metric-oriented, precise, tabular thinking",
        "info_density": "high",
        "tolerance": "medium",
        "traits": "Focuses on aggregations and sorting, prefers distinct/grouped results, analytical tone"
    },
    PersonaType.CASUAL_USER: {
        "name": "Casual User",
        "language_style": "terse, colloquial, context-dependent",
        "info_density": "low",
        "tolerance": "high",
        "traits": "Short natural sentences, contractions, relies heavily on context, conversational"
    },
    PersonaType.IMPATIENT_EXEC: {
        "name": "Impatient Executive",
        "language_style": "commanding, ultra-concise",
        "info_density": "medium",
        "tolerance": "low",
        "traits": "Direct commands, no pleasantries, wants raw data fast, occasional urgency markers"
    },
    PersonaType.ADVERSARIAL: {
        "name": "Adversarial Tester",
        "language_style": "edge cases, challenging, skeptical",
        "info_density": "high",
        "tolerance": "low",
        "traits": "Poses challenging queries, expresses doubt, uses questioning suffixes"
    },
    PersonaType.NON_NATIVE: {
        "name": "Non-Native English Speaker",
        "language_style": "simple English, occasional grammar imperfections",
        "info_density": "low",
        "tolerance": "high",
        "traits": "Simple vocabulary, shorter sentences, occasional article/preposition errors, clear intent"
    },
    PersonaType.VERBOSE_USER: {
        "name": "Verbose User",
        "language_style": "over-detailed, redundant context",
        "info_density": "high",
        "tolerance": "high",
        "traits": "Provides excessive background, repeats the question, uses multiple clauses, over-explains"
    },
}


class Protocol(str, Enum):
    GUIDED = "guided"
    AGENTIC = "agentic"


# ═══════════════════════════════════════════════════════════════════════════
# Query skeleton types (stored in benchmark data)
# ═══════════════════════════════════════════════════════════════════════════

class BlockType(str, Enum):
    ORIGIN = "origin"
    TRAVERSE = "traverse"
    FILTER = "filter"
    SORT_LIMIT = "sort_limit"
    COUNT = "count"
    PROJECT = "project"
    ANAPHORA = "anaphora"


class ChainMode(str, Enum):
    NARROW = "narrow"
    EXPAND = "expand"
    AGGREGATE = "aggregate"
    CONTRAST = "contrast"
    PIVOT = "pivot"
    TOPIC_SHIFT = "topic_shift"
    VALUE_NARROW = "value_narrow"
    AGG_NUMERIC = "agg_numeric"


class QueryBlock(BaseModel):
    block_type: BlockType
    label: Optional[str] = None
    relation_type: Optional[str] = None
    direction: str = "out"
    property_name: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[Any] = None
    sort_order: str = "DESC"
    limit_k: Optional[int] = None
    ref_turn_id: Optional[int] = None
    ref_position: Optional[str] = None
    count_distinct: bool = True
    agg_func: Optional[str] = None
    extra_conditions: Optional[list] = None
    is_edge_filter: bool = False
    edge_filter_rel: Optional[str] = None


class QuerySkeleton(BaseModel):
    blocks: List[QueryBlock] = []
    chain_mode: Optional[ChainMode] = None
    cypher_template: Optional[str] = None
    actual_cypher: Optional[str] = None
    result_size: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════
# Action and result models
# ═══════════════════════════════════════════════════════════════════════════

class Action(BaseModel):
    action_type: ActionType
    payload: str
    raw_llm_output: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: Optional[str] = None


class ActionResult(BaseModel):
    action_type: ActionType
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# Turn and Session models
# ═══════════════════════════════════════════════════════════════════════════

class Turn(BaseModel):
    turn_id: int
    user_utterance: str
    phenomena: List[Phenomenon] = []
    gold_cypher: str = ""
    gold_answer: Optional[Any] = None
    gold_answer_json: Optional[str] = None
    depends_on_result_of: Optional[int] = None
    reference_description: Optional[str] = None
    skeleton: Optional[QuerySkeleton] = None
    chain_mode: Optional[ChainMode] = None
    pred_actions: List[Action] = []
    pred_action_results: List[ActionResult] = []
    pred_cypher: Optional[str] = None
    budget_used: int = 0
    metrics: Dict[str, float] = {}


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    graph: str
    protocol: Protocol = Protocol.GUIDED
    persona: PersonaType = PersonaType.JOURNALIST
    turns: List[Turn] = []
    metadata: Dict[str, Any] = {}
    split: Optional[str] = None
    session_metrics: Dict[str, float] = {}
    model_name: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════
# Budget configuration
# ═══════════════════════════════════════════════════════════════════════════

class BudgetConfig(BaseModel):
    guided_per_turn: int = 3
    agentic_total: int = 15
    cypher_timeout_sec: int = 120
