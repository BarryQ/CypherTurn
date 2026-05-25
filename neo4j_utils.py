"""Neo4j utilities for the GraphTurn evaluation framework."""
from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graph_eval_utils.neo4j_connector import Neo4jConnector
from graph_eval_utils.schema import DataType, PropertyGraphSchema, EntitySchema, RelationSchema

from config import GRAPHS_DIR, GRAPH_PORTS, NEO4J_HOST, NEO4J_PASSWORD, NEO4J_USERNAME

logger = logging.getLogger(__name__)

_connector_cache: Dict[str, Neo4jConnector] = {}
_schema_cache: Dict[str, PropertyGraphSchema] = {}


def get_connector(graph: str) -> Neo4jConnector:
    """Get a cached Neo4j connector for the given graph."""
    if graph not in _connector_cache:
        if graph not in GRAPH_PORTS:
            raise ValueError(f"Unknown graph: {graph}. Available: {list(GRAPH_PORTS.keys())}")
        port = GRAPH_PORTS[graph]
        _connector_cache[graph] = Neo4jConnector(
            name=graph, host=NEO4J_HOST, port=port,
            username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
        )
    return _connector_cache[graph]


def get_all_connectors() -> Dict[str, Neo4jConnector]:
    return {g: get_connector(g) for g in GRAPH_PORTS}


def close_all_connectors():
    for conn in _connector_cache.values():
        try:
            conn.close()
        except Exception:
            pass
    _connector_cache.clear()


def load_schema(graph: str) -> PropertyGraphSchema:
    """Load graph schema from JSON file."""
    if graph not in _schema_cache:
        schema_path = GRAPHS_DIR / graph / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        with open(schema_path) as f:
            schema_data = json.load(f)
        _schema_cache[graph] = _parse_schema(schema_data)
    return _schema_cache[graph]


def _parse_schema(data: dict) -> PropertyGraphSchema:
    type_map = {
        "string": DataType.STR, "str": DataType.STR,
        "int": DataType.INT, "integer": DataType.INT,
        "float": DataType.FLOAT,
        "bool": DataType.BOOL, "boolean": DataType.BOOL,
    }
    entities = []
    for node in data.get("nodes", []):
        props = {}
        for prop_name, prop_type in node.get("properties", {}).items():
            props[prop_name] = type_map.get(prop_type.lower(), DataType.STR)
        if "name" not in props:
            props["name"] = DataType.STR
        entities.append(EntitySchema(
            label=node["label"], properties=props,
            description=node.get("description", ""),
        ))
    relations = []
    for rel in data.get("relationships", []):
        props = {}
        for prop_name, prop_type in rel.get("properties", {}).items():
            props[prop_name] = type_map.get(prop_type.lower(), DataType.STR)
        relations.append(RelationSchema(
            label=rel["type"], subj_label=rel["source"], obj_label=rel["target"],
            properties=props, description=rel.get("description", ""),
        ))
    return PropertyGraphSchema(
        name=data.get("graph_name", "unknown"),
        entities=entities, relations=relations,
    )


def get_schema_text(graph: str) -> str:
    """Get full schema text for prompts."""
    schema = load_schema(graph)
    return schema.to_str(exclude_description=True)


def get_schema_summary(graph: str) -> str:
    """Get a brief schema summary (labels and relation types only)."""
    schema = load_schema(graph)
    entities = [e.label for e in schema.entities]
    relations = [f"{r.label}({r.subj_label} -> {r.obj_label})" for r in schema.relations]
    lines = ["Entity types: " + ", ".join(entities)]
    lines.append("Relationship types: " + ", ".join(relations))
    return "\n".join(lines)


def execute_cypher(
    connector: Neo4jConnector, cypher: str, timeout: int = 120,
) -> Tuple[Any, Optional[str]]:
    """Execute a Cypher query safely. Returns (result, error_message)."""
    start_time = time.time()
    try:
        result = connector.run_query(cypher, timeout=timeout)
        return result, None
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        if elapsed >= timeout - 1:
            error_msg = f"Query timeout ({timeout}s): {error_msg}"
        logger.debug(f"Cypher execution failed: {error_msg[:200]}")
        return None, error_msg


def inspect_schema(graph: str, filter_labels: Optional[List[str]] = None) -> str:
    """INSPECT_SCHEMA action: return schema text, optionally filtered."""
    schema = load_schema(graph)
    if filter_labels:
        filter_set = set(filter_labels)
        filtered = deepcopy(schema)
        filtered.entities = [e for e in filtered.entities if e.label in filter_set]
        filtered.relations = [
            r for r in filtered.relations
            if r.label in filter_set or r.subj_label in filter_set or r.obj_label in filter_set
        ]
        return filtered.to_str(exclude_description=True)
    return schema.to_str(exclude_description=True)


def search_values(
    connector: Neo4jConnector, label: str, property_name: str = "name",
    query: str = "", limit: int = 10,
) -> List[str]:
    """SEARCH_VALUES action: fuzzy-search entity property values."""
    try:
        if not query or not query.strip():
            cypher = (
                f"MATCH (n:{label}) WHERE n.{property_name} IS NOT NULL "
                f"RETURN DISTINCT n.{property_name} AS value LIMIT {limit}"
            )
            results = connector.run_query(cypher)
            return [r["value"] for r in results if r.get("value") is not None]

        cypher_contains = (
            f"MATCH (n:{label}) "
            f"WHERE toLower(n.{property_name}) CONTAINS toLower($search_term) "
            f"RETURN DISTINCT n.{property_name} AS value LIMIT {limit}"
        )
        results = connector.run_query(cypher_contains, search_term=query)
        values = [r["value"] for r in results if r.get("value") is not None]
        if values:
            return values

        cypher_prefix = (
            f"MATCH (n:{label}) "
            f"WHERE toLower(n.{property_name}) STARTS WITH toLower($search_term) "
            f"RETURN DISTINCT n.{property_name} AS value LIMIT {limit}"
        )
        results = connector.run_query(cypher_prefix, search_term=query)
        values = [r["value"] for r in results if r.get("value") is not None]
        if values:
            return values

        words = query.strip().split()
        if len(words) > 1:
            for word in words:
                if len(word) < 3:
                    continue
                cypher_word = (
                    f"MATCH (n:{label}) "
                    f"WHERE toLower(n.{property_name}) CONTAINS toLower($search_term) "
                    f"RETURN DISTINCT n.{property_name} AS value LIMIT {limit}"
                )
                results = connector.run_query(cypher_word, search_term=word)
                values = [r["value"] for r in results if r.get("value") is not None]
                if values:
                    return values
        return []
    except Exception as e:
        logger.warning(f"SEARCH_VALUES failed: {e}")
        return []


def serialize_result(result: Any, max_rows: int = 50) -> str:
    """Serialize query result to readable string for conversation context."""
    if result is None:
        return "(no results)"
    if not result:
        return "(empty result set)"
    if isinstance(result, list):
        if len(result) > max_rows:
            truncated = result[:max_rows]
            return json.dumps(truncated, ensure_ascii=False, default=str) + \
                   f"\n... (Total {len(result)} rows, showing first {max_rows})"
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)
