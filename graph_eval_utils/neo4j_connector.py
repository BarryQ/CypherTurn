"""Neo4j connector wrapper for query execution and graph introspection."""
from __future__ import annotations

from neo4j import GraphDatabase


class Neo4jConnector:
    """Thin wrapper around the Neo4j Python driver."""

    def __init__(self, name: str, host: str, port: int, username: str, password: str):
        self.name = name
        self.host = host
        self.port = port
        uri = f"bolt://{host}:{port}"
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def run_query(self, cypher: str, timeout: int = 120, **params) -> list:
        with self.driver.session() as session:
            result = session.run(cypher, parameters=params, timeout=timeout)
            return [dict(record) for record in result]

    def get_num_entities(self) -> int:
        result = self.run_query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_num_relations(self) -> int:
        result = self.run_query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0

    def close(self):
        self.driver.close()
