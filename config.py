"""GraphTurn configuration. All credentials via environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_DIR = DATA_DIR / "scenarios"
GRAPHS_DIR = DATA_DIR / "graphs"

NEO4J_HOST = os.environ.get("NEO4J_HOST", "localhost")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

GRAPH_PORTS = {
    "stellar_colony": 7687,
    "magic_academy": 7688,
    "ocean_kingdom": 7689,
    "ancient_empire": 7690,
    "arcane_archive": 7691,
    "merchant_harbor": 7692,
    "celestial_court": 7693,
}
ALL_GRAPHS = list(GRAPH_PORTS.keys())

GUIDED_BUDGET_PER_TURN = 3
AGENTIC_BUDGET_MULTIPLIERS = [3, 5, 10]
CYPHER_TIMEOUT_SEC = 120
