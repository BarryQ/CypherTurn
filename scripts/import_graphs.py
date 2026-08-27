"""Import graph data into Neo4j instances.

Usage:
    python scripts/import_graphs.py --graph ancient_empire
    python scripts/import_graphs.py --all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_GRAPHS, GRAPHS_DIR
from neo4j_utils import get_connector


def import_graph(graph: str):
    """Import a single graph's data.cypher into its Neo4j instance."""
    cypher_file = GRAPHS_DIR / graph / "data.cypher"
    if not cypher_file.exists():
        print(f"ERROR: {cypher_file} not found")
        return False

    print(f"Importing {graph}...")
    connector = get_connector(graph)

    # Clear existing data
    connector.run_query("MATCH (n) DETACH DELETE n")

    # Execute import statements
    with open(cypher_file, encoding="utf-8") as f:
        statements = f.read().split(";")

    count = 0
    for stmt in statements:
        stmt = stmt.strip()
        if stmt:
            try:
                connector.run_query(stmt)
                count += 1
            except Exception as e:
                print(f"  Warning: statement failed: {str(e)[:100]}")

    # Verify
    num_nodes = connector.get_num_entities()
    num_rels = connector.get_num_relations()
    print(f"  Done: {count} statements, {num_nodes} nodes, {num_rels} relationships")
    return True


def main():
    parser = argparse.ArgumentParser(description="Import graph data into Neo4j")
    parser.add_argument("--graph", type=str, help="Graph name to import")
    parser.add_argument("--all", action="store_true", help="Import all graphs")
    args = parser.parse_args()

    if args.all:
        for graph in ALL_GRAPHS:
            import_graph(graph)
    elif args.graph:
        if args.graph not in ALL_GRAPHS:
            print(f"Unknown graph: {args.graph}. Available: {ALL_GRAPHS}")
            sys.exit(1)
        import_graph(args.graph)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
