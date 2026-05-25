# GraphTurn: A Multi-Turn Benchmark for Conversational Text-to-Cypher

GraphTurn is the first benchmark for evaluating multi-turn conversational Text-to-Cypher generation. It comprises 721 sessions and 5,927 turns across 7 purpose-built knowledge graphs, annotated with 13 conversational phenomena and 10 user personas.

## Key Statistics

| Metric | Value |
|--------|-------|
| Sessions | 721 |
| Turns | 5,927 |
| Knowledge Graphs | 7 |
| Conversational Phenomena | 13 |
| User Personas | 10 |
| Mean Turns/Session | 8.2 |
| Inter-annotator Agreement | Cohen's κ = 0.83 |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Neo4j Instances

Each of the 7 graphs requires its own Neo4j instance. Use Docker Compose:

```bash
cd docker
export NEO4J_PASSWORD=password
docker compose up -d
```

### 3. Import Graph Data

```bash
python scripts/import_graphs.py --all
```

### 4. Configure Your Model API

Set environment variables for your model:

```bash
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://api.openai.com/v1
export NEO4J_HOST=localhost
export NEO4J_PASSWORD=password
```

## Running Evaluation

### Guided Protocol

The model receives gold-annotated history at each turn (oracle context), with a budget of 3 actions per turn:

```bash
python -m eval.evaluator \
    --model gpt-4o \
    --protocol guided \
    --output_dir output/guided_gpt4o
```

### Agentic Protocol

The model operates autonomously with a shared action budget (turns × multiplier). No oracle context is provided:

```bash
python -m eval.evaluator \
    --model gpt-4o \
    --protocol agentic \
    --budget 3 \
    --sim_model gpt-4o \
    --output_dir output/agentic3_gpt4o
```

Budget multiplier options: `3` (default), `5`, or `10`.

### Using a Local Model (vLLM)

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=dummy
python -m eval.evaluator --model my-model --protocol guided --output_dir output/guided_local
```

## Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **EX** | Turn | Execution Accuracy: 1 if predicted and gold query produce identical result sets |
| **PSJS** | Turn | Provenance Subgraph Jaccard Similarity: overlap of node-IDs bound during pattern matching |
| **CER** | Turn | Chain Error Rate: P(EX_t=0 \| EX_{t-1}=0) for chain-dependent turns |
| **SEM** | Session | Session Exact Match: 1 iff all turns achieve EX=1 |

## Data Format

Benchmark data is split into 7 files (one per graph) in `data/scenarios/`:

```
data/scenarios/
├── ancient_empire_sessions.json
├── arcane_archive_sessions.json
├── celestial_court_sessions.json
├── magic_academy_sessions.json
├── merchant_harbor_sessions.json
├── ocean_kingdom_sessions.json
└── stellar_colony_sessions.json
```

Each file contains a JSON array of sessions. Each session has:
- `session_id`: unique identifier
- `graph`: knowledge graph name
- `persona`: user persona type
- `turns`: array of turn objects

Each turn contains:
- `user_utterance`: natural language question
- `gold_cypher`: ground-truth Cypher query
- `gold_answer`: execution result of the gold query
- `phenomena`: list of conversational phenomenon tags
- `depends_on_result_of`: turn ID this turn depends on (null for independent turns)
- `chain_mode`: how this turn relates to its predecessor

## Project Structure

```
.
├── config.py              # Configuration (env-var based)
├── models.py              # Data models (Pydantic)
├── api_client.py          # LLM client (OpenAI-compatible)
├── neo4j_utils.py         # Neo4j utilities and action space
├── eval/
│   ├── evaluator.py       # Evaluation orchestrator (entry point)
│   ├── guided_runner.py   # Guided protocol implementation
│   ├── agentic_runner.py  # Agentic protocol implementation
│   ├── user_sim_eval.py   # User simulator for agentic evaluation
│   └── metrics.py         # EX, PSJS, CER, SEM computation
├── prompts/
│   └── eval_prompts.py    # Prompt templates
├── graph_eval_utils/      # Bundled evaluation utilities
├── data/
│   ├── scenarios/         # Benchmark data (7 per-graph files)
│   └── graphs/            # Graph schemas and import data
├── docker/
│   └── docker-compose.yml # Neo4j container definitions
├── scripts/
│   ├── import_graphs.py   # Graph data import script
│   └── run_eval.sh        # Example evaluation commands
└── requirements.txt
```

## Knowledge Graphs

| Graph | Topology | Nodes | Edges |
|-------|----------|-------|-------|
| Ancient Empire | Directed cycle with numeric edge properties | 440 | 572 |
| Arcane Archive | Deep citation chain | 420 | 897 |
| Celestial Court | Hierarchical DAG with self-reference | 451 | 770 |
| Magic Academy | Bipartite cycle | 378 | 813 |
| Merchant Harbor | Tripartite hub | 490 | 1,059 |
| Ocean Kingdom | Diamond convergence | 369 | 483 |
| Stellar Colony | Tree hierarchy with lateral alliances | 430 | 607 |

## License

This benchmark is released for research purposes under the terms described in the accompanying paper.
