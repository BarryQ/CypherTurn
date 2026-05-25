#!/bin/bash
# Example: Run GraphTurn evaluation
#
# Prerequisites:
#   1. Neo4j instances running (see docker/docker-compose.yml)
#   2. Graph data imported (python scripts/import_graphs.py --all)
#   3. Environment variables set for your model API

export NEO4J_HOST=localhost
export NEO4J_PASSWORD=password
export LLM_API_KEY=your-api-key-here
export LLM_BASE_URL=https://api.openai.com/v1

# Guided protocol evaluation
python -m eval.evaluator \
    --scenarios_dir data/scenarios \
    --model gpt-4o \
    --protocol guided \
    --output_dir output/guided_gpt4o \
    --num_workers 4

# Agentic protocol evaluation (budget x3)
python -m eval.evaluator \
    --scenarios_dir data/scenarios \
    --model gpt-4o \
    --protocol agentic \
    --budget 3 \
    --sim_model gpt-4o \
    --output_dir output/agentic3_gpt4o \
    --num_workers 4
