#!/bin/bash
# Wrapper script for Memory-Graph Linkage Inspector
# Automatically sets correct ArangoDB credentials

cd "$(dirname "$0")"

# Set correct credentials for Docker ArangoDB
export ARANGODB_PASSWORD="${ARANGODB_PASSWORD:-dappy_dev_password}"
export ARANGODB_DATABASE="dappy_memories"

# Activate venv and run inspector
source .venv/bin/activate
python utils/inspect_memory_graph_linkage.py "$@"

