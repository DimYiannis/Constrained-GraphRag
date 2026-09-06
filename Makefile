.PHONY: install neo4j-up neo4j-down neo4j-logs index search answer test clean

install:
	uv sync

# Neo4j via Docker - see docker-setup.md for first-time setup/troubleshooting.
neo4j-up:
	docker start graphrag-neo4j || docker run -d \
		--name graphrag-neo4j \
		-p 7474:7474 -p 7687:7687 \
		-e NEO4J_AUTH=neo4j/$${NEO4J_PASSWORD:-changeme} \
		neo4j:latest

neo4j-down:
	docker stop graphrag-neo4j

neo4j-logs:
	docker logs -f graphrag-neo4j

# Chunk a corpus and build the BM25 index. Override DATA_DIR to point at
# a different corpus root, e.g. make index DATA_DIR=data/raw/vllm-0.10.1
DATA_DIR ?= data/raw/vllm-0.10.1
index:
	uv run python -m src index --data_directory $(DATA_DIR)

# make search QUERY="enable lora"
search:
	uv run python -m src search "$(QUERY)" --k $${K:-5}

# make answer QUERY="how does lora work"
answer:
	uv run python -m src answer "$(QUERY)" --data_directory $(DATA_DIR)

test:
	uv run pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
