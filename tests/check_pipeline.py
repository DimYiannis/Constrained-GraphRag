#  PYTHONPATH=/Users/yiannis/Developer/graphrag uv run python tests/check_pipeline.py
import src.__main__  # loads .env
from pathlib import Path
from src.pipeline import index_pipeline

if __name__ == "__main__":
    processed = index_pipeline.run(
        Path('data/raw/vllm-0.10.1/vllm/lora'),
        limit=2,
    )
    print(f"processed {processed} chunks")