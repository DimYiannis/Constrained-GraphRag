"""
    recall@k eval for the lexical retrieval pipeline
    against evaluation/test_queries.json
    Formula:
                                Number of Relevant Items in Top K
        Recall@K=       ------------------------------------------------------
                        Total Number of Relevant Items in Ground Truth Dataset
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.graph import load_dotenv
from src.retrieval import lexical

load_dotenv()

DEFAULT_TEST_QUERIES = Path("evaluation/test_queries.json")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_K_VALUES = (3, 5, 10)
DEFAULT_HOPS = 2

def _overlaps(a_first: int, a_last: int, b_first: int, b_last: int) -> bool:
    """
        true if two spans overlap
    """
    return a_first < b_last and a_last > b_first

def _is_hit(sources: list[dict], chunks: list[tuple[str, int, int]]) -> bool:
    """
        true if groound-truth source overlaps any retrieved chunk
    """
    