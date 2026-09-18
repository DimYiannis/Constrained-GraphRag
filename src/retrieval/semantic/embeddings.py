"""
    semantic search with dense retrieval
    relevance is based on cosine similarity amongst
    the vectors generated from the embedding
    model
"""

from pathlib import Path
from typing import cast

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentnenceTransformer

