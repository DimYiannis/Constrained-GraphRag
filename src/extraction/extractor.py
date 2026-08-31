"""
    runs Qwen3-0.6B, constrained by Outlines, over a chunk to 
    extract entities/nodes-relationships/edges

    No logic for invalid schema, we do not need to validate after generation
    because Outlines' FSM makes invalid schema output impossible to sample
    so there is nothing to reject and retry against
"""

import outlines
from transformers import AutoModelCasualLM, AutoTokenizer

from src.chunking.spans import Chunks
from src.extraction.prompts import build_prompt
from src.extraction.schema import ExtractionResult

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512

def load_model(model: str = DEFAULT_MODEL_NAME):
    """
        wrap a Hugging face causal LM in an Outlines model
    """
    # example wrap a Transformers model + tokenizer
    """
    model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    """
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return Outlines.from_transformers(hf_model, hf_tokenizer)

def build_generator(model):
    """
        build a reusable constrained generator for ExtractionResults
        
        build once, called per chunk, avoids recompiling the FSM
        constraint for every one of all the chunks.
    """
    generator = outlines.Generator(model, output_type=ExtractionResult)
    return generator 

def extract(
    generator, chunk: Chunk, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> ExtractResult:
    """
        extract nodes/edges from one chunk

        args:
            generator: constrained generator for entities/relationships
            chunk: routed to the code/text prompt by chunk.source_type
            max_new_tokens

        return:
            validated ExtractionResult -> nodes, edges
    """
    prompt = build_prompt(chunk)
    raw = generator(prompt, max_new_tokens=max_new_tokens)
    return ExtractionResult.model_validate_json(raw)
