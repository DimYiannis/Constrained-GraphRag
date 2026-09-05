# uv run pytest tests/test_extraction_quality.py -v -s
import re
import pytest
import src.__main__
from src.chunking.chunk_corpus import read_text, chunk
from src.extraction import extractor
PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

@pytest.fixture(scope="module")
def generator():
    model = extractor.load_model()
    return extractor.build_generator(model)

def test_constraint(generator):
  text = read_text('data/raw/vllm-0.10.1/vllm/transformers_utils/tokenizer_group.py')
  chunks = chunk(text, 'vllm/transformers_utils/tokenizer_group.py')
  target = chunks[0]

  result = extractor.extract(generator, target, max_new_tokens=600)
  for rel in result.relationships:
    assert PATTERN.match(rel.subject), rel.subject
    assert PATTERN.match(rel.target), rel.target