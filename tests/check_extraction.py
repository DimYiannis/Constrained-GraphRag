#   PYTHONPATH=/Users/yiannis/Developer/graphrag uv run python tests/check_extraction.py
from src.chunking.chunk_corpus import read_text, chunk
from src.extraction import extractor

def check_file(chunk_index=0):
    file_path =     text = 'data/raw/vllm-0.10.1/vllm/transformers_utils/tokenizer_group.py'
    text = read_text(file_path)
    chunks = chunk(text,file_path)
    target = chunks[chunk_index]

    print(f"chunk ({target.source_type}, {target.first}:{target.last}):")
    print(target.text)
    print()

    model = extractor.load_model()
    generator = extractor.build_generator(model)
    result = extractor.extract(generator, chunk, 600)

    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    check_file()