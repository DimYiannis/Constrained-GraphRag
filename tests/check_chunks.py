 #   PYTHONPATH=/Users/yiannis/Developer/graphrag uv run python tests/check_chunks.py
def check_file():
    from src.chunking.chunk_corpus import read_text, chunk

    text = read_text('data/raw/vllm-0.10.1/vllm/transformers_utils/tokenizer_group.py')
    chunks = chunk(text, 'vllm/transformers_utils/tokenizer_group.py')

    print(f'{len(chunks)} chunks total')
    for i, c in enumerate(chunks):
        print(f'--- chunk {i} ({c.source_type}, {c.first}:{c.last}) ---')
        print(c.text)
        print()

def check_folder():
    from src.chunking.chunk_corpus import read_text, chunk, iter_corpus_files
    from pathlib import Path

    root = Path('data/raw/vllm-0.10.1/vllm/lora')
    for path in iter_corpus_files(root):
        text = read_text(path)
        if text is None:
            continue
        for c in chunk(text, str(path)):
            print(f'{c.file_path} [{c.first}:{c.last}] ({c.source_type})')

if __name__ == "__main__":
    check_file()