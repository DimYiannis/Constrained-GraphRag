from pathlib import Path

from .ast_chunker import chunk_python, chunk_lines
from .plain_chunker import chunk_markdown
from .spans import Chunk


TEXT_EXTE = {
    ".py", ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".jinja", ".cmake", ".in",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh",
    ".js", ".html", ".css",
}

# extensions routed to the "code" extraction prompt (source_type). Everything
# else in TEXT_EXTE - docs, config/data files - routes to "text": config
# files aren't prose, but they're not code with function calls either, and
# "text" degrades more gracefully than "code" would for them.
CODE_EXTE = {
    ".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".js",
}


def read_text(path:Path) -> str | None:
    """
        read a file as utf-8
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeError, OSError):
        return None

def iter_corpus_files(root: Path) -> list[Path]:
    """
        list indexable files under a corpus root
        deterministically
    """
    return sorted(
        path

        # matches pattern against every file/dir at any depth under root
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_EXTE
    )

def chunk(
    text:str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """
        chunk a file's content with the matching startegy
        based on the files type
    """
    suffix = Path(file_path).suffix.lower()
    source_type = "code" if suffix in CODE_EXTE else "text"
    if suffix == ".py":
        return chunk_python(text, file_path, max_chunk_size, source_type)
    if suffix in {".md", ".rst", ".txt"}:
        return chunk_markdown(text, file_path, max_chunk_size, source_type)
    return chunk_lines(text, file_path, max_chunk_size, source_type)
    