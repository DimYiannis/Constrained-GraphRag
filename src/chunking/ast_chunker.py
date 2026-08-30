import ast
from .spans import Chunk, _to_chunks, _split_span


# abstract syntax tree definitions to handle code smoothly
PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
    source_type: str = "code",
) -> list[Chunk]:
    """
        chunk python script, each top level func/class is a chunk
        a class larger than the cap is re_chunked per method, its
        header is a seperate chunk. Files that fail to parse fall
        back to line-window chunking

        args:
            text
            file_path
            max_chunk_size
            source_type: forwarded to the syntax-error fallback too, so a
                .py file that fails to parse still comes out as "code".

        return:
            chunks
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_lines(text, file_path, max_chunk_size, source_type)
    line_starts = _line_starts(text)
    spans = _split_body(
        tree.body,
        0,
        len(text),
        text,
        line_starts,
        max_chunk_size
    )
    return _to_chunks(text, file_path, spans, source_type)


def chunk_lines(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
    source_type: str = "text",
) -> list[Chunk]:
    """
        chunk text into windows at line boundaries
        fallback for unstructured files and unparsable python

        args:
            text
            file_path
            max_chunk_size
            source_type

        return:
            chunks
    """

    spans = _split_span(text, 0, len(text), max_chunk_size)
    return _to_chunks(text, file_path, spans, source_type)


def _split_body(
    body: list[ast.stmt],
    region_start: int,
    region_end: int,
    text: str,
    line_starts: list[int],
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    """
        split code into spans

        args:
            body
            region_start
            region_end
            text
            line_starts
            max_chunk_size

         return:
            spans -> [region_start, region_end)
    """
    spans: list[tuple[int, int]] = []
    cursor = region_start
    for node in body:
        if not isinstance(node, PY_DEFS):
            continue
        start, end = _node_span(node, line_starts)
        if start > cursor:
            spans.extend(_split_span(text, cursor, start, max_chunk_size))
        oversized_class = (
            isinstance(node, ast.ClassDef) and end - start > max_chunk_size
        )
        if oversized_class:
            spans.extend(
                _split_body(
                    node.body, start, end, text, line_starts, max_chunk_size
                )
            )
        else:
            spans.extend(_split_span(text, start, end, max_chunk_size))
        cursor = end
    if cursor < region_end:
        spans.extend(_split_span(text, cursor, region_end, max_chunk_size))
    return spans


def _node_span(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    line_starts: list[int],
) -> tuple[int, int]:
    """
        char span of a definition, decorators included.

        the span starts at column 0 of the first decorator's line (or the
        def/class line), so indentation stays inside the chunk.

        args:
            node: the definition node.
            line_starts

        return:
            (first, last)
    """
    first_node: ast.expr | ast.stmt = node
    if node.decorator_list:
        first_node = node.decorator_list[0]
    end_lineno = node.end_lineno if node.end_lineno else node.lineno
    end_col = node.end_col_offset if node.end_col_offset else 0
    start = line_starts[first_node.lineno - 1]
    end = line_starts[end_lineno - 1] + end_col
    return start, end

def _line_starts(text: str) -> list[int]:
    """
        char offset of every line start, index i = 0-based line i.

        args:
            text

        return:
            offsets list offset = starts[lineno - 1] converts an ast
            1-based lineno to a character position.
    """
    starts = [0]
    pos = text.find("\n", 0)
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts