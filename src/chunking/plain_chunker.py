import re
from .spans import Chunk, _to_chunks, _split_span

HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
MIN_SECTION_SIZE = 600

def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
    source_type: str = "text",
) -> list[Chunk]:
    """
        chunk markdown like text,
        treat each ATX header (``#`` .. ``######``) as the start
        of a new section, so a header always stays attached to its own
        body rather than being separated from it.
        Sections shorter than MIN_SECTION_SIZE are merged into the previous
        section (as long as the merge stays within max_chunk_size),
        since a short section on its own is too small to usefully overlap.
        Sections still over max_chunk_size after merging are split at
        paragraph, then line, boundaries.
        A file with no headers is treated as one section spanning the
        whole file, which then falls through the same oversize-split
        path as a fixed, non-overlapping window.

        args:
            text
            file_path
            max_chunk_size

        return:
            chunks in file order
    """
    # handle headers
    bounds = [match.start() for match in HEADER_RE.finditer(text)]
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    bounds.append(len(text))

    # build sections then merge undersized ones
    merged: list[list[int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if merged:
            prev_start, prev_end = merged[-1]
            small = (
                end - start < MIN_SECTION_SIZE
                or prev_end - prev_start < MIN_SECTION_SIZE
            )
            if small and end - prev_start <= max_chunk_size:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    # enforce size cap and return
    spans: list[tuple[int, int]] = []
    for start, end in merged:
        spans.extend(_split_span(text, start, end, max_chunk_size))
    return _to_chunks(text, file_path, spans, source_type)
