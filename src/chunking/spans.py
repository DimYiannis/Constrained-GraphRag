from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    """
        retrievable span from a corpus file
    """

    file_path: str
    first: int
    last: int
    text: str


def _split_span(
    text: str, first: int, last: int, max_chunk_size: int
) -> list[tuple[int, int]]:
    """
        cut a span into pieces no longer than the cap.

        prefers cutting at a blank line,
        then any newline,
        then hard mid-line (single lines longer than the cap).

        args:
            text
            first
            last
            max_chunk_size

        return:
            (first, last)
    """

    spans: list[tuple[int, int]] = []
    start = first
    while last - start > max_chunk_size:
        window_end = start + max_chunk_size
        cut = text.rfind("\n\n", start + 1, window_end)
        if cut <= start:
            cut = text.rfind("\n", start + 1, window_end)
        if cut <= start:
            cut = window_end
        spans.append((start, cut))
        start = cut
    if start < last:
        spans.append((start, last))
    return spans



def _to_chunks(
    text: str, file_path: str, spans: list[tuple[int, int]]
) -> list[Chunk]:
    """
        materialize spans as chunks, dropping whitespace-only ones.

        args:
            text
            file_path
            spans: (first, last) pairs.

        returns:
            chunks whose text contains at least one non-space character.
    """
    chunks = []
    for first, last in spans:
        piece = text[first:last]
        if piece.strip():
            chunks.append((Chunk(file_path, first, last, piece)))
    return chunks

