"""
    identifier-aware tokenizer shared by indexing and querying.

    Identifiers are kept whole AND split into subtokens
    -> (enable_lora -> enable_lora, enable, lora):
    questions either quote whole token matches or paraphrase
    them subtokens match.
    This dual emission is the main lever for the code-recall target.
"""

import re

# both compiled patterns precompiled at module load (re.compile)
# so they're not recompiled per call, cheap to reuse across every chunk/query.

# matches a maximal run of letters/digits/underscore
WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# four alternatives, tried in order, used to split a single identifier
# piece into CamelCase/acronym-aware subwords:
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Standard english stopword list, stripped from QUERIES only.
# Every question contains these, so they cannot separate
# one chunk from another - yet BM25 still adds a document-dependent amount
# for each, which ranks on whichever chunk repeats them most.
# Long source files do exactly that, which is how prose questions ended
# up answered by .py files.
# Deliberately the ordinary english list so it transfers to any corpus.
# Single-character words are absent because _keep already drops them.
STOPWORDS = frozenset("""
about above after again against all am an and any are as at be because been
before being below between both but by can did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which
while who whom why will with you your yours yourself yourselves
""".split())


def tokenize(text: str) -> list[str]:
    """
        tokenize text for indexing, querying

        used the compiled patterns to split properly

        args: text

        return:
            tokens in occurence order
    """
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        word = match.group()
        lowered = word.lower()
        if _keep(lowered):
            tokens.append(lowered)
        parts = [
            part
            # underscore fragments 'enable_HTTPServer'
            for piece in word.split("_")
            # scan fragments for capitals, [enable, HTTPServer]
            for part in CAMEL_RE.findall(piece)
        ]
        for part in parts:
            part_lower = part.lower()
            if part_lower != lowered and _keep(part_lower):
                tokens.append(part_lower)
    return tokens


def strip_stopwords(terms: list[str]) -> list[str]:
    """
        drop english stopwords from a query's term list

        args:
            terms: tokenized query terms

        return:
            the discriminative terms, or the original list when every
            term is a stopword - a degraded ranking beats no results
    """
    kept = [term for term in terms if term not in STOPWORDS]
    return kept or terms


def _keep(token: str) -> bool:
    """
        filter out noise

        args:
            token: lowercased candidate token

        return:
            false for single chars and pure nums, true otherwise
    """
    return len(token) > 1 and not token.isdigit()
