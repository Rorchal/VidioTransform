"""Token estimation.

A deterministic char-based estimate (~4 chars/token for English/code). It is
only used to compare methods against each other under the same budget, so the
absolute number does not need to match any particular tokenizer. Swap in a real
tokenizer here if you need absolute numbers.
"""


def count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
