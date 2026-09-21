"""Deterministic synthetic contexts with controlled length and distinct starts."""

import hashlib
import random


WORDS = (
    "amber", "cedar", "harbor", "quartz", "violet", "orbit", "river", "signal",
    "copper", "meadow", "summit", "lantern", "vector", "silver", "garden",
    "willow", "matrix", "forest", "cobalt", "bridge", "marble", "valley",
    "delta", "compass", "planet", "timber", "anchor", "magnet", "prairie",
    "coral", "tunnel", "falcon", "canvas", "frost", "island", "granite",
)


def make_prefix(identity: str, lines: int) -> str:
    """Give each identity its own seeded text, with equal line structure."""
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    rows = [f"Reference set {identity}. Use it only as background context.\n"]
    for number in range(1, lines + 1):
        terms = rng.sample(WORDS, 10)
        rows.append(
            f"Record {number:03d}: {terms[0]} {terms[1]} {terms[2]} links "
            f"{terms[3]} {terms[4]} to {terms[5]} {terms[6]}; "
            f"status {terms[7]}, marker {terms[8]}, owner {terms[9]}.\n"
        )
    return "".join(rows)


def question(index: int) -> str:
    # Equal shape and size; distinct terminal tokens preserve the common prefix.
    return (
        f"\nQuestion {index:03d}: Explain in three concise bullet points how to "
        "summarize a reference set for an engineer. Use 50 to 80 words.\nAnswer:"
    )


def prompt(prefix: str, index: int) -> str:
    return prefix + question(index)
