"""RULER niah_single_1 prompts with the needle d tokens before the question.

Text from the RULER repository, with its single-needle template. Each haystack line is a random
permutation of the five noise sentences (25 tokens), so L and d are multiples of 25.

    A (instruction first):          INSTRUCTION · L haystack · NEEDLE · d haystack · QUESTION
    B (instruction before needle):  L haystack · INSTRUCTION · NEEDLE · d haystack · QUESTION
    C (no instruction):             L haystack · NEEDLE · d haystack · QUESTION

The prefix before the needle depends only on the haystack sample and L.
"""
from __future__ import annotations

import random

INSTRUCTION = ("A special magic number is hidden within the following text. Make sure to memorize it. "
               "I will quiz you about the number afterwards.\n")
NOISE_SENTENCES = ["The grass is green.", "The sky is blue.", "The sun is yellow.", "Here we go.",
                   "There and back again."]
NEEDLE = "One of the special magic numbers for {key} is: {value}.\n"
QUESTION = ("What is the special magic number for {key} mentioned in the provided text?"
            " The special magic number for {key} mentioned in the provided text is")
LINE_TOKENS = 25
MAX_LINES = 8192            # longest haystack piece: 8192 lines = 204,800 tokens

# RULER draws keys as "adjective-noun" from the wonderwords lists; we draw from these short lists.
ADJECTIVES = ["abundant", "agreeable", "ancient", "brave", "calm", "careful", "cheerful", "clever",
              "curious", "delightful", "eager", "elegant", "fancy", "gentle", "glorious", "handsome",
              "jolly", "kind", "lively", "nervous", "obedient", "polite", "proud", "quiet", "silly",
              "smooth", "thankful", "tidy", "victorious", "witty", "wooden", "zealous"]
NOUNS = ["glove", "harbor", "pencil", "garden", "lantern", "violin", "blanket", "meadow", "hammer",
         "kettle", "orchard", "saddle", "teapot", "tunnel", "wagon", "whistle", "anchor", "basket",
         "candle", "fountain", "ladder", "mirror", "pillow", "ribbon", "spoon", "thimble", "umbrella",
         "window", "feather", "compass", "bicycle", "cabinet"]


def make_prompts(n: int, seed: int = 0) -> list[dict]:
    """The first n of a fixed pool of 100 prompts (random keys, 7-digit values)."""
    pool = 100
    assert n <= pool
    rng = random.Random(seed)
    keys = rng.sample([f"{a}-{b}" for a in ADJECTIVES for b in NOUNS], pool)
    return [{"key": k, "value": str(rng.randint(10**6, 10**7 - 1))} for k in keys][:n]


class PromptBuilder:
    def __init__(self, tokenizer, seed: int = 0):
        self.tok = tokenizer
        self.seed = seed
        self.instruction = tokenizer.encode(INSTRUCTION).ids
        self._streams = {}
        self._line_ids = {}

    def _line(self, perm: tuple) -> list[int]:
        if perm not in self._line_ids:
            ids = self.tok.encode(" ".join(NOISE_SENTENCES[i] for i in perm) + "\n").ids
            assert len(ids) == LINE_TOKENS
            self._line_ids[perm] = ids
        return self._line_ids[perm]

    def _stream(self, h: int, part: str) -> list[tuple]:
        """MAX_LINES random lines (sentence permutations) of haystack sample h; part 'before' or 'after'."""
        if (h, part) not in self._streams:
            rng = random.Random(f"{self.seed}-{h}-{part}")
            self._streams[(h, part)] = [tuple(rng.sample(range(5), 5)) for _ in range(MAX_LINES)]
        return self._streams[(h, part)]

    def before(self, h: int, L: int) -> list[int]:
        """The L haystack tokens before the needle: the last L / 25 lines of the 'before' stream."""
        assert L % LINE_TOKENS == 0 and L // LINE_TOKENS <= MAX_LINES
        n = L // LINE_TOKENS
        lines = self._stream(h, "before")[MAX_LINES - n:] if n else []
        return [t for perm in lines for t in self._line(perm)]

    def after(self, h: int, d: int) -> list[int]:
        """The d haystack tokens between the needle and the question: the first d / 25 'after' lines."""
        assert d % LINE_TOKENS == 0 and d // LINE_TOKENS <= MAX_LINES
        return [t for perm in self._stream(h, "after")[:d // LINE_TOKENS] for t in self._line(perm)]

    def prefix(self, layout: str, h: int, L: int) -> list[int]:
        """The prompt-independent prefix."""
        return self.instruction + self.before(h, L) if layout == "A" else self.before(h, L)

    def suffix(self, layout: str, prompt: dict, h: int, d: int) -> list[int]:
        needle = self.tok.encode(NEEDLE.format(**prompt)).ids
        question = self.tok.encode(QUESTION.format(key=prompt["key"])).ids
        head = self.instruction + needle if layout == "B" else needle
        return head + self.after(h, d) + question

    def full(self, layout: str, prompt: dict, h: int, L: int, d: int) -> list[int]:
        return self.prefix(layout, h, L) + self.suffix(layout, prompt, h, d)
