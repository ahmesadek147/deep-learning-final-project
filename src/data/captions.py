"""Caption splitting, analysis and cleaning (Task 1.1).

The ``description`` column of CUB holds ten captions in a single string,
separated by newlines. This module:

1. splits that string into individual captions,
2. computes statistics (length, vocabulary, duplicates, rare words / typos),
3. cleans the captions (normalisation + conservative typo correction),
4. saves the cleaned captions so the captioning task can reuse them.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

_NON_WORD = re.compile(r"[^a-z0-9']+")  # everything except letters, digits, '
_SPACES = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Splitting and basic cleaning
# --------------------------------------------------------------------------
def split_captions(description) -> list[str]:
    """Split one ``description`` value into its individual captions.

    Empty lines and surrounding whitespace are removed. A list/tuple input
    is accepted too, in case the dataset version already stores a list.
    """
    if isinstance(description, (list, tuple)):
        raw = list(description)
    else:
        raw = str(description or "").split("\n")
    return [c.strip() for c in raw if c and c.strip()]


def clean_caption(text: str) -> str:
    """Normalise a single caption.

    Steps: Unicode NFKC -> lowercase -> replace punctuation / hyphens with
    spaces (apostrophes kept) -> collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)
    text = _SPACES.sub(" ", text).strip(" '")
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def analyze_captions(descriptions: Iterable) -> dict:
    """Compute dataset-level caption statistics for the EDA section.

    Returns a dict with raw statistics (before cleaning) and the vocabulary
    counter after cleaning, ready for plotting.
    """
    descriptions = list(descriptions)
    captions_per_image = Counter()
    raw_lengths, clean_lengths = [], []
    raw_vocab, clean_vocab = Counter(), Counter()
    empty_captions = non_ascii = dup_within_image = 0
    n_missing_period = n_uppercase = 0

    for desc in descriptions:
        caps = split_captions(desc)
        captions_per_image[len(caps)] += 1
        cleaned_seen = set()
        for cap in caps:
            raw_lengths.append(len(cap.split()))
            raw_vocab.update(cap.split())
            if any(ord(ch) > 127 for ch in cap):
                non_ascii += 1
            if not cap.endswith("."):
                n_missing_period += 1
            if any(ch.isupper() for ch in cap):
                n_uppercase += 1

            cleaned = clean_caption(cap)
            if not cleaned:
                empty_captions += 1
                continue
            if cleaned in cleaned_seen:
                dup_within_image += 1
            cleaned_seen.add(cleaned)
            toks = tokenize(cleaned)
            clean_lengths.append(len(toks))
            clean_vocab.update(toks)

    return {
        "n_images": len(descriptions),
        "captions_per_image": dict(captions_per_image),
        "n_captions": len(raw_lengths),
        "raw_lengths": raw_lengths,
        "clean_lengths": clean_lengths,
        "raw_vocab_size": len(raw_vocab),
        "clean_vocab_size": len(clean_vocab),
        "hapax_words": sum(1 for c in clean_vocab.values() if c == 1),
        "empty_captions": empty_captions,
        "non_ascii_captions": non_ascii,
        "duplicate_captions_within_image": dup_within_image,
        "captions_without_final_period": n_missing_period,
        "captions_with_uppercase": n_uppercase,
        "clean_vocab": clean_vocab,
    }


def summarize(stats: dict) -> str:
    """Human-readable summary (paste into the report)."""
    import statistics as st

    lens = stats["clean_lengths"]
    lines = [
        f"Images:                         {stats['n_images']}",
        f"Captions:                       {stats['n_captions']}",
        f"Captions per image:             {stats['captions_per_image']}",
        f"Words per caption (mean/median):{st.mean(lens):.1f} / {st.median(lens):.0f}",
        f"Words per caption (min/max):    {min(lens)} / {max(lens)}",
        f"Vocabulary raw -> cleaned:      {stats['raw_vocab_size']} -> {stats['clean_vocab_size']}",
        f"Words appearing only once:      {stats['hapax_words']}",
        f"Empty captions after cleaning:  {stats['empty_captions']}",
        f"Duplicate captions in an image: {stats['duplicate_captions_within_image']}",
        f"Captions with non-ASCII chars:  {stats['non_ascii_captions']}",
        f"Captions without final period:  {stats['captions_without_final_period']}",
        f"Captions containing uppercase:  {stats['captions_with_uppercase']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Typo detection
# --------------------------------------------------------------------------
def _is_inflection(a: str, b: str) -> bool:
    """True if one word is a prefix of the other (plural, -ed, -ing ...).

    Such pairs are legitimate words, not typos, so they are never corrected.
    """
    return a.startswith(b) or b.startswith(a)


def _edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Levenshtein distance with early exit once it exceeds ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def find_typo_candidates(
    vocab: Counter,
    min_common: int = 20,
    max_rare: int = 2,
) -> dict[str, str]:
    """Map rare words to a very similar frequent word (likely typos).

    A word is a typo candidate when it appears at most ``max_rare`` times and
    is within edit distance 1 (words up to 7 letters) or 2 (longer words) of a
    word appearing at least ``min_common`` times, with the same first letter.
    Prefix pairs (plural, -ed, -ing) are skipped. The rule is deliberately
    conservative: it prefers leaving a typo in over breaking a valid word.
    """
    common_by_letter: dict[str, list[str]] = {}
    for w, c in vocab.items():
        if c >= min_common and len(w) >= 4 and w.isalpha():
            common_by_letter.setdefault(w[0], []).append(w)

    corrections: dict[str, str] = {}
    for word, count in vocab.items():
        if count > max_rare or len(word) < 4 or not word.isalpha():
            continue
        limit = 1 if len(word) <= 7 else 2
        best, best_d = None, limit + 1
        for cand in common_by_letter.get(word[0], []):
            if _is_inflection(word, cand):
                continue
            d = _edit_distance(word, cand, limit)
            if d < best_d or (d == best_d and best and vocab[cand] > vocab[best]):
                best, best_d = cand, d
        if best is not None:
            corrections[word] = best
    return corrections


# --------------------------------------------------------------------------
# Full cleaning pipeline
# --------------------------------------------------------------------------
class CaptionCleaner:
    """Fit on the training descriptions, then clean any description.

    Fit on the TRAIN split only and reuse the same correction map for
    validation / test, so no information leaks between splits.
    """

    def __init__(
        self,
        fix_typos: bool = True,
        min_words: int = 3,
        max_words: int = 60,
        drop_duplicates: bool = True,
    ):
        self.fix_typos = fix_typos
        self.min_words = min_words
        self.max_words = max_words
        self.drop_duplicates = drop_duplicates
        self.corrections: dict[str, str] = {}

    def fit(self, descriptions: Iterable) -> "CaptionCleaner":
        vocab = Counter()
        for desc in descriptions:
            for cap in split_captions(desc):
                vocab.update(tokenize(clean_caption(cap)))
        if self.fix_typos:
            self.corrections = find_typo_candidates(vocab)
        return self

    def clean_description(self, description) -> list[str]:
        """Return the list of cleaned captions for one image."""
        out, seen = [], set()
        for cap in split_captions(description):
            toks = [self.corrections.get(t, t) for t in tokenize(clean_caption(cap))]
            if not (self.min_words <= len(toks) <= self.max_words):
                continue
            text = " ".join(toks)
            if self.drop_duplicates and text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def save_corrections(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.corrections, indent=2, sort_keys=True))


def build_clean_table(hf_dataset, cleaner: CaptionCleaner) -> list[dict]:
    """Clean every row. Reads only text columns, so no images are decoded."""
    descs = hf_dataset["description"]
    names = hf_dataset["file_name"] if "file_name" in hf_dataset.column_names else [None] * len(descs)
    labels = hf_dataset["label"]
    return [
        {"index": i, "file_name": n, "label": l, "captions": cleaner.clean_description(d)}
        for i, (d, n, l) in enumerate(zip(descs, names, labels))
    ]


def save_clean_table(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False))
