from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MultiRepresentationData:
    reps: List[Dict]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _norm(s: str) -> str:
    return (s or "").strip()


def _abbr_variants(term: str) -> List[str]:
    """Generate simple abbreviation variants.

    Examples:
      - "Standard & Poor's" -> ["Standard & Poor's", "Standard Poors", "S P"] (rough)
      - "S&P" -> ["S&P", "SP"]
    """

    t = _norm(term)
    if not t:
        return []

    variants = {t}

    # Remove punctuation
    variants.add(re.sub(r"[^A-Za-z0-9\s]", "", t).strip())

    # Letter tokens abbreviation
    tokens = _TOKEN_RE.findall(t)
    if tokens:
        initials = " ".join(tok[0].upper() for tok in tokens if tok)
        if initials:
            variants.add(initials)
        initials2 = "".join(tok[0].upper() for tok in tokens if tok)
        if initials2:
            variants.add(initials2)

    return [v for v in variants if v]


def build_multi_representation_data(terms: List[Dict]) -> MultiRepresentationData:
    """Build multiple representations for embedding.

    Implements "03-构建多表示的索引":
      - For each term, create multiple textual representations.
      - All reps point back to the same canonical term_id.

    For fin-term-std data, common helpful reps:
      - raw term
      - normalized term (punctuation removed)
      - abbreviation/initials
      - term + label combo (if label exists)
    """

    reps: List[Dict] = []
    for term_id, item in enumerate(terms):
        term = _norm(item.get("term", ""))
        if not term:
            continue
        label = _norm(item.get("label", ""))

        base_reps = []
        base_reps.extend(_abbr_variants(term))
        if label:
            base_reps.append(f"{term} ({label})")

        # Deduplicate
        seen = set()
        for r in base_reps:
            rr = _norm(r)
            if not rr or rr in seen:
                continue
            seen.add(rr)
            reps.append(
                {
                    "term_id": term_id,
                    "canonical_term": term,
                    "label": label or None,
                    "representation": rr,
                    "rep_type": "derived" if rr != term else "raw",
                }
            )

    return MultiRepresentationData(reps=reps)
