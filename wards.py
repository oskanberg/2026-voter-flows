"""Ward-level data loading shared across the modelling and plotting scripts.

Loads the Democracy Club candidate-level CSVs for May 2024 and May 2026,
aggregates them to (year, ward) -> party totals, filters to a comparable
pool, and produces the share-of-electorate matrices X (2024) and Y (2026)
that the models fit against.

Public surface:
    CATS, SHORT, K   — the 7 categories used throughout
    ward_vector(d)   — turn one ward's aggregated dict into a 7-vector
    load_pool()      — load + filter + return (pool, X, Y)

See README.md for the substantive notes on the filter choices and the
multi-member-ward correction.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
FILES = {
    "2024": DATA_DIR / "candidates_2024.csv",
    "2026": DATA_DIR / "candidates_2026.csv",
}

# Labour and Labour/Co-op are politically the same group; merge for modelling.
PARTY_MAP = {"Labour and Co-operative Party": "Labour Party"}

CATS = [
    "Labour Party",
    "Conservative and Unionist Party",
    "Liberal Democrats",
    "Green Party",
    "Reform UK",
    "Other",
    "Abstain",
]
SHORT = ["Lab", "Con", "LD", "Grn", "Ref", "Oth", "DNV"]
K = len(CATS)


def ward_vector(d):
    """Return the ward's 7-vector of share-of-electorate values, summing to 1.

    Multi-member wards: a voter casts up to `seats` ballot marks, so raw
    vote totals over-count parties that field full slates. We convert to
    voter-equivalents by dividing each party's votes by the number of
    candidates that party fielded in the ward, under the assumption that
    supporters cast their full slate for one party.

    Layout:
        out[non-Abstain] = (voter_equiv[party] / sum(voter_equiv)) * turnout
        out[Abstain]     = 1 - turnout
    """
    turnout = float(d["turnout_pct"]) / 100
    voter_equiv = np.zeros(K)
    for party, v in d["votes"].items():
        cand = d["candidates"][party]
        idx = CATS.index(party) if party in CATS else CATS.index("Other")
        voter_equiv[idx] += v / cand
    total_ve = voter_equiv.sum()
    out = (voter_equiv / total_ve) * turnout
    out[CATS.index("Abstain")] = 1 - turnout
    return out


def _aggregate():
    """Load candidate-level rows into a {(year, ward): aggregated-dict} map.

    Ward identity = ballot_paper_id with the date stripped and ".by." removed,
    so a regular ward and its by-election variant share the same key.
    """
    contests = defaultdict(
        lambda: {
            "votes": defaultdict(int),
            "candidates": defaultdict(int),
            "is_by": False,
            "turnout_pct": "",
        }
    )
    for year, path in FILES.items():
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["cancelled_poll"] == "t":
                    continue
                v = row["votes_cast"].strip()
                if not v:
                    continue
                bp = row["ballot_paper_id"]
                ward = re.sub(r"\.\d{4}-\d{2}-\d{2}$", "", bp).replace(".by", "")
                party = PARTY_MAP.get(row["party_name"], row["party_name"])
                d = contests[(year, ward)]
                d["votes"][party] += int(v)
                d["candidates"][party] += 1
                if ".by." in bp:
                    d["is_by"] = True
                tp = row["turnout_percentage"].strip()
                if tp and not d["turnout_pct"]:
                    d["turnout_pct"] = tp
    return contests


def load_pool():
    """Load and filter the canonical ward pool.

    A ward is included iff:
      - it appears in both 2024 and 2026,
      - neither year is a by-election (idiosyncratic dynamics),
      - turnout is reported in both years (so DNV is well-defined).

    We do not filter on "same number of seats up in both years": by-thirds
    councils caught in different parts of their 4-year cycle carry useful
    information once ward_vector()'s multi-member correction is applied.

    Returns:
        pool: sorted list of ward identifiers (length ~754)
        X:    (N, 7) array of 2024 share-of-electorate vectors
        Y:    (N, 7) array of 2026 share-of-electorate vectors
    """
    contests = _aggregate()
    overlap = {w for (y, w) in contests if y == "2024"} & {
        w for (y, w) in contests if y == "2026"
    }
    pool = []
    for ward in sorted(overlap):
        a, b = contests[("2024", ward)], contests[("2026", ward)]
        if a["is_by"] or b["is_by"]:
            continue
        if not a["turnout_pct"] or not b["turnout_pct"]:
            continue
        pool.append(ward)
    X = np.array([ward_vector(contests[("2024", w)]) for w in pool])
    Y = np.array([ward_vector(contests[("2026", w)]) for w in pool])
    return pool, X, Y
