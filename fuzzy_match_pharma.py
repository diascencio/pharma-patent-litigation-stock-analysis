"""
Fast + safer fuzzy matching for legal-case entity names against pharma ticker names.

This version incorporates the performance fix for very large `general_df` inputs:
by default it fuzzy-matches ONLY unique entity names, then merges the results back
onto the full dataframe. That keeps the safer ACCEPT / REVIEW / REJECT logic while
avoiding repeated fuzzy searches for duplicate entity names.

Typical use, drop-in compatible with the older call:

    from fuzzy_match_pharma import fuzzy_match_pharma

    pharma_general_df = fuzzy_match_pharma(
        general_df,
        pharma_tickers_df,
        min_score=95,
    )

Recommended explicit use:

    from fuzzy_match_pharma import fuzzy_match_pharma_safe

    pharma_general_df = fuzzy_match_pharma_safe(
        general_df,
        pharma_tickers_df,
        entity_col="entity_name",
        company_col="Company_Name",
        min_auto_score=95,
        match_unique_entities=True,   # default; important for large data
        candidate_limit=4,            # faster than the old safer default of 8
    )

Dependency:
    pip install rapidfuzz
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz, process


# -----------------------------------------------------------------------------
# Name parsing rules
# -----------------------------------------------------------------------------

# Pure legal/entity suffixes: safe to strip before comparison.
LEGAL_SUFFIXES: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "lp", "llp", "pllc", "plc", "pc",
    "ag", "sa", "nv", "se", "gmbh", "spa", "kg", "srl", "bv",
    "pte", "pty", "kk", "ab", "oyj", "oy", "asa",
    # Common fragments created by punctuation in names like A/S and N.V.
    "a", "s", "n", "v",
}

CONNECTOR_WORDS: Set[str] = {
    "the", "and", "of", "for", "in", "on", "by", "with", "to", "from",
}

# Descriptor words are NOT stripped entirely. They are retained as weak signals.
# This prevents false positives such as United Technologies -> United Therapeutics.
DESCRIPTOR_MAP: Dict[str, str] = {
    "pharmaceuticals": "pharma",
    "pharmaceutical": "pharma",
    "pharma": "pharma",
    "therapeutics": "therapeutics",
    "therapeutic": "therapeutics",
    "biopharma": "biopharma",
    "biopharmaceuticals": "biopharma",
    "biopharmaceutical": "biopharma",
    "biosciences": "bioscience",
    "bioscience": "bioscience",
    "biotech": "biotech",
    "biotechnology": "biotech",
    "technologies": "technology",
    "technology": "technology",
    "tech": "technology",
    "laboratories": "lab",
    "laboratory": "lab",
    "labs": "lab",
    "lab": "lab",
    "sciences": "science",
    "science": "science",
    "research": "research",
    "industries": "industry",
    "industry": "industry",
    "healthcare": "health",
    "health": "health",
    "group": "group",
    "holdings": "holding",
    "holding": "holding",
}

PHARMAISH_DESCRIPTORS: Set[str] = {
    "pharma", "therapeutics", "biopharma", "bioscience", "biotech",
}
NONPHARMA_DESCRIPTORS: Set[str] = {"technology", "industry", "research", "health"}
LIFE_SCIENCE_WEAK_DESCRIPTORS: Set[str] = {"lab", "science"}
STRUCTURE_DESCRIPTORS: Set[str] = {"group", "holding"}

# Known historical renames or aliases that plain fuzzy matching cannot infer.
# Keep this list short and evidence-based. Add more after manual review.
DEFAULT_TOKEN_ALIASES: Dict[str, str] = {
    "isis": "ionis",  # Isis Pharmaceuticals renamed Ionis Pharmaceuticals.
}


@dataclass(frozen=True)
class NameParts:
    original: str
    tokens: Tuple[str, ...]          # Non-legal tokens, including descriptors.
    core_tokens: Tuple[str, ...]     # Brand/distinctive tokens only.
    descriptors: Tuple[str, ...]     # Canonical descriptor categories.

    @property
    def full(self) -> str:
        return " ".join(self.tokens)

    @property
    def core(self) -> str:
        return " ".join(self.core_tokens)

    @property
    def core_set(self) -> Set[str]:
        return set(self.core_tokens)

    @property
    def descriptor_set(self) -> Set[str]:
        return set(self.descriptors)


def normalize_name_parts(
    name: object,
    token_aliases: Optional[Dict[str, str]] = None,
) -> NameParts:
    """Parse a company/entity name into full, core, and descriptor tokens."""
    token_aliases = token_aliases or DEFAULT_TOKEN_ALIASES

    if pd.isna(name):
        return NameParts(original="", tokens=(), core_tokens=(), descriptors=())

    original = str(name)
    s = original.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"['’]", "", s)          # reddy's -> reddys
    s = s.replace(".", "")              # L.P. -> lp before punctuation split
    raw_tokens = re.sub(r"[^a-z0-9\s]", " ", s).split()

    tokens: List[str] = []
    descriptors: List[str] = []
    core_tokens: List[str] = []

    for token in raw_tokens:
        if token in CONNECTOR_WORDS or token in LEGAL_SUFFIXES:
            continue
        token = token_aliases.get(token, token)
        if not token:
            continue
        tokens.append(token)
        if token in DESCRIPTOR_MAP:
            descriptors.append(DESCRIPTOR_MAP[token])
        elif len(token) > 1 or token.isdigit():
            core_tokens.append(token)

    return NameParts(
        original=original,
        tokens=tuple(tokens),
        core_tokens=tuple(core_tokens),
        descriptors=tuple(descriptors),
    )


# -----------------------------------------------------------------------------
# Pair classification
# -----------------------------------------------------------------------------

def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(fuzz.token_sort_ratio(a, b))


def _core_overlap(q: NameParts, c: NameParts) -> float:
    if not q.core_set or not c.core_set:
        return 0.0
    return len(q.core_set & c.core_set) / min(len(q.core_set), len(c.core_set))


def _one_exact_core(q: NameParts, c: NameParts) -> bool:
    return len(q.core_set) == 1 and q.core_set == c.core_set


def _is_short_core(q: NameParts, c: NameParts) -> bool:
    if not _one_exact_core(q, c):
        return False
    token = next(iter(q.core_set))
    return len(token) <= 4


def classify_name_parts(
    q: NameParts,
    c: NameParts,
    *,
    min_auto_score: float = 92.0,
    min_review_score: float = 72.0,
) -> Dict[str, object]:
    """Classify already-normalized name parts as ACCEPT, REVIEW, or REJECT."""
    full_score = _ratio(q.full, c.full)
    core_score = _ratio(q.core, c.core)
    core_overlap = _core_overlap(q, c)
    shared_core = sorted(q.core_set & c.core_set)
    q_desc = q.descriptor_set
    c_desc = c.descriptor_set
    desc_overlap = q_desc & c_desc

    # Composite score favors the actual brand/core word but still penalizes
    # descriptor mismatch via the full-name component.
    score = round((0.70 * core_score) + (0.30 * full_score), 2)

    def result(
        decision: str,
        reason: str,
        adjusted_score: Optional[float] = None,
    ) -> Dict[str, object]:
        return {
            "decision": decision,
            "reason": reason,
            "score": round(
                float(score if adjusted_score is None else adjusted_score),
                2,
            ),
            "full_score": round(full_score, 2),
            "core_score": round(core_score, 2),
            "core_overlap": round(core_overlap, 3),
            "entity_clean": q.full,
            "company_clean": c.full,
            "entity_core": q.core,
            "company_core": c.core,
            "entity_descriptors": ",".join(sorted(q_desc)),
            "company_descriptors": ",".join(sorted(c_desc)),
            "shared_core_tokens": ",".join(shared_core),
        }

    if not q.tokens or not c.tokens:
        return result("REJECT", "Missing usable name tokens", 0.0)

    if q.full == c.full:
        return result("ACCEPT", "Normalized full names match", 100.0)

    # Do not allow descriptor overlap alone to create a match.
    if not shared_core and core_score < 90:
        return result("REJECT", "No shared distinctive/core token")

    # High-quality full-name match, including punctuation/very minor typo variants.
    if full_score >= 96 and (core_overlap >= 0.80 or core_score >= 94):
        return result(
            "ACCEPT",
            "High full-name similarity with strong core overlap",
            full_score,
        )

    # Typo-tolerant multi-token exactish match.
    if (
        len(q.core_set) >= 2
        and len(c.core_set) >= 2
        and full_score >= 94.5
        and core_score >= 94.5
    ):
        return result(
            "ACCEPT",
            "High multi-token similarity; likely typo/punctuation variant",
            min(full_score, core_score),
        )

    single_core_exact = _one_exact_core(q, c)

    if single_core_exact:
        # Examples: United Technologies -> United Therapeutics,
        # Rocket Science Group -> Rocket Pharmaceuticals.
        if (
            (q_desc & NONPHARMA_DESCRIPTORS)
            and (c_desc & PHARMAISH_DESCRIPTORS)
            and not desc_overlap
        ):
            return result(
                "REJECT",
                "Exact core token but conflicting non-pharma vs pharma descriptor",
            )

        if (
            (q_desc & STRUCTURE_DESCRIPTORS)
            and (c_desc & PHARMAISH_DESCRIPTORS)
            and not desc_overlap
        ):
            return result(
                "REVIEW",
                "Exact core token but entity is holding/group context",
            )

        # A bare acronym mapped to a pharma company is risky, for example:
        # PTC Inc -> PTC Therapeutics.
        if not q_desc and (c_desc & PHARMAISH_DESCRIPTORS):
            return result(
                "REVIEW",
                "Exact core token but entity lacks pharma descriptor",
            )

        if _is_short_core(q, c) and q_desc != c_desc and not desc_overlap:
            return result(
                "REVIEW",
                "Short exact core with weak or different descriptors",
            )

        if (
            (q_desc & LIFE_SCIENCE_WEAK_DESCRIPTORS)
            and (c_desc & PHARMAISH_DESCRIPTORS)
            and not desc_overlap
        ):
            return result(
                "REVIEW",
                "Exact core token but lab/science descriptor differs from "
                "pharma descriptor",
            )

        if (q_desc & NONPHARMA_DESCRIPTORS) and not desc_overlap:
            return result(
                "REVIEW",
                "Exact core token but entity has a different non-pharma descriptor",
            )

        if q_desc and c_desc and not desc_overlap:
            if (q_desc & PHARMAISH_DESCRIPTORS) and (c_desc & PHARMAISH_DESCRIPTORS):
                return result(
                    "REVIEW",
                    "Exact core token with different life-science descriptors",
                )
            if (q_desc & NONPHARMA_DESCRIPTORS) or (c_desc & NONPHARMA_DESCRIPTORS):
                return result(
                    "REVIEW",
                    "Exact core token with different non-legal descriptors",
                )

        if score >= min_auto_score or core_score == 100:
            return result(
                "ACCEPT",
                "Exact distinctive/core token with compatible descriptors",
            )

    # Multi-token brands are more reliable.
    if core_score >= min_auto_score and core_overlap >= 0.80:
        return result("ACCEPT", "Strong multi-token core match")

    if (
        max(score, core_score, full_score) >= min_review_score
        and (shared_core or core_score >= 82)
    ):
        return result("REVIEW", "Borderline similarity; manual review recommended")

    return result("REJECT", "Similarity below review threshold")


def classify_name_pair(
    entity_name: object,
    company_name: object,
    *,
    token_aliases: Optional[Dict[str, str]] = None,
    min_auto_score: float = 92.0,
    min_review_score: float = 72.0,
) -> Dict[str, object]:
    """Classify a single raw entity/company-name pair."""
    q = normalize_name_parts(entity_name, token_aliases)
    c = normalize_name_parts(company_name, token_aliases)
    return classify_name_parts(
        q,
        c,
        min_auto_score=min_auto_score,
        min_review_score=min_review_score,
    )


# -----------------------------------------------------------------------------
# Matching engine
# -----------------------------------------------------------------------------

BASE_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "match_decision",
    "match_reason",
    # Score for an ACCEPTED match only. Keeping this blank for REVIEW/REJECT
    # makes legacy filters such as df["fuzz_match_score"] >= min_score safe.
    "fuzz_match_score",
    # Best-candidate similarity for diagnostics, regardless of decision.
    "candidate_fuzz_match_score",
    "full_name_score",
    "core_name_score",
    "core_overlap",
    "entity_clean",
    "entity_core",
    "entity_descriptors",
    "matched_company_name",
    "Ticker",
    "CIK_10",
)

REVIEW_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "review_candidate_company_name",
    "review_candidate_ticker",
    "review_candidate_cik_10",
    "candidate_company_clean",
    "candidate_company_core",
    "candidate_company_descriptors",
    "shared_core_tokens",
)


def _output_columns(keep_review_candidates: bool) -> List[str]:
    cols = list(BASE_OUTPUT_COLUMNS)
    if keep_review_candidates:
        cols.extend(REVIEW_OUTPUT_COLUMNS)
    return cols


def _candidate_indices(
    query_parts: NameParts,
    core_choices: Dict[int, str],
    full_choices: Dict[int, str],
    limit: int,
) -> List[int]:
    indices: Set[int] = set()

    if query_parts.core:
        for _choice, _score, idx in process.extract(
            query_parts.core,
            core_choices,
            scorer=fuzz.token_sort_ratio,
            limit=limit,
        ):
            indices.add(idx)

    if query_parts.full:
        for _choice, _score, idx in process.extract(
            query_parts.full,
            full_choices,
            scorer=fuzz.token_sort_ratio,
            limit=limit,
        ):
            indices.add(idx)

    return list(indices)


def _prepare_pharma(
    pharma_tickers_df: pd.DataFrame,
    company_col: str,
    token_aliases: Optional[Dict[str, str]],
) -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, str]]:
    required_pharma = {company_col, "Ticker"}
    missing_pharma = required_pharma - set(pharma_tickers_df.columns)
    if missing_pharma:
        raise ValueError(
            "pharma_tickers_df is missing required column(s): "
            f"{sorted(missing_pharma)}"
        )

    pharma = pharma_tickers_df.reset_index(drop=True).copy()
    if "CIK_10" not in pharma.columns:
        pharma["CIK_10"] = pd.NA

    pharma["_parts"] = pharma[company_col].apply(
        lambda value: normalize_name_parts(value, token_aliases)
    )
    pharma["_core_norm"] = pharma["_parts"].apply(lambda p: p.core)
    pharma["_full_norm"] = pharma["_parts"].apply(lambda p: p.full)

    return pharma, pharma["_core_norm"].to_dict(), pharma["_full_norm"].to_dict()


def _match_entity_table(
    entity_df: pd.DataFrame,
    pharma: pd.DataFrame,
    core_choices: Dict[int, str],
    full_choices: Dict[int, str],
    *,
    entity_col: str,
    company_col: str,
    min_auto_score: float,
    min_review_score: float,
    candidate_limit: int,
    token_aliases: Optional[Dict[str, str]],
    keep_review_candidates: bool,
    verbose: bool = False,
    progress_every: int = 10_000,
) -> pd.DataFrame:
    """Match the rows in `entity_df`. Usually this should contain unique names."""
    entity_table = entity_df.copy()
    entity_table["_query_parts"] = entity_table[entity_col].apply(
        lambda x: normalize_name_parts(x, token_aliases)
    )

    output_rows: List[Dict[str, object]] = []
    total = len(entity_table)

    for n, (_row_idx, row) in enumerate(entity_table.iterrows(), start=1):
        if verbose and progress_every and n % progress_every == 0:
            print(f"Matched {n:,}/{total:,} unique entity names...")

        query_parts: NameParts = row["_query_parts"]
        candidate_ids = _candidate_indices(
            query_parts,
            core_choices,
            full_choices,
            candidate_limit,
        )

        candidate_scores: List[Dict[str, object]] = []
        for cand_idx in candidate_ids:
            company_name = pharma.at[cand_idx, company_col]
            company_parts: NameParts = pharma.at[cand_idx, "_parts"]
            scored = classify_name_parts(
                query_parts,
                company_parts,
                min_auto_score=min_auto_score,
                min_review_score=min_review_score,
            )
            scored["candidate_idx"] = cand_idx
            scored["candidate_company_name"] = company_name
            scored["candidate_ticker"] = pharma.at[cand_idx, "Ticker"]
            scored["candidate_cik_10"] = pharma.at[cand_idx, "CIK_10"]
            candidate_scores.append(scored)

        if not candidate_scores:
            best = {
                "decision": "NO_MATCH",
                "reason": "No candidates found",
                "score": 0.0,
                "full_score": 0.0,
                "core_score": 0.0,
                "core_overlap": 0.0,
                "candidate_idx": None,
                "candidate_company_name": pd.NA,
                "candidate_ticker": pd.NA,
                "candidate_cik_10": pd.NA,
                "entity_clean": query_parts.full,
                "company_clean": "",
                "entity_core": query_parts.core,
                "company_core": "",
                "entity_descriptors": ",".join(sorted(query_parts.descriptor_set)),
                "company_descriptors": "",
                "shared_core_tokens": "",
            }
        else:
            rank = {"ACCEPT": 2, "REVIEW": 1, "REJECT": 0}
            candidate_scores.sort(
                key=lambda d: (
                    rank.get(str(d["decision"]), -1),
                    float(d["score"]),
                    float(d["core_score"]),
                ),
                reverse=True,
            )
            best = candidate_scores[0]

            # If the best ACCEPT is only narrowly better than another plausible
            # candidate, downgrade to REVIEW. This protects crowded names.
            if best["decision"] == "ACCEPT":
                plausible = [
                    candidate
                    for candidate in candidate_scores[1:]
                    if candidate["decision"] in {"ACCEPT", "REVIEW"}
                ]
                if plausible:
                    runner_up = plausible[0]
                    different_core = runner_up.get("company_core") != best.get(
                        "company_core"
                    )
                    margin = float(best["score"]) - float(runner_up["score"])
                    if different_core and margin < 3.0 and float(best["score"]) < 100.0:
                        best = dict(best)
                        best["decision"] = "REVIEW"
                        best["reason"] = (
                            "Best candidate has narrow margin over another "
                            "plausible candidate"
                        )

        accepted = best["decision"] == "ACCEPT"
        out = {
            "match_decision": best["decision"],
            "match_reason": best["reason"],
            # IMPORTANT: this is the score of a populated/accepted match.
            # REVIEW/REJECT rows deliberately use pd.NA so a legacy numeric
            # threshold cannot select a row whose matched fields are blank.
            "fuzz_match_score": best["score"] if accepted else pd.NA,
            # Preserve the raw best-candidate score for QA/manual review.
            "candidate_fuzz_match_score": best["score"],
            "full_name_score": best["full_score"],
            "core_name_score": best["core_score"],
            "core_overlap": best["core_overlap"],
            "entity_clean": best["entity_clean"],
            "entity_core": best["entity_core"],
            "entity_descriptors": best["entity_descriptors"],
            "matched_company_name": (
                best["candidate_company_name"] if accepted else pd.NA
            ),
            "Ticker": best["candidate_ticker"] if accepted else pd.NA,
            "CIK_10": best["candidate_cik_10"] if accepted else pd.NA,
        }

        if keep_review_candidates:
            out.update({
                "review_candidate_company_name": (
                    best["candidate_company_name"]
                    if best["decision"] == "REVIEW"
                    else pd.NA
                ),
                "review_candidate_ticker": (
                    best["candidate_ticker"] if best["decision"] == "REVIEW" else pd.NA
                ),
                "review_candidate_cik_10": (
                    best["candidate_cik_10"] if best["decision"] == "REVIEW" else pd.NA
                ),
                "candidate_company_clean": best["company_clean"],
                "candidate_company_core": best["company_core"],
                "candidate_company_descriptors": best["company_descriptors"],
                "shared_core_tokens": best["shared_core_tokens"],
            })

        output_rows.append(out)

    diagnostics = pd.DataFrame(output_rows, index=entity_table.index)
    return pd.concat([entity_table.drop(columns=["_query_parts"]), diagnostics], axis=1)


def fuzzy_match_pharma_safe(
    general_df: pd.DataFrame,
    pharma_tickers_df: pd.DataFrame,
    entity_col: str = "entity_name",
    company_col: str = "Company_Name",
    *,
    min_auto_score: float = 92.0,
    min_review_score: float = 72.0,
    candidate_limit: int = 4,
    token_aliases: Optional[Dict[str, str]] = None,
    keep_review_candidates: bool = True,
    match_unique_entities: bool = True,
    drop_existing_match_cols: bool = True,
    verbose: bool = False,
    progress_every: int = 10_000,
) -> pd.DataFrame:
    """Fuzzy-match entity names to pharma ticker companies with safer gating.

    Parameters
    ----------
    match_unique_entities:
        If True, fuzzy-match only unique values of `general_df[entity_col]`, then
        merge the result back to all rows. Keep this True for large data.

    candidate_limit:
        Number of top candidates to inspect from the core-name search and the
        full-name search. A value of 4 is usually much faster than 8 while still
        checking several plausible candidates.

    keep_review_candidates:
        If True, REVIEW rows include review_candidate_* columns. ACCEPT rows
        populate matched_company_name/Ticker/CIK_10. REVIEW and REJECT rows do
        not populate ticker fields. Their raw similarity remains available in
        candidate_fuzz_match_score, while fuzz_match_score is blank unless the
        match was accepted. This keeps legacy score-based filters consistent.

    drop_existing_match_cols:
        If True, existing output columns such as Ticker/matched_company_name are
        removed before the new results are merged back. This avoids _x/_y suffixes
        when rerunning the matcher on a previously matched dataframe.
    """
    if entity_col not in general_df.columns:
        raise ValueError(f"general_df is missing required column: {entity_col!r}")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be >= 1")

    pharma, core_choices, full_choices = _prepare_pharma(
        pharma_tickers_df,
        company_col,
        token_aliases,
    )

    output_cols = _output_columns(keep_review_candidates)

    if not match_unique_entities:
        base = general_df.copy()
        if drop_existing_match_cols:
            base = base.drop(
                columns=[column for column in output_cols if column in base.columns],
                errors="ignore",
            )
        return _match_entity_table(
            base,
            pharma,
            core_choices,
            full_choices,
            entity_col=entity_col,
            company_col=company_col,
            min_auto_score=min_auto_score,
            min_review_score=min_review_score,
            candidate_limit=candidate_limit,
            token_aliases=token_aliases,
            keep_review_candidates=keep_review_candidates,
            verbose=verbose,
            progress_every=progress_every,
        )

    # Fast path: match unique entity names once, then merge back.
    original_index = general_df.index
    base = general_df.copy()
    base["_fm_row_id"] = range(len(base))

    if drop_existing_match_cols:
        base = base.drop(
            columns=[column for column in output_cols if column in base.columns],
            errors="ignore",
        )

    unique_entities = base[[entity_col]].drop_duplicates().copy()

    if verbose:
        print(
            f"Matching {len(unique_entities):,} unique {entity_col!r} values "
            f"from {len(general_df):,} total rows."
        )

    unique_matches = _match_entity_table(
        unique_entities,
        pharma,
        core_choices,
        full_choices,
        entity_col=entity_col,
        company_col=company_col,
        min_auto_score=min_auto_score,
        min_review_score=min_review_score,
        candidate_limit=candidate_limit,
        token_aliases=token_aliases,
        keep_review_candidates=keep_review_candidates,
        verbose=verbose,
        progress_every=progress_every,
    )

    merge_cols = [entity_col] + [c for c in output_cols if c in unique_matches.columns]
    merged = base.merge(
        unique_matches[merge_cols],
        on=entity_col,
        how="left",
        sort=False,
        validate="m:1",
    )
    merged = merged.sort_values("_fm_row_id").drop(columns=["_fm_row_id"])
    merged.index = original_index
    return merged


# Backward-compatible function name: safer implementation, now using unique-first
# matching by default. This means the user's existing call can stay the same:
#     fuzzy_match_pharma(general_df, pharma_tickers_df, min_score=95)
def fuzzy_match_pharma(
    general_df: pd.DataFrame,
    pharma_tickers_df: pd.DataFrame,
    entity_col: str = "entity_name",
    company_col: str = "Company_Name",
    min_score: float = 92.0,
    *,
    candidate_limit: int = 4,
    keep_review_candidates: bool = True,
    match_unique_entities: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Backward-compatible wrapper around fuzzy_match_pharma_safe.

    `min_score` maps to `min_auto_score`. REVIEW candidates do not populate
    matched_company_name/Ticker/CIK_10. Their diagnostic similarity is stored in
    candidate_fuzz_match_score; fuzz_match_score is populated only for ACCEPT.
    """
    return fuzzy_match_pharma_safe(
        general_df,
        pharma_tickers_df,
        entity_col=entity_col,
        company_col=company_col,
        min_auto_score=min_score,
        candidate_limit=candidate_limit,
        keep_review_candidates=keep_review_candidates,
        match_unique_entities=match_unique_entities,
        verbose=verbose,
    )


def find_review_candidates(
    general_df: pd.DataFrame,
    pharma_tickers_df: pd.DataFrame,
    entity_col: str = "entity_name",
    company_col: str = "Company_Name",
    **kwargs,
) -> pd.DataFrame:
    """Return only rows whose best candidate requires manual review."""
    matched = fuzzy_match_pharma_safe(
        general_df,
        pharma_tickers_df,
        entity_col=entity_col,
        company_col=company_col,
        keep_review_candidates=True,
        **kwargs,
    )
    return matched.loc[matched["match_decision"] == "REVIEW"].copy()


def evaluate_existing_matches(
    matched_df: pd.DataFrame,
    entity_col: str = "entity_name",
    matched_company_col: str = "matched_company_name",
    *,
    token_aliases: Optional[Dict[str, str]] = None,
    min_auto_score: float = 92.0,
    min_review_score: float = 72.0,
) -> pd.DataFrame:
    """Audit an already-created matched dataframe without changing its columns."""
    required = {entity_col, matched_company_col}
    missing = required - set(matched_df.columns)
    if missing:
        raise ValueError(f"matched_df is missing required column(s): {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for _, row in matched_df.iterrows():
        rows.append(classify_name_pair(
            row[entity_col],
            row[matched_company_col],
            token_aliases=token_aliases,
            min_auto_score=min_auto_score,
            min_review_score=min_review_score,
        ))
    scored = pd.DataFrame(rows, index=matched_df.index).add_prefix("audit_")
    return pd.concat([matched_df.copy(), scored], axis=1)


def summarize_match_decisions(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Small helper for quick QA after running the matcher."""
    if "match_decision" not in matched_df.columns:
        raise ValueError("matched_df must contain a 'match_decision' column")
    return (
        matched_df["match_decision"]
        .value_counts(dropna=False)
        .rename_axis("match_decision")
        .reset_index(name="row_count")
    )


if __name__ == "__main__":
    # Small demo with duplicate rows to show that unique-first matching works.
    general_df = pd.DataFrame({
        "case_id": [1, 2, 3, 4, 5, 6, 7],
        "entity_name": [
            "Pfizer Inc.",
            "Pfizer Inc.",
            "Johnson & Johnson",
            "United Technologies Corporation",
            "United Therapeutics Corporation",
            "Quantum Corp.",
            "Quantum Biopharma Ltd.",
        ],
    })
    pharma_tickers_df = pd.DataFrame({
        "Ticker": ["PFE", "JNJ", "UTHR", "QNTM"],
        "Company_Name": [
            "Pfizer Inc",
            "Johnson and Johnson",
            "United Therapeutics Corp",
            "Quantum Biopharma Ltd.",
        ],
        "CIK_10": ["0000078003", "0000200406", "0001082554", "0001406068"],
    })

    result = fuzzy_match_pharma(
        general_df,
        pharma_tickers_df,
        min_score=95,
        verbose=True,
    )
    print(result[[
        "case_id", "entity_name", "match_decision", "match_reason",
        "matched_company_name", "Ticker", "review_candidate_company_name",
        "fuzz_match_score", "candidate_fuzz_match_score",
    ]])
    print(summarize_match_decisions(result))
