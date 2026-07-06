#Tempoary motif features
import numpy as np
import pandas as pd
from collections import defaultdict

import os
from joblib import Parallel, delayed

# ---------------------------------------------------------
# 1. Define event representation
# ---------------------------------------------------------

COORD = set("HCDEY")
COORD_EXT = set("HCDEYST")
ACIDIC = set("DE")
BASIC = set("KRH")
HYDROPHOBIC = set("AILMFWVYC")
POLAR = set("STNQHDEKR")
AROMATIC = set("FWY")

mode = "hybrid"  # default mode for event representation

def sequence_events(seq, mode):
    """
    Convert a peptide into position-indexed events.

    mode='aa'     -> only raw amino acids
    mode='group'  -> only biochemical groups
    mode='hybrid' -> both raw amino acids and biochemical groups
    """
    seq = str(seq).strip().upper()
    events_by_pos = []

    for aa in seq:
        events = []

        if mode in ["aa", "hybrid"]:
            events.append(f"AA_{aa}")

        if mode in ["group", "hybrid"]:
            if aa in COORD:
                events.append("PROP_COORD")
            if aa in COORD_EXT:
                events.append("PROP_COORD_EXT")
            if aa in ACIDIC:
                events.append("PROP_ACIDIC")
            if aa in BASIC:
                events.append("PROP_BASIC")
            if aa in HYDROPHOBIC:
                events.append("PROP_HYDROPHOBIC")
            if aa in POLAR:
                events.append("PROP_POLAR")
            if aa in AROMATIC:
                events.append("PROP_AROMATIC")

        events_by_pos.append(events)

    return events_by_pos

def peptide_contains_rule(events_by_pos, event_a, event_b, min_dist, max_dist):
    """
    Checks whether event_a is followed by event_b after min_dist..max_dist positions.
    Does the peptide contain a coordinating residue followed by an acidic residue
    between 1 and 8 positions later?
    """
    L = len(events_by_pos)

    for i in range(L):
        if event_a not in events_by_pos[i]:
            continue

        for d in range(min_dist, max_dist + 1):
            j = i + d
            if j < L and event_b in events_by_pos[j]:
                return True

    return False


def mine_pair_rules(
    sequences,
    min_support=0.05,
    min_count=10,
    min_dist=1,
    max_dist=8,
    event_mode= mode
):
    """
    Mine position-independent temporal pair motifs:
    event_a -> event_b with flexible spacing.
    event_a at position i
    event_b at position i + d
    where d is between min_dist and max_dist
    Then it counts in how many peptides each rule appears.
    """
    sequences = list(sequences)
    n = len(sequences)

    rule_counts = defaultdict(int)

    for seq in sequences:
        events_by_pos = sequence_events(seq, mode)
        L = len(events_by_pos)

        rules_in_seq = set()

        for i in range(L):
            for event_a in events_by_pos[i]:
                for d in range(min_dist, max_dist + 1):
                    j = i + d
                    if j >= L:
                        continue
                    for event_b in events_by_pos[j]:
                        rule = (event_a, event_b, min_dist, max_dist)
                        rules_in_seq.add(rule)

        for rule in rules_in_seq:
            rule_counts[rule] += 1

    rules = []
    for rule, count in rule_counts.items():
        support = count / n
        if support >= min_support and count >= min_count:
            rules.append({
                "event_a": rule[0],
                "event_b": rule[1],
                "min_dist": rule[2],
                "max_dist": rule[3],
                "count": count,
                "support": support
            })

    return pd.DataFrame(rules)

def mine_contrastive_temporal_motifs(
    df,
    peptide_col="peptide",
    score_col="coam",
    high_quantile=0.75,
    low_quantile=0.25,
    min_support=0.05,
    min_count=10,
    min_dist=1,
    max_dist=8,
    event_mode= mode,
    top_n=200
):
    """
    Mine motifs separately in high-score and low-score peptides,
    then keep motifs that differ most between groups.
    """

    high_thr = df[score_col].quantile(high_quantile)
    low_thr = df[score_col].quantile(low_quantile)

    high_df = df[df[score_col] >= high_thr].copy()
    low_df = df[df[score_col] <= low_thr].copy()

    high_rules = mine_pair_rules(
        high_df[peptide_col],
        min_support=min_support,
        min_count=min_count,
        min_dist=min_dist,
        max_dist=max_dist,
        event_mode=event_mode
    )

    low_rules = mine_pair_rules(
        low_df[peptide_col],
        min_support=min_support,
        min_count=min_count,
        min_dist=min_dist,
        max_dist=max_dist,
        event_mode=event_mode
    )

    high_rules["hybrid"] = "high"
    low_rules["hybrid"] = "low"

    all_rules = pd.concat([high_rules, low_rules], ignore_index=True)

    key_cols = ["event_a", "event_b", "min_dist", "max_dist"]

    high_support = high_rules.set_index(key_cols)["support"]
    low_support = low_rules.set_index(key_cols)["support"]

    all_keys = high_support.index.union(low_support.index)

    contrast_rows = []

    for key in all_keys:
        hs = high_support.get(key, 0.0)
        ls = low_support.get(key, 0.0)

        contrast_rows.append({
            "event_a": key[0],
            "event_b": key[1],
            "min_dist": key[2],
            "max_dist": key[3],
            "support_high": hs,
            "support_low": ls,
            "support_diff": hs - ls,
            "abs_support_diff": abs(hs - ls)
        })

    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df = contrast_df.sort_values("abs_support_diff", ascending=False)

    if top_n is not None:
        contrast_df = contrast_df.head(top_n).reset_index(drop=True)

    return contrast_df

def mine_contrastive_temporal_motifs_distance_windows(
    df,
    peptide_col="peptide",
    score_col="coam",
    high_quantile=0.75,
    low_quantile=0.25,
    min_support=0.05,
    min_count=10,
    distance_windows=None,
    event_mode=mode,
    top_n_per_window=None,
    final_top_n=300
):
    """
    Mines contrastive temporal motifs across multiple distance windows.

    Example:
    distance_windows = [(1, 1), (2, 3), (4, 6), (7, 8)]

    This lets the model distinguish nearby motifs from long-range motifs.
    """

    if distance_windows is None:
        distance_windows = [(1, 8)]

    all_rules = []

    for min_dist, max_dist in distance_windows:
        rules = mine_contrastive_temporal_motifs(
            df=df,
            peptide_col=peptide_col,
            score_col=score_col,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
            min_support=min_support,
            min_count=min_count,
            min_dist=min_dist,
            max_dist=max_dist,
            event_mode=event_mode,
            top_n=top_n_per_window
        )

        rules["distance_window"] = f"d{min_dist}_{max_dist}"
        all_rules.append(rules)

    all_rules = pd.concat(all_rules, axis=0, ignore_index=True)

    # Remove duplicate rules, just in case
    all_rules = all_rules.drop_duplicates(
        subset=["event_a", "event_b", "min_dist", "max_dist"]
    )

    # Keep the globally strongest contrastive motifs
    all_rules = all_rules.sort_values(
        "abs_support_diff",
        ascending=False
    ).reset_index(drop=True)

    if final_top_n is not None:
        all_rules = all_rules.head(final_top_n).reset_index(drop=True)

    return all_rules

def transform_sequences_with_rules_parallel(
    sequences,
    rules_df,
    prefix="motif",
    event_mode=mode,
    n_jobs=None
):
    if n_jobs is None:
        n_jobs = os.cpu_count()

    def transform_chunk(seq_chunk):
        return transform_sequences_with_rules(
            seq_chunk,
            rules_df,
            prefix=prefix,
            event_mode=event_mode
        )

    chunks = np.array_split(list(sequences), n_jobs)
    df_chunks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(transform_chunk)(chunk) for chunk in chunks
    )
    return pd.concat(df_chunks, ignore_index=True)

def transform_sequences_with_rules(sequences, rules_df, prefix="motif", event_mode=mode):
    """
    Convert peptide sequences into a binary motif-feature matrix.
    Each rule becomes one column.
    """
    sequences = list(sequences)
    X = pd.DataFrame(index=range(len(sequences)))

    parsed_rules = []

    for idx, row in rules_df.reset_index(drop=True).iterrows():
        event_a = row["event_a"]
        event_b = row["event_b"]
        min_dist = int(row["min_dist"])
        max_dist = int(row["max_dist"])

        col = f"{prefix}_{idx}_{event_a}_to_{event_b}_d{min_dist}_{max_dist}"
        parsed_rules.append((col, event_a, event_b, min_dist, max_dist))

    for seq_idx, seq in enumerate(sequences):
        events_by_pos = sequence_events(seq, mode=event_mode)

        for col, event_a, event_b, min_dist, max_dist in parsed_rules:
            X.loc[seq_idx, col] = int(
                peptide_contains_rule(
                    events_by_pos,
                    event_a,
                    event_b,
                    min_dist,
                    max_dist
                )
            )

    X = X.fillna(0).astype(np.int8)
    return X

distance_windows = [
    (1, 1),   # direct neighbors
    (2, 3),   # short-range motif spacing
    (4, 6),   # medium-range spacing
    (7, 8),  
    (9, 10),
    (11, 12) # long-range spacing in 12-aa peptides
]

retention_rules = mine_contrastive_temporal_motifs_distance_windows(
    df=features_df_train,
    peptide_col="peptide",
    score_col="wash4",
    high_quantile=0.75,
    low_quantile=0.25,
    min_support=0.05,
    min_count=10,
    distance_windows=distance_windows,
    event_mode=mode,
    top_n_per_window=None,
    final_top_n=300
)

X_ret_train = transform_sequences_with_rules_parallel(
    features_df_train["peptide"],
    retention_rules,
    prefix="retention_motif",
    event_mode=mode,
    n_jobs= os.cpu_count()
)

X_ret_test = transform_sequences_with_rules_parallel(
    features_df_test["peptide"],
    retention_rules,
    prefix="retention_motif",
    event_mode=mode,
    n_jobs= os.cpu_count()
)

features_df_train = pd.concat(
    [
        features_df_train.reset_index(drop=True),
        X_ret_train.reset_index(drop=True)
    ],
    axis=1
)

features_df_test = pd.concat(
    [
        features_df_test.reset_index(drop=True),
        X_ret_test.reset_index(drop=True)
    ],
    axis=1
)

features_df_train = features_df_train.loc[:, ~features_df_train.columns.duplicated()]
features_df_test = features_df_test.loc[:, ~features_df_test.columns.duplicated()]

print("X_ret_train shape:", X_ret_train.shape)
print("X_ret_test  shape:", X_ret_test.shape)