
# =========================================================
# Temporal motif mining for strong / medium / weak binders
# =========================================================

import os
import math
import numpy as np
import pandas as pd

from collections import defaultdict
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

mode = "group"


def sequence_events(seq, mode="group"):
    """
    Convert a peptide sequence into position-indexed events.

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
    Check whether a peptide contains:

        event_a -> event_b

    where event_b occurs between min_dist and max_dist positions
    after event_a.
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


# ---------------------------------------------------------
# 2. Count temporal motifs in one peptide group
#    ROW-COUNT VERSION, not weighted by total
# ---------------------------------------------------------

def count_pair_rules_rowcount(
    sequences,
    min_dist=1,
    max_dist=8,
    event_mode="hybrid"
):
    """
    Count temporal pair motifs in a group of peptide sequences.

    A rule is:

        event_a -> event_b

    where event_b appears min_dist..max_dist positions after event_a.

    This version uses row counts only.

    A peptide row contributes 1 to a motif if it contains that motif,
    regardless of the value in the total column.

    Therefore:
        row_count = number of peptide rows containing the motif

        row_support =
            row_count / number of peptide rows in the group
    """

    sequences = list(sequences)

    output_cols = [
        "event_a",
        "event_b",
        "min_dist",
        "max_dist",
        "row_count",
        "row_support"
    ]

    n_rows = len(sequences)

    if n_rows == 0:
        return pd.DataFrame(columns=output_cols)

    rule_counts = defaultdict(int)

    for seq in sequences:

        events_by_pos = sequence_events(seq, mode=event_mode)
        L = len(events_by_pos)

        # Important:
        # Use a set so that the same peptide contributes at most 1 count
        # per motif, even if the motif appears multiple times.
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

    rows = []

    for rule, count in rule_counts.items():

        rows.append({
            "event_a": rule[0],
            "event_b": rule[1],
            "min_dist": rule[2],
            "max_dist": rule[3],
            "row_count": count,
            "row_support": count / n_rows
        })

    return pd.DataFrame(rows, columns=output_cols)
    # ---------------------------------------------------------
# 3. Mine group-specific motifs using ROW SUPPORT
# ---------------------------------------------------------

def mine_group_specific_temporal_motifs(
    df,
    peptide_col="peptide",
    wash4_col="wash4",
    distance_windows=None,
    event_mode="hybrid",
    min_support_target=0.05,
    max_support_other=0.05,
    final_top_n=None
):
    """
    Find temporal motifs that are common in one binder group
    but not common in the other two groups.

    Groups:
        weak   : 0.2 < wash4 < 0.5
        medium : 0.5 < wash4 < 0.8
        strong : wash4 > 0.8

    This version uses ROW-COUNT support, not total-weighted support.

    A motif is specific to one group if:

        row_support_target >= min_support_target

    and

        row_support_other_groups < max_support_other

    Example:
        min_support_target = 0.05

    means:
        the motif appears in at least 5% of peptide rows
        in the target group.
    """

    if distance_windows is None:
        distance_windows = [
            (1, 1),
            (2, 3),
            (4, 6),
            (7, 8),
            (9, 10),
            (11, 12)
        ]

    df = df.copy()

    # Convert wash4 to numeric
    df[wash4_col] = (
        df[wash4_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[wash4_col] = pd.to_numeric(df[wash4_col], errors="coerce")

    df = df.dropna(subset=[peptide_col, wash4_col]).copy()

    # -----------------------------------------------------
    # Define explicit binder groups
    # -----------------------------------------------------

    strong_df = df[df[wash4_col] > 0.8].copy()

    medium_df = df[
        (df[wash4_col] > 0.5) &
        (df[wash4_col] < 0.8)
    ].copy()

    weak_df = df[
        (df[wash4_col] > 0.2) &
        (df[wash4_col] < 0.5)
    ].copy()

    groups = {
        "strong": strong_df,
        "medium": medium_df,
        "weak": weak_df
    }

    print("Group sizes:")
    for group_name, group_df in groups.items():
        print(f"{group_name}: {len(group_df)} peptide rows")

    key_cols = ["event_a", "event_b", "min_dist", "max_dist"]

    all_window_results = []

    # -----------------------------------------------------
    # Mine motifs separately for each distance window
    # -----------------------------------------------------

    for min_dist, max_dist in distance_windows:

        merged_window = None

        for group_name, group_df in groups.items():

            rules = count_pair_rules_rowcount(
                sequences=group_df[peptide_col],
                min_dist=min_dist,
                max_dist=max_dist,
                event_mode=event_mode
            )

            rules = rules.rename(
                columns={
                    "row_count": f"row_count_{group_name}",
                    "row_support": f"row_support_{group_name}"
                }
            )

            keep_cols = key_cols + [
                f"row_count_{group_name}",
                f"row_support_{group_name}"
            ]

            rules = rules[keep_cols].copy()

            if merged_window is None:
                merged_window = rules
            else:
                merged_window = merged_window.merge(
                    rules,
                    on=key_cols,
                    how="outer"
                )

        if merged_window is None or len(merged_window) == 0:
            continue

        # Missing values mean motif was not observed in that group
        for group_name in groups.keys():

            merged_window[f"row_count_{group_name}"] = (
                merged_window[f"row_count_{group_name}"]
                .fillna(0)
                .astype(int)
            )

            merged_window[f"row_support_{group_name}"] = (
                merged_window[f"row_support_{group_name}"]
                .fillna(0.0)
            )

        merged_window["distance_window"] = f"d{min_dist}_{max_dist}"

        all_window_results.append(merged_window)

    if len(all_window_results) == 0:
        print("No temporal motifs found.")

        empty = pd.DataFrame()

        return {
            "strong": empty,
            "medium": empty,
            "weak": empty
        }, empty

    all_rules = pd.concat(all_window_results, ignore_index=True)

    all_rules = all_rules.drop_duplicates(
        subset=["event_a", "event_b", "min_dist", "max_dist"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Extract motifs specific to each group
    # -----------------------------------------------------

    group_specific_tables = {}

    for target_group in ["strong", "medium", "weak"]:

        other_groups = [
            group_name
            for group_name in ["strong", "medium", "weak"]
            if group_name != target_group
        ]

        target_count_col = f"row_count_{target_group}"
        target_support_col = f"row_support_{target_group}"

        other_support_cols = [
            f"row_support_{other_group}"
            for other_group in other_groups
        ]

        n_target_rows = len(groups[target_group])

        if n_target_rows == 0:
            group_specific_tables[target_group] = pd.DataFrame()
            continue

        # Common in target group:
        # motif appears in at least 5% of target-group peptide rows
        target_common_mask = (
            all_rules[target_support_col] >= min_support_target
        )

        # Not common in other groups:
        # motif appears in less than 5% of other-group peptide rows
        other_not_common_mask = np.ones(len(all_rules), dtype=bool)

        for other_support_col in other_support_cols:
            other_not_common_mask &= (
                all_rules[other_support_col] < max_support_other
            )

        specific = all_rules[
            target_common_mask &
            other_not_common_mask
        ].copy()

        specific["specific_to"] = target_group

        specific["target_row_count"] = specific[target_count_col]
        specific["target_row_support"] = specific[target_support_col]

        specific["max_other_row_support"] = (
            specific[other_support_cols].max(axis=1)
        )

        specific["row_support_margin"] = (
            specific["target_row_support"] -
            specific["max_other_row_support"]
        )

        specific["target_row_support_percent"] = (
            specific["target_row_support"] * 100
        )

        specific["max_other_row_support_percent"] = (
            specific["max_other_row_support"] * 100
        )

        specific["row_support_margin_percent"] = (
            specific["row_support_margin"] * 100
        )

        specific = specific.sort_values(
            [
                "row_support_margin",
                "target_row_support",
                "target_row_count"
            ],
            ascending=False
        ).reset_index(drop=True)

        if final_top_n is not None:
            specific = specific.head(final_top_n).reset_index(drop=True)

        group_specific_tables[target_group] = specific

    return group_specific_tables, all_rules

# ---------------------------------------------------------
# 4. Transform sequences into binary motif features
# ---------------------------------------------------------

def transform_sequences_with_rules(
    sequences,
    rules_df,
    prefix="motif",
    event_mode="hybrid"
):
    """
    Convert peptide sequences into a binary motif-feature matrix.

    Each motif rule becomes one binary column:
        1 = peptide contains motif
        0 = peptide does not contain motif
    """

    sequences = list(sequences)
    X = pd.DataFrame(index=range(len(sequences)))

    if rules_df is None or len(rules_df) == 0:
        return X

    parsed_rules = []

    for idx, row in rules_df.reset_index(drop=True).iterrows():

        event_a = row["event_a"]
        event_b = row["event_b"]
        min_dist = int(row["min_dist"])
        max_dist = int(row["max_dist"])

        col = f"{prefix}_{idx}_{event_a}_to_{event_b}_d{min_dist}_{max_dist}"

        parsed_rules.append(
            (col, event_a, event_b, min_dist, max_dist)
        )

    for seq_idx, seq in enumerate(sequences):

        events_by_pos = sequence_events(seq, mode=event_mode)

        for col, event_a, event_b, min_dist, max_dist in parsed_rules:

            X.loc[seq_idx, col] = int(
                peptide_contains_rule(
                    events_by_pos=events_by_pos,
                    event_a=event_a,
                    event_b=event_b,
                    min_dist=min_dist,
                    max_dist=max_dist
                )
            )

    X = X.fillna(0).astype(np.int8)

    return X


def transform_sequences_with_rules_parallel(
    sequences,
    rules_df,
    prefix="motif",
    event_mode="hybrid",
    n_jobs=None
):
    """
    Parallel version of transform_sequences_with_rules().
    """

    if n_jobs is None:
        n_jobs = os.cpu_count()

    sequences = list(sequences)

    if rules_df is None or len(rules_df) == 0:
        return pd.DataFrame(index=range(len(sequences)))

    chunks = np.array_split(sequences, n_jobs)

    def transform_chunk(seq_chunk):
        return transform_sequences_with_rules(
            sequences=seq_chunk,
            rules_df=rules_df,
            prefix=prefix,
            event_mode=event_mode
        )

    df_chunks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(transform_chunk)(chunk)
        for chunk in chunks
    )

    return pd.concat(df_chunks, ignore_index=True)


# ---------------------------------------------------------
# 5. Run motif mining
# ---------------------------------------------------------

distance_windows = [
    (1, 1),
    (2, 3),
    (4, 6),
    (7, 8),
    (9, 10),
    (11, 12)
]

group_specific_motifs, all_motif_supports = mine_group_specific_temporal_motifs(
    df=features_df_train,
    peptide_col="peptide",
    wash4_col="wash4",
    distance_windows=distance_windows,
    event_mode=mode,
    min_support_target=0.05,
    max_support_other=0.05,
    final_top_n=None
)

strong_specific_motifs = group_specific_motifs["strong"]
medium_specific_motifs = group_specific_motifs["medium"]
weak_specific_motifs = group_specific_motifs["weak"]


# ---------------------------------------------------------
# 6. Save motif tables
# ---------------------------------------------------------

strong_specific_motifs.to_csv(
    "strong_specific_temporal_motifs.csv",
    index=False
)

medium_specific_motifs.to_csv(
    "medium_specific_temporal_motifs.csv",
    index=False
)

weak_specific_motifs.to_csv(
    "weak_specific_temporal_motifs.csv",
    index=False
)

all_motif_supports.to_csv(
    "all_temporal_motif_supports_strong_medium_weak.csv",
    index=False
)

print("Strong-specific motifs:", strong_specific_motifs.shape)
print("Medium-specific motifs:", medium_specific_motifs.shape)
print("Weak-specific motifs:", weak_specific_motifs.shape)
print("All motif supports:", all_motif_supports.shape)

#6.1 Show the motifs
cols_to_show = [
    "specific_to",
    "event_a",
    "event_b",
    "distance_window",
    "target_row_count",
    "target_row_support_percent",
    "max_other_row_support_percent",
    "row_support_margin_percent"
]

display(strong_specific_motifs[cols_to_show].head(30))
display(medium_specific_motifs[cols_to_show].head(30))
display(weak_specific_motifs[cols_to_show].head(30))


strong_specific_motifs.to_csv(
    "strong_specific_temporal_motifs.csv",
    index=False
)

medium_specific_motifs.to_csv(
    "medium_specific_temporal_motifs.csv",
    index=False
)

weak_specific_motifs.to_csv(
    "weak_specific_temporal_motifs.csv",
    index=False
)


# ---------------------------------------------------------
# 7. Optional: create motif-feature matrices
# ---------------------------------------------------------
# Use this only if you want to add the group-specific motifs
# as binary features to your train/test datasets.

all_specific_motifs = pd.concat(
    [
        strong_specific_motifs,
        medium_specific_motifs,
        weak_specific_motifs
    ],
    axis=0,
    ignore_index=True
)

all_specific_motifs = all_specific_motifs.drop_duplicates(
    subset=["event_a", "event_b", "min_dist", "max_dist"]
).reset_index(drop=True)

X_group_motif_train = transform_sequences_with_rules_parallel(
    sequences=features_df_train["peptide"],
    rules_df=all_specific_motifs,
    prefix="group_specific_motif",
    event_mode=mode,
    n_jobs=os.cpu_count()
)

X_group_motif_test = transform_sequences_with_rules_parallel(
    sequences=features_df_test["peptide"],
    rules_df=all_specific_motifs,
    prefix="group_specific_motif",
    event_mode=mode,
    n_jobs=os.cpu_count()
)

features_df_train = pd.concat(
    [
        features_df_train.reset_index(drop=True),
        X_group_motif_train.reset_index(drop=True)
    ],
    axis=1
)

features_df_test = pd.concat(
    [
        features_df_test.reset_index(drop=True),
        X_group_motif_test.reset_index(drop=True)
    ],
    axis=1
)

features_df_train = features_df_train.loc[
    :,
    ~features_df_train.columns.duplicated()
]

features_df_test = features_df_test.loc[
    :,
    ~features_df_test.columns.duplicated()
]
print("X_group_motif_train shape:", X_group_motif_train.shape)
print("X_group_motif_test shape:", X_group_motif_test.shape)