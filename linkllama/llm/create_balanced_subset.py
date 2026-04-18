#!/usr/bin/env python3
"""
Create a balanced training subset from the full CSV to reduce linker memorization bias.

Strategies:
1. cap: Limit max occurrences per linker.
2. hybrid: Cap high-frequency linkers, down-sample the rest.

When capping, sampling is diversity-aware: sort by mol properties (mol_weight,
mol_logp, mol_tpsa, etc.), then select 1 sample every n so the subset spans
property space. Use --no-diversity-sampling for plain random sampling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random

# Mol property columns used for property-based diversity (use only those present)
DEFAULT_PROPERTY_COLS = [
    'mol_weight', 'mol_logp', 'mol_tpsa', 'mol_num_hbd', 'mol_num_hba',
    'distance_angstrom', 'angle_degrees',
]


def _get_property_cols(df):
    """Return list of diversity property columns that exist in df."""
    return [c for c in DEFAULT_PROPERTY_COLS if c in df.columns]


def sample_by_properties(linker_group, target_n, property_cols, random_seed=42):
    """
    Sample up to target_n rows: sort by mol properties, then take 1 every n
    (systematic sampling) so the subset is diverse in property space.
    """
    if len(linker_group) <= target_n or not property_cols:
        return linker_group if len(linker_group) <= target_n else linker_group.sample(n=target_n, random_state=random_seed)

    rng = np.random.default_rng(random_seed)
    sorted_df = linker_group.sort_values(by=property_cols, ascending=True, na_position='last').reset_index(drop=True)
    N = len(sorted_df)
    step = N / target_n
    start = rng.integers(0, max(1, int(step)))
    indices = (start + np.arange(target_n) * step).astype(int)
    indices = np.clip(indices, 0, N - 1)
    indices = np.unique(indices)
    if len(indices) < target_n:
        short = target_n - len(indices)
        extra = rng.choice(np.setdiff1d(np.arange(N), indices), size=min(short, N - len(indices)), replace=False)
        indices = np.concatenate([indices, extra])
    return sorted_df.iloc[indices[:target_n]].reset_index(drop=True)


def _sample_capped(linker_group, target_n, diversity_sampling, property_cols, random_seed):
    """Sample target_n rows: by property diversity if enabled, else random."""
    if target_n >= len(linker_group):
        return linker_group
    if diversity_sampling and property_cols:
        return sample_by_properties(linker_group, target_n, property_cols, random_seed)
    return linker_group.sample(n=target_n, random_state=random_seed)


def strategy_cap_based(df, max_occurrences_per_linker=100, random_seed=42, diversity_sampling=True,
                      property_cols=None):
    """Cap the maximum number of times each linker can appear. When capping, sample by property diversity."""
    if property_cols is None:
        property_cols = _get_property_cols(df)
    print(f"\nStrategy: Cap-based (max {max_occurrences_per_linker} per linker, diversity_sampling={diversity_sampling})")
    if property_cols:
        print(f"  Property cols: {property_cols}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    linker_counts = df['linker'].value_counts()
    print(f"  Original: {len(df)} rows, {len(linker_counts)} unique linkers")

    balanced_rows = []
    for i, (linker, count) in enumerate(linker_counts.items()):
        linker_group = df[df['linker'] == linker]
        if len(linker_group) <= max_occurrences_per_linker:
            balanced_rows.append(linker_group)
        else:
            balanced_rows.append(_sample_capped(
                linker_group, max_occurrences_per_linker, diversity_sampling, property_cols, random_seed
            ))
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(linker_counts)} linkers...", end='\r')
    print()

    balanced_rest = pd.concat(balanced_rows, ignore_index=True)
    balanced_rest = balanced_rest.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_rest = len(df)
    print(f"  Processed (capped) linkers: {len(balanced_rest)} rows, {balanced_rest['linker'].nunique()} unique linkers")
    print(f"  Reduction in this subset: {n_rest - len(balanced_rest)} rows removed")
    return balanced_rest


def strategy_hybrid(df, keep_under=50, hard_cap=500, mid_keep_fraction=0.9, random_seed=42,
                    diversity_sampling=True, property_cols=None):
    """
    Hybrid: three tiers.
    - Occurrences ≤ keep_under (e.g. 50): keep all, no trim.
    - Occurrences > hard_cap (e.g. 500): cap at hard_cap (property-diverse sample).
    - Occurrences in (keep_under, hard_cap]: trim slightly (keep mid_keep_fraction, e.g. 90%).
    """
    if property_cols is None:
        property_cols = _get_property_cols(df)
    print(f"\nStrategy: Hybrid (keep all ≤{keep_under}, hard cap at {hard_cap}, mid range keep {mid_keep_fraction:.0%})")
    print(f"  Diversity sampling: {diversity_sampling}")
    if property_cols:
        print(f"  Property cols: {property_cols}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    linker_counts = df['linker'].value_counts()
    # Three buckets: keep all, trim slightly, hard cap
    keep_all = linker_counts[linker_counts <= keep_under].index
    mid_range = linker_counts[(linker_counts > keep_under) & (linker_counts <= hard_cap)].index
    cap_linkers = linker_counts[linker_counts > hard_cap].index
    print(f"  Keep all (≤{keep_under}): {len(keep_all)} linkers")
    print(f"  Trim slightly ({keep_under+1}–{hard_cap}): {len(mid_range)} linkers")
    print(f"  Hard cap at {hard_cap} (>{hard_cap}): {len(cap_linkers)} linkers")

    balanced_rows = []
    for linker in keep_all:
        balanced_rows.append(df[df['linker'] == linker])
    for i, linker in enumerate(mid_range):
        linker_group = df[df['linker'] == linker]
        count = len(linker_group)
        n_take = max(1, int(round(count * mid_keep_fraction)))
        n_take = min(n_take, count)
        balanced_rows.append(_sample_capped(
            linker_group, n_take, diversity_sampling and n_take < count, property_cols, random_seed
        ))
        if (i + 1) % 5000 == 0 and len(mid_range) > 5000:
            print(f"  Processed {i+1}/{len(mid_range)} mid-range linkers...", end='\r')
    if len(mid_range) > 5000:
        print()
    for i, linker in enumerate(cap_linkers):
        linker_group = df[df['linker'] == linker]
        n_take = min(hard_cap, len(linker_group))
        balanced_rows.append(_sample_capped(
            linker_group, n_take, diversity_sampling, property_cols, random_seed
        ))
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(cap_linkers)} capped linkers...", end='\r')
    print()

    balanced_rest = pd.concat(balanced_rows, ignore_index=True)
    balanced_rest = balanced_rest.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_rest = len(df)
    print(f"  Processed (trimmed/capped) linkers: {len(balanced_rest)} rows, {balanced_rest['linker'].nunique()} unique linkers")
    print(f"  Reduction in this subset: {n_rest - len(balanced_rest)} rows removed")
    return balanced_rest


def analyze_balanced_subset(n_original, original_counts, total_balanced_rows, balanced_counts, output_dir, analysis_stem=None):
    """Analyze and compare original vs balanced subset (uses precomputed counts to avoid holding full dfs)."""
    print("\n" + "="*60)
    print("BALANCED SUBSET ANALYSIS")
    print("="*60)

    print(f"\nOriginal dataset:")
    print(f"  Total rows: {n_original:,}")
    print(f"  Unique linkers: {len(original_counts):,}")
    print(f"  Most common linker: {original_counts.max():,} occurrences")
    print(f"  Median occurrences: {original_counts.median():.1f}")

    print(f"\nBalanced dataset:")
    print(f"  Total rows: {total_balanced_rows:,}")
    print(f"  Unique linkers: {len(balanced_counts):,}")
    print(f"  Most common linker: {balanced_counts.max():,} occurrences")
    print(f"  Median occurrences: {balanced_counts.median():.1f}")

    max_reduction = (1 - balanced_counts.max() / original_counts.max()) * 100
    print(f"\nImprovement:")
    print(f"  Max occurrence reduction: {max_reduction:.1f}%")
    print(f"  Dataset size reduction: {(1 - total_balanced_rows / n_original)*100:.1f}%")

    name = (analysis_stem + '_analysis') if analysis_stem else 'balanced_subset_analysis'
    analysis_path = output_dir / f'{name}.txt'
    with open(analysis_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BALANCED SUBSET ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Original dataset:\n")
        f.write(f"  Total rows: {n_original:,}\n")
        f.write(f"  Unique linkers: {len(original_counts):,}\n")
        f.write(f"  Most common linker: {original_counts.max():,} occurrences\n")
        f.write(f"  Median occurrences: {original_counts.median():.1f}\n\n")
        f.write(f"Balanced dataset:\n")
        f.write(f"  Total rows: {total_balanced_rows:,}\n")
        f.write(f"  Unique linkers: {len(balanced_counts):,}\n")
        f.write(f"  Most common linker: {balanced_counts.max():,} occurrences\n")
        f.write(f"  Median occurrences: {balanced_counts.median():.1f}\n\n")
        f.write(f"Top 20 linkers in balanced set:\n")
        for i, (linker, count) in enumerate(balanced_counts.head(20).items(), 1):
            f.write(f"  {i:2d}. {linker[:60]}... ({count} occurrences)\n")

    print(f"\nSaved analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a balanced training subset to reduce linker memorization bias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cap each linker to 100 occurrences
  %(prog)s input.csv -s cap --max-occurrences 100 -o balanced.csv
  
  # Hybrid: keep ≤50, cap at 500, trim 50–500 slightly (90%%)
  %(prog)s input.csv -s hybrid --keep-under 50 --hard-cap 500 --mid-keep-fraction 0.9 -o balanced.csv
        """
    )
    
    parser.add_argument('input_csv', type=Path, help='Input CSV file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output CSV file path')
    parser.add_argument('-s', '--strategy', type=str, required=True, choices=['cap', 'hybrid'],
                        help='Balancing strategy to use')
    parser.add_argument('--max-occurrences', type=int, default=100,
                        help='Max occurrences per linker (cap strategy)')
    parser.add_argument('--keep-under', type=int, default=50,
                        help='Hybrid: keep all linkers with ≤ this many occurrences (no trim)')
    parser.add_argument('--hard-cap', type=int, default=500,
                        help='Hybrid: cap linkers with more than this at hard_cap (property-diverse sample)')
    parser.add_argument('--mid-keep-fraction', type=float, default=0.9,
                        help='Hybrid: for linkers with count in (keep_under, hard_cap], keep this fraction (e.g. 0.9 = 90%%)')
    parser.add_argument('--no-diversity-sampling', action='store_true',
                        help='Use random sampling when capping instead of property-based diversity')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    diversity_sampling = not args.no_diversity_sampling

    if args.strategy == 'hybrid':
        if args.keep_under < 0 or args.hard_cap <= args.keep_under:
            parser.error("hybrid requires --keep-under >= 0 and --hard-cap > --keep-under")
        if not 0 < args.mid_keep_fraction <= 1:
            parser.error("hybrid requires 0 < --mid-keep-fraction <= 1")
    
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv, low_memory=False)
    n_original = len(df)
    linker_counts = df['linker'].value_counts()
    original_counts = linker_counts.copy()
    print(f"Loaded {n_original:,} rows, {len(linker_counts):,} unique linkers")

    # Define linkers to keep in full (≤ threshold); write them first to reduce memory
    if args.strategy == 'cap':
        keep_all = linker_counts[linker_counts <= args.max_occurrences].index
    else:
        keep_all = linker_counts[linker_counts <= args.keep_under].index

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df_keep = df[df['linker'].isin(keep_all)]
    n_keep = len(df_keep)
    thresh = args.max_occurrences if args.strategy == 'cap' else args.keep_under
    print(f"\nWriting {n_keep:,} rows (linkers with ≤{thresh} occurrences) to {args.output}...")
    df_keep.to_csv(args.output, index=False)
    del df_keep

    df_rest = df[~df['linker'].isin(keep_all)].copy()
    del df
    print(f"  Kept {n_keep:,} rows in full; processing {len(df_rest):,} remaining rows (fewer linkers, lower memory).")

    property_cols = _get_property_cols(df_rest)
    if args.strategy == 'cap':
        balanced_rest = strategy_cap_based(
            df_rest, args.max_occurrences, args.random_seed,
            diversity_sampling=diversity_sampling,
            property_cols=property_cols,
        )
    else:
        balanced_rest = strategy_hybrid(
            df_rest, keep_under=args.keep_under, hard_cap=args.hard_cap,
            mid_keep_fraction=args.mid_keep_fraction, random_seed=args.random_seed,
            diversity_sampling=diversity_sampling,
            property_cols=property_cols,
        )

    print(f"\nAppending {len(balanced_rest):,} processed rows to {args.output}...")
    balanced_rest.to_csv(args.output, mode='a', header=False, index=False)
    total_balanced = n_keep + len(balanced_rest)
    keep_counts = original_counts[original_counts.index.isin(keep_all)]
    balanced_counts = pd.concat([keep_counts, balanced_rest['linker'].value_counts()])
    del balanced_rest, df_rest

    analyze_balanced_subset(
        n_original, original_counts, total_balanced, balanced_counts,
        output_dir, analysis_stem=args.output.stem,
    )
    print(f"\nSaved {total_balanced:,} rows total to {args.output}")
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
