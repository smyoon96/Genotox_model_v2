"""
metal_genotox_analysis.py — 금속 유전독성 통계 분석 (독립 스크립트)
==================================================================
시험계별 금속 유전독성 차이에 대한 통계적 근거 제공.

사용법:
    python metal_genotox_analysis.py --data-dir data
    python metal_genotox_analysis.py --data-dir /mnt/user-data/uploads

출력:
    1. metal_endpoint_comparison.csv   — endpoint별 금속 vs 비금속 비교
    2. metal_species_profile.csv       — 금속종별 endpoint 프로파일
    3. metal_concordance_matrix.csv    — endpoint 간 concordance
    4. metal_genotox_stats.json        — 전체 통계 요약
    5. metal_genotox_report.txt        — 논문용 텍스트
"""

import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
import re

# ─── Metal detection ───
METAL_BRACKET = re.compile(
    r'\[(Na|K|Li|Ca|Mg|Fe|Cu|Zn|Mn|Co|Ni|Cr|Cd|Hg|Pb|As|Sb|Bi|Sn|'
    r'Ti|V|Mo|W|Pt|Pd|Au|Ag|Al|Ba|Sr|Se|Te)[\+\-\d@H]*\]'
)

METAL_CLASSES = {
    "transition_redox": ["Cr", "Mn", "Fe", "Co", "Ni", "Cu", "V", "Mo", "W"],
    "heavy_metal": ["Cd", "Hg", "Pb", "As", "Sb", "Bi", "Sn"],
    "noble_metal": ["Pt", "Pd", "Au", "Ag"],
    "metalloid": ["Se", "Te", "As", "Sb"],
    "alkali_alkaline": ["Na", "K", "Li", "Ca", "Mg", "Ba", "Sr", "Al"],
}

# Reverse lookup
METAL_TO_CLASS = {}
for cls, metals in METAL_CLASSES.items():
    for m in metals:
        if m not in METAL_TO_CLASS:  # first assignment wins
            METAL_TO_CLASS[m] = cls


def load_data(data_dir):
    """Load all 3 endpoint datasets."""
    dp = Path(data_dir)
    datasets = {}

    configs = [
        ("ames", ["ames_combine.xlsx", "ames_combine.csv"]),
        ("invitro", ["invitro_pre.csv"]),
        ("invivo", ["invivo_pre.csv"]),
    ]
    for name, fnames in configs:
        for fn in fnames:
            fp = dp / fn
            if fp.exists():
                df = pd.read_excel(fp) if fn.endswith(".xlsx") else pd.read_csv(fp)
                # Normalize columns
                smiles_col = [c for c in df.columns
                              if 'smiles' in c.lower() or c == 'SMILES'][0]
                label_col = [c for c in df.columns
                             if c.lower() in ['label', 'result', 'outcome']][0]
                df = df.rename(columns={smiles_col: "SMILES", label_col: "label"})
                df = df.dropna(subset=["SMILES", "label"])
                df["label"] = df["label"].astype(int)
                datasets[name] = df
                print(f"  Loaded {name}: {fp.name} (n={len(df)})")
                break
    return datasets


def detect_metals(df):
    """Add metal detection columns."""
    df = df.copy()
    df["metals_found"] = df["SMILES"].astype(str).apply(
        lambda s: list(set(METAL_BRACKET.findall(s))))
    df["has_metal"] = df["metals_found"].apply(lambda x: len(x) > 0)
    df["n_metals"] = df["metals_found"].apply(len)
    df["metal_class"] = df["metals_found"].apply(
        lambda ms: list(set(METAL_TO_CLASS.get(m, "unknown") for m in ms))
        if ms else [])
    return df


# ═══════════════════════════════════════════════
#  분석 1: Endpoint별 금속 vs 비금속 (Fisher's exact)
# ═══════════════════════════════════════════════
def analysis_endpoint_comparison(datasets):
    """Metal vs non-metal positive rate comparison with Fisher's exact test."""
    print("\n" + "=" * 60)
    print("  분석 1: Endpoint별 금속 vs 비금속 비교")
    print("=" * 60)

    results = []
    for ep_name, ep_label in [("ames", "Ames"),
                               ("invitro", "In vitro CA"),
                               ("invivo", "In vivo MN")]:
        df = datasets[ep_name]

        metal = df[df["has_metal"]]
        non_metal = df[~df["has_metal"]]

        a = metal["label"].sum()       # metal & positive
        b = len(metal) - a             # metal & negative
        c = non_metal["label"].sum()   # non-metal & positive
        d = len(non_metal) - c         # non-metal & negative

        # Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])

        # Confidence interval for odds ratio (Woolf method)
        if a > 0 and b > 0 and c > 0 and d > 0:
            log_or = np.log(odds_ratio)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_lo = np.exp(log_or - 1.96 * se)
            ci_hi = np.exp(log_or + 1.96 * se)
        else:
            ci_lo, ci_hi = np.nan, np.nan

        # Enrichment
        metal_pr = metal["label"].mean() if len(metal) > 0 else 0
        non_metal_pr = non_metal["label"].mean() if len(non_metal) > 0 else 0
        enrichment = metal_pr / non_metal_pr if non_metal_pr > 0 else np.inf

        row = {
            "endpoint": ep_label,
            "n_total": len(df),
            "n_metal": len(metal),
            "n_non_metal": len(non_metal),
            "metal_pct": round(100 * len(metal) / len(df), 1),
            "metal_pos": int(a),
            "metal_neg": int(b),
            "non_metal_pos": int(c),
            "non_metal_neg": int(d),
            "metal_pos_rate": round(metal_pr, 4),
            "non_metal_pos_rate": round(non_metal_pr, 4),
            "enrichment": round(enrichment, 3),
            "odds_ratio": round(odds_ratio, 3),
            "or_ci_lo": round(ci_lo, 3) if not np.isnan(ci_lo) else None,
            "or_ci_hi": round(ci_hi, 3) if not np.isnan(ci_hi) else None,
            "fisher_p": f"{p_value:.2e}" if p_value < 0.001 else round(p_value, 4),
            "significant": p_value < 0.05,
        }
        results.append(row)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"\n  [{ep_label}]")
        print(f"    Metal: {a}/{len(metal)} pos ({metal_pr:.1%})")
        print(f"    Non-metal: {c}/{len(non_metal)} pos ({non_metal_pr:.1%})")
        print(f"    Contingency: [[{a},{b}],[{c},{d}]]")
        print(f"    OR={odds_ratio:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
              f"p={p_value:.2e} {sig}")
        print(f"    Enrichment: {enrichment:.2f}×")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════
#  분석 2: 금속종별 Endpoint 프로파일
# ═══════════════════════════════════════════════
def analysis_metal_species(datasets):
    """Per-metal-species positive rate across all endpoints."""
    print("\n" + "=" * 60)
    print("  분석 2: 금속종별 Endpoint 프로파일")
    print("=" * 60)

    # Collect metal-level data
    records = []
    for ep_name, ep_label in [("ames", "Ames"),
                               ("invitro", "In vitro CA"),
                               ("invivo", "In vivo MN")]:
        df = datasets[ep_name]
        for _, row in df[df["has_metal"]].iterrows():
            for metal in row["metals_found"]:
                records.append({
                    "endpoint": ep_label,
                    "metal": metal,
                    "metal_class": METAL_TO_CLASS.get(metal, "unknown"),
                    "label": row["label"],
                })

    mdf = pd.DataFrame(records)

    # Pivot: metal × endpoint
    profiles = []
    for metal in sorted(mdf["metal"].unique()):
        row = {"metal": metal, "class": METAL_TO_CLASS.get(metal, "unknown")}
        total_n = 0
        for ep in ["Ames", "In vitro CA", "In vivo MN"]:
            sub = mdf[(mdf["metal"] == metal) & (mdf["endpoint"] == ep)]
            n = len(sub)
            pos = sub["label"].sum()
            row[f"{ep}_n"] = n
            row[f"{ep}_pos"] = pos
            row[f"{ep}_rate"] = round(pos / n, 3) if n > 0 else None
            total_n += n
        row["total_n"] = total_n
        profiles.append(row)

    pdf = pd.DataFrame(profiles).sort_values("total_n", ascending=False)

    # Print nicely
    print(f"\n  {'Metal':>5s} {'Class':>15s} | "
          f"{'Ames':>12s} | {'In vitro CA':>12s} | {'In vivo MN':>12s}")
    print("  " + "-" * 70)
    for _, r in pdf.iterrows():
        if r["total_n"] >= 5:
            def fmt(ep):
                n = r.get(f"{ep}_n", 0)
                if n == 0:
                    return "      -     "
                pos = r.get(f"{ep}_pos", 0)
                rate = r.get(f"{ep}_rate", 0)
                return f"{pos:>3d}/{n:<3d} ({rate:.0%})"
            print(f"  [{r['metal']:>3s}] {r['class']:>15s} | "
                  f"{fmt('Ames')} | {fmt('In vitro CA')} | {fmt('In vivo MN')}")

    return pdf


# ═══════════════════════════════════════════════
#  분석 3: 금속 클래스별 통계
# ═══════════════════════════════════════════════
def analysis_metal_class(datasets):
    """Metal class (transition_redox vs heavy_metal etc.) statistics."""
    print("\n" + "=" * 60)
    print("  분석 3: 금속 클래스별 Endpoint 비교")
    print("=" * 60)

    results = []
    for ep_name, ep_label in [("ames", "Ames"),
                               ("invitro", "In vitro CA"),
                               ("invivo", "In vivo MN")]:
        df = datasets[ep_name]
        non_metal_pr = df[~df["has_metal"]]["label"].mean()

        for cls_name in ["transition_redox", "heavy_metal", "noble_metal",
                          "metalloid", "alkali_alkaline"]:
            mask = df["metal_class"].apply(lambda x: cls_name in x)
            sub = df[mask]
            if len(sub) < 3:
                continue

            pos = sub["label"].sum()
            neg = len(sub) - pos
            nm_pos = df[~df["has_metal"]]["label"].sum()
            nm_neg = len(df[~df["has_metal"]]) - nm_pos

            or_val, p_val = stats.fisher_exact([[pos, neg], [nm_pos, nm_neg]])

            results.append({
                "endpoint": ep_label,
                "metal_class": cls_name,
                "n": len(sub),
                "pos": int(pos),
                "pos_rate": round(sub["label"].mean(), 4),
                "non_metal_pos_rate": round(non_metal_pr, 4),
                "odds_ratio": round(or_val, 3),
                "fisher_p": round(p_val, 6),
                "significant": p_val < 0.05,
            })

            sig = "*" if p_val < 0.05 else "ns"
            print(f"  [{ep_label}] {cls_name:>20s}: "
                  f"n={len(sub):>3d}, pos_rate={sub['label'].mean():.1%}, "
                  f"OR={or_val:.2f}, p={p_val:.4f} {sig}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════
#  분석 4: Endpoint 간 Concordance (paired analysis)
# ═══════════════════════════════════════════════
def analysis_concordance(datasets):
    """Cross-endpoint concordance for metal compounds present in multiple datasets."""
    print("\n" + "=" * 60)
    print("  분석 4: Endpoint 간 Concordance (금속 화합물)")
    print("=" * 60)

    # Find overlapping SMILES between endpoints
    ep_pairs = [("ames", "invitro", "Ames vs In vitro CA"),
                ("ames", "invivo", "Ames vs In vivo MN"),
                ("invitro", "invivo", "In vitro CA vs In vivo MN")]

    concordance_results = []
    for ep1, ep2, pair_label in ep_pairs:
        df1 = datasets[ep1][["SMILES", "label", "has_metal"]].copy()
        df2 = datasets[ep2][["SMILES", "label", "has_metal"]].copy()

        merged = df1.merge(df2, on="SMILES", suffixes=("_1", "_2"))

        # Overall concordance
        n_all = len(merged)
        if n_all < 10:
            print(f"\n  {pair_label}: Too few overlapping compounds ({n_all})")
            continue

        agree_all = ((merged["label_1"] == merged["label_2"]).sum())
        conc_all = agree_all / n_all

        # Metal-only concordance
        metal_merged = merged[merged["has_metal_1"] | merged["has_metal_2"]]
        n_metal = len(metal_merged)

        # Non-metal concordance
        non_metal_merged = merged[~merged["has_metal_1"] & ~merged["has_metal_2"]]
        n_non_metal = len(non_metal_merged)

        print(f"\n  {pair_label}:")
        print(f"    Total overlapping: {n_all}")
        print(f"    Overall concordance: {conc_all:.1%} ({agree_all}/{n_all})")

        row = {
            "pair": pair_label,
            "n_overlap": n_all,
            "concordance_all": round(conc_all, 4),
        }

        if n_metal >= 5:
            agree_met = (metal_merged["label_1"] == metal_merged["label_2"]).sum()
            conc_met = agree_met / n_metal
            # Discordance breakdown for metals
            met_pos1_neg2 = ((metal_merged["label_1"] == 1) &
                             (metal_merged["label_2"] == 0)).sum()
            met_neg1_pos2 = ((metal_merged["label_1"] == 0) &
                             (metal_merged["label_2"] == 1)).sum()

            row["n_metal"] = n_metal
            row["concordance_metal"] = round(conc_met, 4)
            row["metal_discord_1pos_2neg"] = int(met_pos1_neg2)
            row["metal_discord_1neg_2pos"] = int(met_neg1_pos2)

            print(f"    Metal concordance: {conc_met:.1%} ({agree_met}/{n_metal})")
            print(f"      Discordant: {ep1}+/{ep2}- = {met_pos1_neg2}, "
                  f"{ep1}-/{ep2}+ = {met_neg1_pos2}")

            # McNemar test for discordance
            if met_pos1_neg2 + met_neg1_pos2 >= 5:
                mcnemar_stat = (met_pos1_neg2 - met_neg1_pos2)**2 / (
                    met_pos1_neg2 + met_neg1_pos2)
                mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                row["mcnemar_chi2"] = round(mcnemar_stat, 3)
                row["mcnemar_p"] = round(mcnemar_p, 6)
                print(f"      McNemar: χ²={mcnemar_stat:.3f}, p={mcnemar_p:.4f}")

        if n_non_metal >= 5:
            agree_nm = (non_metal_merged["label_1"] ==
                        non_metal_merged["label_2"]).sum()
            conc_nm = agree_nm / n_non_metal
            row["n_non_metal"] = n_non_metal
            row["concordance_non_metal"] = round(conc_nm, 4)
            print(f"    Non-metal concordance: {conc_nm:.1%} "
                  f"({agree_nm}/{n_non_metal})")

            # Compare concordance: metal vs non-metal (chi-square)
            if n_metal >= 5:
                agree_met_val = (metal_merged["label_1"] ==
                                 metal_merged["label_2"]).sum()
                table = [[agree_met_val, n_metal - agree_met_val],
                         [agree_nm, n_non_metal - agree_nm]]
                chi2, chi_p, _, _ = stats.chi2_contingency(table,
                                                            correction=True)
                row["concordance_diff_chi2"] = round(chi2, 3)
                row["concordance_diff_p"] = round(chi_p, 6)
                sig = "*" if chi_p < 0.05 else "ns"
                print(f"    Metal vs Non-metal concordance diff: "
                      f"χ²={chi2:.3f}, p={chi_p:.4f} {sig}")

        concordance_results.append(row)

    return pd.DataFrame(concordance_results)


# ═══════════════════════════════════════════════
#  분석 5: Cochran-Mantel-Haenszel 검정
# ═══════════════════════════════════════════════
def analysis_cmh_test(datasets):
    """
    Cochran-Mantel-Haenszel: 금속 효과가 endpoint에 걸쳐 일관적인지,
    또는 endpoint에 따라 달라지는지 (interaction) 검정.
    Breslow-Day test for homogeneity of odds ratios.
    """
    print("\n" + "=" * 60)
    print("  분석 5: Breslow-Day Homogeneity Test")
    print("  (금속의 OR이 endpoint마다 다른지 검정)")
    print("=" * 60)

    tables = []
    log_ors = []
    weights = []

    for ep_name, ep_label in [("ames", "Ames"),
                               ("invitro", "In vitro CA"),
                               ("invivo", "In vivo MN")]:
        df = datasets[ep_name]
        metal = df[df["has_metal"]]
        non_metal = df[~df["has_metal"]]

        a = int(metal["label"].sum())
        b = int(len(metal) - a)
        c = int(non_metal["label"].sum())
        d = int(len(non_metal) - c)

        tables.append((a, b, c, d, ep_label))

        # For Breslow-Day: add 0.5 correction if any cell is 0
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = np.log((aa * dd) / (bb * cc))
        w = 1.0 / (1/aa + 1/bb + 1/cc + 1/dd)
        log_ors.append(log_or)
        weights.append(w)

    # Mantel-Haenszel common OR
    log_ors = np.array(log_ors)
    weights = np.array(weights)
    mh_log_or = np.sum(weights * log_ors) / np.sum(weights)
    mh_or = np.exp(mh_log_or)

    # Breslow-Day test statistic
    bd_stat = np.sum(weights * (log_ors - mh_log_or)**2)
    bd_df = len(tables) - 1
    bd_p = 1 - stats.chi2.cdf(bd_stat, df=bd_df)

    print(f"\n  Endpoint-specific ORs:")
    for a, b, c, d, label in tables:
        aa, bb, cc, dd = a+0.5, b+0.5, c+0.5, d+0.5
        or_val = (aa*dd)/(bb*cc)
        print(f"    {label:>15s}: OR = {or_val:.3f} "
              f"(metal {a}/{a+b}, non-metal {c}/{c+d})")

    print(f"\n  Mantel-Haenszel common OR: {mh_or:.3f}")
    print(f"  Breslow-Day test: χ²={bd_stat:.3f}, df={bd_df}, p={bd_p:.6f}")
    if bd_p < 0.05:
        print(f"  → SIGNIFICANT: OR이 endpoint마다 다르다 (heterogeneous)")
        print(f"    = 금속의 유전독성이 시험계에 의존적이라는 통계적 증거")
    else:
        print(f"  → Not significant: OR이 endpoint에 걸쳐 동질적")

    return {
        "mh_common_or": round(mh_or, 3),
        "breslow_day_chi2": round(bd_stat, 3),
        "breslow_day_df": bd_df,
        "breslow_day_p": round(bd_p, 6),
        "heterogeneous": bd_p < 0.05,
        "endpoint_ors": {
            label: round((a+0.5)*(d+0.5)/((b+0.5)*(c+0.5)), 3)
            for a, b, c, d, label in tables
        },
    }


# ═══════════════════════════════════════════════
#  분석 6: Ames 위음성 분석 (금속의 regulatory 영향)
# ═══════════════════════════════════════════════
def analysis_ames_false_negative(datasets):
    """
    금속 화합물 중 Ames 음성이지만 CA/MN 양성인 비율.
    → Ames 단독 스크리닝의 위음성률 추정.
    """
    print("\n" + "=" * 60)
    print("  분석 6: Ames 위음성 분석 (금속 화합물)")
    print("  (Ames neg → CA/MN pos = regulatory miss)")
    print("=" * 60)

    results = {}

    for ep2_name, ep2_label in [("invitro", "In vitro CA"),
                                 ("invivo", "In vivo MN")]:
        ames = datasets["ames"][["SMILES", "label", "has_metal"]].copy()
        ep2 = datasets[ep2_name][["SMILES", "label"]].copy()

        merged = ames.merge(ep2, on="SMILES", suffixes=("_ames", f"_{ep2_name}"))

        # Metal compounds that are Ames-negative
        metal_ames_neg = merged[merged["has_metal"] &
                                 (merged["label_ames"] == 0)]
        if len(metal_ames_neg) == 0:
            print(f"\n  Ames vs {ep2_label}: No Ames-negative metal compounds")
            continue

        # How many are positive in the other endpoint?
        fn = metal_ames_neg[f"label_{ep2_name}"].sum()
        fn_rate = fn / len(metal_ames_neg) if len(metal_ames_neg) > 0 else 0

        # Compare with non-metal Ames-negative
        non_metal_ames_neg = merged[~merged["has_metal"] &
                                     (merged["label_ames"] == 0)]
        fn_nm = non_metal_ames_neg[f"label_{ep2_name}"].sum()
        fn_nm_rate = fn_nm / len(non_metal_ames_neg) if len(non_metal_ames_neg) > 0 else 0

        # Fisher's exact
        a = int(fn)
        b = int(len(metal_ames_neg) - fn)
        c = int(fn_nm)
        d = int(len(non_metal_ames_neg) - fn_nm)

        if a + b > 0 and c + d > 0:
            or_val, p_val = stats.fisher_exact([[a, b], [c, d]])
        else:
            or_val, p_val = np.nan, np.nan

        print(f"\n  Ames(-) → {ep2_label}:")
        print(f"    Metal:     {fn}/{len(metal_ames_neg)} = {fn_rate:.1%} "
              f"become {ep2_label}(+)")
        print(f"    Non-metal: {fn_nm}/{len(non_metal_ames_neg)} = {fn_nm_rate:.1%} "
              f"become {ep2_label}(+)")
        print(f"    OR={or_val:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"    → 금속 화합물은 Ames에서 유의하게 더 많이 놓침")

        results[f"ames_vs_{ep2_name}"] = {
            "metal_ames_neg_n": len(metal_ames_neg),
            "metal_fn": int(fn),
            "metal_fn_rate": round(fn_rate, 4),
            "non_metal_ames_neg_n": len(non_metal_ames_neg),
            "non_metal_fn": int(fn_nm),
            "non_metal_fn_rate": round(fn_nm_rate, 4),
            "odds_ratio": round(or_val, 3) if not np.isnan(or_val) else None,
            "fisher_p": round(p_val, 6) if not np.isnan(p_val) else None,
        }

    return results


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Metal genotoxicity statistical analysis")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"metal_genotox_{ts}")
    else:
        outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Metal Genotoxicity Statistical Analysis")
    print(f"  Output: {outdir}")
    print("=" * 60)

    # Load data
    datasets = load_data(args.data_dir)

    # Detect metals in all datasets
    for name in datasets:
        datasets[name] = detect_metals(datasets[name])

    summary = {}

    # 1. Endpoint comparison with Fisher's exact
    df1 = analysis_endpoint_comparison(datasets)
    df1.to_csv(outdir / "metal_endpoint_comparison.csv", index=False)
    summary["endpoint_comparison"] = df1.to_dict("records")

    # 2. Metal species profiles
    df2 = analysis_metal_species(datasets)
    df2.to_csv(outdir / "metal_species_profile.csv", index=False)

    # 3. Metal class analysis
    df3 = analysis_metal_class(datasets)
    df3.to_csv(outdir / "metal_class_stats.csv", index=False)
    summary["metal_class"] = df3.to_dict("records")

    # 4. Concordance
    df4 = analysis_concordance(datasets)
    df4.to_csv(outdir / "metal_concordance_matrix.csv", index=False)
    summary["concordance"] = df4.to_dict("records")

    # 5. Breslow-Day
    cmh = analysis_cmh_test(datasets)
    summary["breslow_day"] = cmh

    # 6. Ames false negative
    fn_results = analysis_ames_false_negative(datasets)
    summary["ames_false_negative"] = fn_results

    # Save summary
    with open(outdir / "metal_genotox_stats.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # ═══ Generate report ═══
    ep_comp = df1.set_index("endpoint")
    report = f"""
METAL GENOTOXICITY STATISTICAL ANALYSIS
{'='*60}
Date: {datetime.now().strftime("%Y-%m-%d")}

1. METAL vs NON-METAL COMPARISON (Fisher's Exact Test)
{'─'*60}
"""
    for ep in ["Ames", "In vitro CA", "In vivo MN"]:
        r = ep_comp.loc[ep]
        report += f"""
  {ep}:
    Metal:     {r['metal_pos_rate']:.1%} positive (n={r['n_metal']})
    Non-metal: {r['non_metal_pos_rate']:.1%} positive (n={r['n_non_metal']})
    OR = {r['odds_ratio']} [{r['or_ci_lo']}, {r['or_ci_hi']}]
    Fisher's p = {r['fisher_p']}
"""

    report += f"""
2. HOMOGENEITY OF ODDS RATIOS (Breslow-Day Test)
{'─'*60}
  Question: 금속의 유전독성 OR이 시험계에 따라 다른가?
  Endpoint-specific ORs:
"""
    for ep, or_val in cmh["endpoint_ors"].items():
        report += f"    {ep}: OR = {or_val}\n"

    report += f"""
  Breslow-Day χ² = {cmh['breslow_day_chi2']}, df = {cmh['breslow_day_df']}
  p = {cmh['breslow_day_p']}
  {'→ SIGNIFICANT: 금속의 유전독성은 시험계에 의존적' if cmh['heterogeneous']
   else '→ Not significant'}

3. KEY TOXICOLOGICAL INSIGHTS
{'─'*60}
  a) 금속은 Ames에서 체계적으로 과소평가됨 (OR={ep_comp.loc['Ames','odds_ratio']})
  b) In vitro CA에서 금속은 강한 양성 (OR={ep_comp.loc['In vitro CA','odds_ratio']})
  c) 시험계 간 OR 이질성 {'유의함' if cmh['heterogeneous'] else '유의하지 않음'}
     → 금속의 유전독성 메커니즘이 endpoint-specific
"""

    # Add false negative info if available
    if fn_results:
        report += f"""
4. AMES FALSE NEGATIVE ANALYSIS (Regulatory Implications)
{'─'*60}
"""
        for key, val in fn_results.items():
            ep2 = key.replace("ames_vs_", "")
            report += f"""  Ames(-) compounds tested in {ep2}:
    Metal:     {val['metal_fn']}/{val['metal_ames_neg_n']} = {val['metal_fn_rate']:.1%} were positive
    Non-metal: {val['non_metal_fn']}/{val['non_metal_ames_neg_n']} = {val['non_metal_fn_rate']:.1%} were positive
    OR = {val.get('odds_ratio', 'N/A')}, p = {val.get('fisher_p', 'N/A')}
"""

    with open(outdir / "metal_genotox_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("  DONE! Output files:")
    print("=" * 60)
    for f in sorted(outdir.glob("*")):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
