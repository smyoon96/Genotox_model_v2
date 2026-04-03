"""
step2_feature_build.py — Broad Feature Table 생성
==================================================
Prompt 2: raw split + fg_descriptor_analysis → endpoint별 broad feature CSV
"""
import sys, json, logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import (
    discover_split_files, ensure_fg_dir, find_fg_files,
    safe_read_csv, unify_smiles_col, clean_merge_duplicates,
    normalize_merge_key, classify_columns, check_broadfp_dirty
)
from utils.feature_utils import (
    add_fp_columns, safe_bool_mask, classify_feature_blocks
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step2_feature_build.log", mode="w"),
    ]
)
logger = logging.getLogger("step2")


def build_broad_features(endpoint: str, force_rebuild: bool = False) -> bool:
    """단일 endpoint의 broad feature table 생성"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Building broad features: {endpoint}")
    logger.info(f"{'='*50}")

    # dirty check
    train_out = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
    test_out = cfg.BROADFP_DIR / f"{endpoint}_broadfp_test.csv"

    if not force_rebuild and train_out.exists() and test_out.exists():
        if not check_broadfp_dirty(train_out) and not check_broadfp_dirty(test_out):
            logger.info(f"[{endpoint}] Clean broadfp files exist — skip")
            return True

    # ── 1. Raw split 읽기 ──
    train_path, test_path = discover_split_files(endpoint)
    if train_path is None:
        logger.error(f"[{endpoint}] Split files not found")
        return False

    df_train = safe_read_csv(train_path)
    df_test = safe_read_csv(test_path)
    logger.info(f"  Raw train: {df_train.shape}, test: {df_test.shape}")

    # ── 2. SMILES 통일 ──
    df_train = unify_smiles_col(df_train)
    df_test = unify_smiles_col(df_test)

    # ── 3. fg_descriptor_analysis merge ──
    fg_dir = ensure_fg_dir()
    if fg_dir:
        fg_files = find_fg_files(fg_dir, endpoint)

        # canonical analysis
        if fg_files["canonical"]:
            logger.info(f"  Merging canonical: {fg_files['canonical'].name}")
            df_ca = safe_read_csv(fg_files["canonical"])

            # valid 컬럼 boolean 변환
            for col in df_ca.columns:
                if "valid" in col.lower():
                    df_ca[col] = safe_bool_mask(df_ca[col])

            # merge
            if "No" in df_ca.columns and "No" in df_train.columns:
                df_ca["No"] = normalize_merge_key(df_ca["No"])
                df_train = df_train.merge(df_ca, on="No", how="left", suffixes=("", "_ca"))
                df_test = df_test.merge(df_ca, on="No", how="left", suffixes=("", "_ca"))
            else:
                logger.warning(f"  No merge key for canonical analysis")

        # preprocessed views
        if fg_files["views"]:
            logger.info(f"  Merging views: {fg_files['views'].name}")
            df_views = safe_read_csv(fg_files["views"])
            if "No" in df_views.columns and "No" in df_train.columns:
                df_views["No"] = normalize_merge_key(df_views["No"])
                # 중복 컬럼 방지
                existing_train = set(df_train.columns)
                view_cols = ["No"] + [c for c in df_views.columns
                                       if c != "No" and c not in existing_train]
                df_views_sub = df_views[view_cols].drop_duplicates(subset=["No"])
                df_train = df_train.merge(df_views_sub, on="No", how="left", suffixes=("", "_vw"))
                df_test = df_test.merge(df_views_sub, on="No", how="left", suffixes=("", "_vw"))
    else:
        logger.warning("fg_descriptor_analysis not available")

    # ── 4. Merge 잔여 정리 ──
    df_train = clean_merge_duplicates(df_train)
    df_test = clean_merge_duplicates(df_test)

    # ── 5. Morgan Fingerprint 생성 ──
    smiles_col = None
    for alias in ["canonical_smiles", "standardized_smiles", "SMILES"]:
        if alias in df_train.columns:
            smiles_col = alias
            break

    if smiles_col:
        logger.info(f"  Generating Morgan FP (radius={cfg.FP_RADIUS}, nBits={cfg.FP_NBITS})...")
        df_train = add_fp_columns(df_train, smiles_col, cfg.FP_RADIUS, cfg.FP_NBITS)
        df_test = add_fp_columns(df_test, smiles_col, cfg.FP_RADIUS, cfg.FP_NBITS)
    else:
        logger.warning("  No SMILES column — FP generation skipped")

    # ── 6. RDKit 전자적 기술자 계산 (SMILES → 직접 계산) ──
    if cfg.COMPUTE_ELECTRONIC and smiles_col:
        try:
            from utils.qm_descriptors import add_electronic_descriptors
            logger.info(f"  Computing electronic descriptors (HOMO/LUMO proxy, charges, etc.)...")
            df_train = add_electronic_descriptors(df_train, smiles_col, prefix=cfg.QM_PREFIX)
            df_test = add_electronic_descriptors(df_test, smiles_col, prefix=cfg.QM_PREFIX)
            qm_computed = [c for c in df_train.columns if c.startswith(cfg.QM_PREFIX)]
            logger.info(f"  → {len(qm_computed)} electronic descriptors added")
        except Exception as e:
            logger.warning(f"  Electronic descriptor computation failed: {e}")
    else:
        logger.info("  Electronic descriptor computation skipped "
                     "(COMPUTE_ELECTRONIC=False or no SMILES)")

    # ── 7. 외부 QM 파일 merge (optional) ──
    if cfg.QM_FILE.exists():
        try:
            from utils.qm_descriptors import merge_external_qm
            logger.info(f"  Merging external QM: {cfg.QM_FILE.name}")
            df_train, qm_report_tr = merge_external_qm(
                df_train, cfg.QM_FILE, merge_key="No",
                qm_cols=None, prefix=cfg.EXT_QM_PREFIX)
            df_test, qm_report_te = merge_external_qm(
                df_test, cfg.QM_FILE, merge_key="No",
                qm_cols=None, prefix=cfg.EXT_QM_PREFIX)
            logger.info(f"  External QM merge: train={qm_report_tr.get('status')}, "
                        f"test={qm_report_te.get('status')}")
        except Exception as e:
            logger.warning(f"  External QM merge failed: {e}")
    else:
        logger.info("  External QM file not found — skip")

    # ── 7. 문자열 컬럼이 numeric block에 섞이지 않도록 차단 ──
    # scaffold raw string은 메타로만 유지
    for col in df_train.columns:
        if df_train[col].dtype == "object" and col not in cfg.META_COLS:
            # 고 카디널리티 문자열은 제거
            if df_train[col].nunique() > 50:
                logger.info(f"  Dropping high-cardinality string col: {col}")
                df_train.drop(columns=[col], inplace=True, errors="ignore")
                df_test.drop(columns=[col], inplace=True, errors="ignore")

    # ── 8. split source 태깅 ──
    df_train["split_source"] = "train"
    df_test["split_source"] = "test"

    # ── 9. 저장 ──
    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)
    logger.info(f"  Saved: {train_out.name} ({df_train.shape})")
    logger.info(f"  Saved: {test_out.name} ({df_test.shape})")

    # ── 10. Feature block audit ──
    col_class = classify_columns(df_train)
    feature_blocks = classify_feature_blocks(col_class["features"])
    block_summary = {block: len(cols) for block, cols in feature_blocks.items()}
    logger.info(f"  Feature blocks: {block_summary}")

    return True


def run_feature_build(force: bool = False):
    """전체 endpoint feature 빌드"""
    logger.info("=" * 60)
    logger.info("Step 2: Broad Feature Table Build")
    logger.info("=" * 60)

    results = {}
    for endpoint in cfg.ENDPOINTS:
        try:
            ok = build_broad_features(endpoint, force_rebuild=force)
            results[endpoint] = "ok" if ok else "failed"
        except Exception as e:
            logger.error(f"[{endpoint}] FAILED: {e}")
            results[endpoint] = f"error: {e}"

    # Feature audit summary
    audit_rows = []
    for endpoint in cfg.ENDPOINTS:
        train_path = cfg.BROADFP_DIR / f"{endpoint}_broadfp_train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path, nrows=0, low_memory=False)
            col_class = classify_columns(df)
            blocks = classify_feature_blocks(col_class["features"])
            row = {"endpoint": endpoint, "total_features": len(col_class["features"])}
            for block, cols in blocks.items():
                row[f"n_{block}"] = len(cols)
            audit_rows.append(row)

    if audit_rows:
        df_audit = pd.DataFrame(audit_rows)
        audit_path = cfg.SUMMARY_DIR / "feature_audit_summary.csv"
        df_audit.to_csv(audit_path, index=False)
        logger.info(f"\nFeature audit saved: {audit_path}")

    # Manifest
    manifest = {"endpoints": results}
    json_path = cfg.SUMMARY_DIR / "broad_feature_manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {json_path}")

    logger.info("\n" + "=" * 60)
    for ep, st in results.items():
        logger.info(f"  {ep}: {st}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    args = parser.parse_args()
    run_feature_build(force=args.force)
