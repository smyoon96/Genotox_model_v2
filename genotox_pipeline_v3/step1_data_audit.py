"""
step1_data_audit.py -- Raw 데이터 감사 및 스키마 통합
====================================================
Prompt 1: raw CSV/ZIP/분석결과를 읽고 endpoint별 공통 스키마를 정리한다.
"""
import sys, json, logging
from pathlib import Path

import pandas as pd
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from utils.data_utils import (
    discover_split_files, ensure_fg_dir, find_fg_files,
    safe_read_csv, unify_smiles_col, clean_merge_duplicates,
    normalize_merge_key, classify_columns
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step1_data_audit.log", mode="w"),
    ]
)
logger = logging.getLogger("step1")


def audit_endpoint(endpoint: str) -> dict:
    """단일 endpoint 데이터 감사"""
    report = {"endpoint": endpoint, "status": "ok"}

    # 1) 파일 탐색
    train_path, test_path = discover_split_files(endpoint)
    if train_path is None:
        report["status"] = "MISSING"
        report["error"] = "train/test files not found"
        logger.error(f"[{endpoint}] {report['error']}")
        return report

    report["train_file"] = train_path.name
    report["test_file"] = test_path.name

    # 2) 읽기
    try:
        df_train = safe_read_csv(train_path)
        df_test = safe_read_csv(test_path)
    except Exception as e:
        report["status"] = "READ_ERROR"
        report["error"] = str(e)
        logger.error(f"[{endpoint}] Read error: {e}")
        return report

    # 3) 기본 통계
    for split_name, df in [("train", df_train), ("test", df_test)]:
        prefix = f"{split_name}_"
        report[f"{prefix}rows"] = len(df)
        report[f"{prefix}cols"] = len(df.columns)

        if "label" in df.columns:
            vc = df["label"].value_counts()
            n_pos = int(vc.get(1, 0))
            n_neg = int(vc.get(0, 0))
            report[f"{prefix}n_positive"] = n_pos
            report[f"{prefix}n_negative"] = n_neg
            report[f"{prefix}positive_rate"] = round(n_pos / len(df), 4) if len(df) > 0 else 0
        else:
            report[f"{prefix}label_col"] = "MISSING"

        # key 컬럼 결측
        for key in ["No", "label", "scaffold_group"]:
            if key in df.columns:
                missing = df[key].isna().sum()
                report[f"{prefix}{key}_missing"] = int(missing)
                report[f"{prefix}{key}_missing_pct"] = round(missing / len(df) * 100, 2)

    # 4) SMILES 컬럼 확인
    smiles_found = []
    for alias in ["SMILES", "smiles", "canonical_smiles", "standardized_smiles",
                   "Canonical_SMILES", "Standardized_SMILES"]:
        if alias in df_train.columns:
            smiles_found.append(alias)
    report["smiles_columns"] = smiles_found

    # 5) 컬럼 분류
    col_class = classify_columns(df_train)
    report["n_meta_cols"] = len(col_class["meta"])
    report["n_feature_candidates"] = len(col_class["features"])
    report["n_excluded_cols"] = len(col_class["excluded"])
    report["meta_cols"] = col_class["meta"]
    report["excluded_cols"] = col_class["excluded"]

    # 6) scaffold_group 확인
    if "scaffold_group" in df_train.columns:
        report["scaffold_groups_unique"] = int(df_train["scaffold_group"].nunique())
        report["scaffold_group_type_values"] = (
            df_train["scaffold_group_type"].unique().tolist()
            if "scaffold_group_type" in df_train.columns else []
        )
    else:
        report["scaffold_group"] = "MISSING"

    logger.info(f"[{endpoint}] train={report.get('train_rows', '?')} rows, "
                f"test={report.get('test_rows', '?')} rows")
    return report


def run_audit():
    """전체 감사 실행"""
    logger.info("=" * 60)
    logger.info("Step 1: Raw Data Audit")
    logger.info(f"Data dir: {cfg.DATA_DIR}")
    logger.info("=" * 60)

    # data dir 존재 확인
    if not cfg.DATA_DIR.exists():
        logger.error(f"Data directory not found: {cfg.DATA_DIR}")
        logger.info("Available files in project root:")
        if cfg.PROJECT_ROOT.exists():
            for f in cfg.PROJECT_ROOT.iterdir():
                logger.info(f"  {f.name}")
        return

    # 사용 가능한 파일 목록
    all_files = list(cfg.DATA_DIR.glob("*"))
    logger.info(f"Files in data dir: {len(all_files)}")
    for f in sorted(all_files)[:30]:
        logger.info(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    # fg_descriptor_analysis 확인
    fg_dir = ensure_fg_dir()
    if fg_dir:
        logger.info(f"FG analysis dir: {fg_dir}")
        for ep in cfg.ENDPOINTS:
            fg_files = find_fg_files(fg_dir, ep)
            logger.info(f"  [{ep}] canonical={fg_files['canonical']}, views={fg_files['views']}")

    # endpoint별 감사
    reports = []
    for endpoint in cfg.ENDPOINTS:
        logger.info(f"\n--- Auditing: {endpoint} ---")
        report = audit_endpoint(endpoint)
        reports.append(report)

    # 결과 저장
    df_report = pd.DataFrame(reports)
    csv_path = cfg.SUMMARY_DIR / "data_audit_summary.csv"
    df_report.to_csv(csv_path, index=False)
    logger.info(f"\nAudit summary saved: {csv_path}")

    # JSON manifest
    manifest = {
        "data_dir": str(cfg.DATA_DIR),
        "endpoints": {r["endpoint"]: r for r in reports},
    }
    json_path = cfg.SUMMARY_DIR / "schema_manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Schema manifest saved: {json_path}")

    # 요약 출력
    logger.info("\n" + "=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)
    for r in reports:
        status = r["status"]
        ep = r["endpoint"]
        if status == "ok":
            logger.info(f"  {ep}: train={r.get('train_rows', '?')}, "
                        f"test={r.get('test_rows', '?')}, "
                        f"pos_rate={r.get('train_positive_rate', '?')}")
        else:
            logger.warning(f"  {ep}: {status} -- {r.get('error', '')}")


if __name__ == "__main__":
    run_audit()
