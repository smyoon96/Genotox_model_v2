"""
step3_train.py -- Leakage-free Training Pipeline
=================================================
genotox_pipeline.run_pipeline()의 래퍼.

* Reviewer 방어 핵심 구조 (genotox_pipeline.py 참조) *
  1. Scaffold-aware fixed split (train/test 완전 분리)
  2. FP bit selection → CV fold 내부에서만 (leakage 차단)
  3. Scenario selection → train CV로 결정 (test 미참조)
  4. Hyperparameter tuning: inner-CV score != outer test
  5. OOF threshold grid search (MCC 최대화, 0.1~0.9 step 0.02)
  6. Locked test set → 최종 1회 평가
  7. Bootstrap CI / AD / calibration / SHAP 자동 계산

사용:
    python step3_train.py
    python step3_train.py --data-dir data/ --tag v12
"""
import sys, logging, argparse, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as cfg

# joblib memmapping 임시 폴더를 프로젝트 내 tmp/로 고정
# (ESTsoft CreatorTemp 경로 사용 시 발생하는 FileNotFoundError 방지)
_TMP_DIR = ROOT / "tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_TMP_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "step3_train.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger("step3")
# ↑ genotox_pipeline.setup_log()은 StreamHandler만 교체하고
#   FileHandler(step3_train.log)는 보존하므로 별도 guard 불필요


def main():
    parser = argparse.ArgumentParser(
        description="Genotox QSAR Training & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None,
                        help="데이터 디렉토리 (기본: config.DATA_DIR)")
    parser.add_argument("--tag", default="v12",
                        help="실행 태그 (run 디렉토리 이름에 포함)")
    parser.add_argument("--all", action="store_true",
                        help="shortlist 외 전체 조합 실행")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else cfg.DATA_DIR
    logger.info(f"Data dir : {data_dir}")
    logger.info(f"Runs dir : {cfg.RUNS_DIR}")
    logger.info(f"Tag      : {args.tag}")

    from genotox_pipeline import run_pipeline
    result = run_pipeline(data_dir=data_dir, tag=args.tag)

    run_dir = result["run_dir"]
    checks  = result.get("checks", {})
    n_pass  = sum(checks.values())
    n_total = len(checks)

    logger.info(f"\nRun complete: {run_dir}")
    logger.info(f"Checklist  : {n_pass}/{n_total} passed")
    if n_pass < n_total:
        logger.warning(f"Failed: {[k for k,v in checks.items() if not v]}")

    return run_dir


if __name__ == "__main__":
    main()
