"""
run_all.py — 전체 파이프라인 마스터 실행기
==========================================
Step 1 (감사) → Step 2 (Feature 빌드) → Step 3 (학습/평가)

사용법:
    python run_all.py                  # shortlist만 실행
    python run_all.py --all            # 전체 조합 실행
    python run_all.py --force-rebuild  # broadfp 강제 재생성
    python run_all.py --skip-audit     # 감사 스킵
"""
import sys, subprocess, time, logging, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_DIR / "run_all.log", mode="w"),
    ]
)
logger = logging.getLogger("run_all")


def run_step(script_name: str, args: list = None, description: str = ""):
    """subprocess로 단계 실행 — 실패해도 다음 단계 진행"""
    script = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script)] + (args or [])

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description or script_name}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=3600 * 2,  # 2시간 제한
        )
        elapsed = time.time() - t0

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-20:]:
                logger.info(f"  | {line}")
        if result.returncode != 0:
            logger.error(f"  FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    logger.error(f"  | {line}")
            return False
        else:
            logger.info(f"  DONE ({elapsed:.1f}s)")
            return True

    except subprocess.TimeoutExpired:
        logger.error(f"  TIMEOUT after {time.time() - t0:.0f}s")
        return False
    except Exception as e:
        logger.error(f"  EXCEPTION: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Genotox Pipeline — Master Runner")
    parser.add_argument("--all", action="store_true",
                        help="Run all endpoint×model×strategy combinations")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild broad feature tables")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip data audit step")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip feature build step")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation/robustness study")
    parser.add_argument("--skip-imbalance", action="store_true",
                        help="Skip imbalance strategy comparison")
    parser.add_argument("--full-study", action="store_true",
                        help="Run ALL steps including ablation, imbalance, dashboard")
    args = parser.parse_args()

    start = datetime.now()
    logger.info(f"Pipeline started: {start.isoformat()}")
    logger.info(f"Project root: {cfg.PROJECT_ROOT}")
    logger.info(f"Data dir: {cfg.DATA_DIR}")

    # run config 저장
    cfg.save_run_config(cfg.OUTPUT_DIR, extra={
        "run_all_args": vars(args),
        "start_time": start.isoformat(),
    })

    results = {}

    # ── Step 1: Data Audit ──
    if not args.skip_audit:
        ok = run_step("step1_data_audit.py",
                       description="[1/7] Raw Data Audit & Schema Unification")
        results["step1_audit"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 1 (audit)")
        results["step1_audit"] = "skipped"

    # ── Step 2: Feature Build ──
    if not args.skip_build:
        step2_args = ["--force"] if args.force_rebuild else []
        ok = run_step("step2_feature_build.py", args=step2_args,
                       description="[2/7] Broad Feature Table Build")
        results["step2_build"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 2 (build)")
        results["step2_build"] = "skipped"

    # ── Step 3: Main Training (shortlist or all) ──
    step3_args = ["--all"] if args.all else []
    ok = run_step("step3_train.py", args=step3_args,
                   description="[3/7] Leakage-free Training & Evaluation")
    results["step3_train"] = "ok" if ok else "failed"

    # ── Step 4: Leakage Control Report ──
    ok = run_step("step4_leakage_report.py",
                   description="[4/7] Leakage Control Report (Reviewer Defense)")
    results["step4_leakage"] = "ok" if ok else "failed"

    # ── Step 5: Ablation Study (optional) ──
    if args.full_study and not args.skip_ablation:
        ok = run_step("step5_ablation.py",
                       description="[5/7] Ablation / Robustness Study")
        results["step5_ablation"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 5 (ablation) — use --full-study to enable")
        results["step5_ablation"] = "skipped"

    # ── Step 6: Imbalance Study (optional) ──
    if args.full_study and not args.skip_imbalance:
        ok = run_step("step6_imbalance_study.py",
                       description="[6/7] Imbalance Strategy Comparison")
        results["step6_imbalance"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 6 (imbalance) — use --full-study to enable")
        results["step6_imbalance"] = "skipped"

    # ── Step 7: Summary Dashboard ──
    ok = run_step("step7_dashboard.py",
                   description="[7/8] Summary Dashboard Generation")
    results["step7_dashboard"] = "ok" if ok else "failed"

    # ── Step 8: Comprehensive Analysis & Visualization ──
    ok = run_step("step8_comprehensive_analysis.py",
                   description="[8/8] Comprehensive Analysis (CM, Chemical Space, QM, Feature)")
    results["step8_analysis"] = "ok" if ok else "failed"

    # ── Final Summary ──
    elapsed_total = (datetime.now() - start).total_seconds()
    logger.info(f"\n{'#'*60}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    for step, status in results.items():
        symbol = "✓" if status == "ok" else ("⊘" if status == "skipped" else "✗")
        logger.info(f"  {symbol} {step}: {status}")
    logger.info(f"{'#'*60}")

    # 최종 결과 파일 위치 안내
    logger.info(f"\n{'='*60}")
    logger.info(f"OUTPUT DIRECTORY: {cfg.OUTPUT_DIR}")
    logger.info(f"{'='*60}")
    logger.info(f"  Core Results:")
    logger.info(f"    summary_metrics.csv:    {cfg.SUMMARY_DIR / 'summary_metrics.csv'}")
    logger.info(f"    best_by_endpoint.csv:   {cfg.SUMMARY_DIR / 'best_by_endpoint.csv'}")
    logger.info(f"  Reviewer Defense:")
    logger.info(f"    leakage_control_report: {cfg.SUMMARY_DIR / 'leakage_report'}")
    if results.get("step5_ablation") == "ok":
        logger.info(f"  Paper Figures:")
        logger.info(f"    ablation:               {cfg.SUMMARY_DIR / 'ablation'}")
    if results.get("step6_imbalance") == "ok":
        logger.info(f"    imbalance:              {cfg.SUMMARY_DIR / 'imbalance'}")
    logger.info(f"  Dashboard:")
    logger.info(f"    dashboard:              {cfg.SUMMARY_DIR / 'dashboard'}")
    logger.info(f"  Comprehensive Analysis:")
    logger.info(f"    analysis:               {cfg.OUTPUT_DIR / 'analysis'}")
    logger.info(f"  Artifacts:                {cfg.ARTIFACT_DIR}")
    logger.info(f"  Logs:                     {cfg.LOG_DIR}")


if __name__ == "__main__":
    main()
