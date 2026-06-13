"""
run_all.py -- 전체 파이프라인 마스터 실행기
==========================================
Step 1~8 (학습/평가) → Step 9 (외부 검증, 선택적)

사용법:
    python run_all.py                        # 기본 실행
    python run_all.py --full-study           # ablation, imbalance 포함
    python run_all.py --all                  # 전체 조합 실행

  # 외부 검증 (step9) 포함:
    python run_all.py --external-ames data\eu_ntp_ames.csv
    python run_all.py --external-ames data\eu_ntp_ames.csv --fn-augmentation --fn-rounds 3
    python run_all.py --external-ames data\eu_ntp_ames.csv --criterion f1 --thr-step 0.01
"""
import sys, subprocess, time, logging, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as cfg
from utils.progress import step_header, task_done

# ── 로그 디렉토리: RUNS_DIR 옆에 logs/ 생성 ──────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(stream=open(_sys.stdout.fileno(),
            mode='w', encoding='utf-8', buffering=1, closefd=False)),
        logging.FileHandler(LOG_DIR / "run_all.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger("run_all")


def run_step(script_name: str, args: list = None, description: str = "",
             step_num: int = 0, total_steps: int = 9):
    """subprocess로 단계 실행 -- 실패해도 다음 단계 진행"""
    script = ROOT / script_name
    cmd = [sys.executable, str(script)] + (args or [])

    step_header(step_num, total_steps, description or script_name)
    logger.info(f"CMD: {' '.join(cmd[:3])} ...")

    t0 = time.time()
    try:
        # stderr를 터미널에 직접 출력(tqdm 등 진행률 표시)
        # stdout은 logger로 캡처
        import io, threading

        # stderr: 터미널 직통(tqdm) + 별도 에러 파일 동시 기록
        _err_path = LOG_DIR / f"{script_name.replace('.py','')}_stderr.log"
        _err_file = open(_err_path, 'w', encoding='utf-8')
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',   # 자식 프로세스 stdout/stderr 인코딩과 일치시킴
            errors='replace',   # 디코드 실패 시 crash 대신 ? 대체 → _drain 생존
            bufsize=1,
        )

        stdout_lines = []
        def _drain(pipe):
            for line in pipe:
                stdout_lines.append(line.rstrip())
                logger.info(f"  | {line.rstrip()}")

        def _drain_err(pipe):
            for line in pipe:
                sys.stderr.write(line)   # 터미널 실시간 출력
                _err_file.write(line)     # 파일에도 저장
                _err_file.flush()
                if any(kw in line for kw in ['Error','error','Traceback','FAILED']):
                    logger.error(f'  | STDERR: {line.rstrip()}')
        t_out = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
        t_err = threading.Thread(target=_drain_err, args=(proc.stderr,), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=None)   # timeout 없음 -- 학습이 완료될 때까지 대기
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"  TIMEOUT after {time.time() - t0:.0f}s")
            return False
        t_out.join()
        t_err.join()
        _err_file.close()
        elapsed = time.time() - t0

        if proc.returncode != 0:
            logger.error(f"  FAILED (exit code {proc.returncode})")
            logger.error(f"  Stderr log: {_err_path}")
            # stderr 마지막 20줄 로그에 출력
            try:
                lines = _err_path.read_text(encoding="utf-8").strip().split("\n")
                for ln in lines[-20:]:
                    if ln.strip():
                        logger.error(f"  | {ln}")
            except Exception:
                pass
            return False
        else:
            task_done(description or script_name, elapsed)
            return True

    except subprocess.TimeoutExpired:
        logger.error(f"  TIMEOUT after {time.time() - t0:.0f}s")
        return False
    except Exception as e:
        logger.error(f"  EXCEPTION: {e}")
        return False


def get_latest_run_dir():
    """LATEST_TXT → RUNS_DIR 내 최신 폴더 순으로 run_dir 탐색"""
    # 1) LATEST_TXT
    if cfg.LATEST_TXT.exists():
        candidate = Path(cfg.LATEST_TXT.read_text().strip())
        if candidate.exists():
            return candidate
    # 2) RUNS_DIR 내 최신 폴더
    if cfg.RUNS_DIR.exists():
        subdirs = sorted(cfg.RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in subdirs:
            if d.is_dir():
                return d
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Genotox Pipeline -- Master Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 기존 옵션 ────────────────────────────────────────────────────
    parser.add_argument("--all", action="store_true",
                        help="전체 endpoint×model×strategy 조합 실행")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Broad feature table 강제 재생성")
    parser.add_argument("--skip-audit",    action="store_true", help="Step 1 스킵")
    parser.add_argument("--skip-build",    action="store_true", help="Step 2 스킵")
    parser.add_argument("--skip-ablation", action="store_true", help="Step 5 스킵")
    parser.add_argument("--skip-imbalance",action="store_true", help="Step 6 스킵")
    parser.add_argument("--full-study",    action="store_true",
                        help="ablation, imbalance 포함 전체 실행")

    # ── Step 9: 외부 검증 ────────────────────────────────────────────
    ext = parser.add_argument_group(
        "Step 9 -- External Validation",
        "외부 CSV를 하나 이상 지정하면 step9 자동 실행. 없으면 스킵."
    )
    ext.add_argument("--external-ames",    default=None, metavar="CSV")
    ext.add_argument("--external-invitro", default=None, metavar="CSV")
    ext.add_argument("--external-invivo",  default=None, metavar="CSV")
    ext.add_argument("--external-smiles-col", default="SMILES", metavar="COL")
    ext.add_argument("--external-label-col",  default="label",  metavar="COL")
    ext.add_argument("--split-comparison", action="store_true",
                     help="Random vs scaffold split 과대추정 정량화 (Mode C)")
    ext.add_argument("--fn-augmentation",  action="store_true",
                     help="FN 물질 반복 편입 실험 (Mode D)")
    ext.add_argument("--fn-rounds", type=int, default=3, metavar="N")
    ext.add_argument("--skip-literature", action="store_true",
                     help="문헌 비교 테이블(Mode A) 스킵")
    ext.add_argument("--ad-threshold", type=float, default=0.4, metavar="T")
    ext.add_argument("--n-bits",   type=int,   default=512,   metavar="N")
    ext.add_argument("--top-k",    type=int,   default=128,   metavar="K")

    thr = parser.add_argument_group(
        "Threshold Grid Search",
        "OOF probability 기반 최적 threshold 그리드서치."
    )
    thr.add_argument("--thr-min",   type=float, default=0.1,  metavar="F")
    thr.add_argument("--thr-max",   type=float, default=0.9,  metavar="F")
    thr.add_argument("--thr-step",  type=float, default=0.02, metavar="F",
                     help="탐색 간격 (0.02 → 40개 후보)")
    thr.add_argument("--criterion", type=str,   default="mcc",
                     choices=["mcc", "f1", "youden"],
                     help="threshold 선택 기준 (mcc: 논문 주지표, f1: sensitivity 우선)")

    # ── Step 10: Coverage Extension ──────────────────────────────────
    cov = parser.add_argument_group("Step 10 -- Coverage Extension & FN Reduction")
    cov.add_argument("--coverage", action="store_true",
                     help="Step 10 실행 (전략 A-J)")
    cov.add_argument("--strategy", nargs="+", default=["all"],
                     help="실행 전략 (A B D F I J 또는 all)")
    cov.add_argument("--fn-penalty", type=float, default=3.0,
                     help="Strategy D: FN penalty 배율")
    cov.add_argument("--n-clusters", type=int, default=200,
                     help="Strategy A: Butina cluster 수")

    args = parser.parse_args()

    start = datetime.now()
    logger.info(f"Pipeline started: {start.isoformat()}")
    logger.info(f"Project root: {ROOT}")
    logger.info(f"Data dir:     {cfg.DATA_DIR}")
    logger.info(f"Runs dir:     {cfg.RUNS_DIR}")

    results = {}

    # ── Step 1 ──────────────────────────────────────────────────────
    if not args.skip_audit:
        ok = run_step("step1_data_audit.py",
                      description="[1/9] Raw Data Audit & Schema Unification",
                      step_num=1)
        results["step1_audit"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 1 (audit)")
        results["step1_audit"] = "skipped"

    # ── Step 2 ──────────────────────────────────────────────────────
    if not args.skip_build:
        step2_args = ["--force"] if args.force_rebuild else []
        ok = run_step("step2_feature_build.py", args=step2_args,
                      description="[2/9] Broad Feature Table Build",
                      step_num=2)
        results["step2_build"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 2 (build)")
        results["step2_build"] = "skipped"

    # ── Step 3 ──────────────────────────────────────────────────────
    step3_args = ["--all"] if args.all else []
    ok = run_step("step3_train.py", args=step3_args,
                  description="[3/9] Leakage-free Training & Evaluation",
                      step_num=3)
    results["step3_train"] = "ok" if ok else "failed"

    # ── Step 4 ──────────────────────────────────────────────────────
    ok = run_step("step4_leakage_report.py",
                  description="[4/9] Leakage Control Report",
                      step_num=4)
    results["step4_leakage"] = "ok" if ok else "failed"

    # ── Step 5 ──────────────────────────────────────────────────────
    if args.full_study and not args.skip_ablation:
        ok = run_step("step5_ablation.py",
                      description="[5/9] Ablation / Robustness Study",
                      step_num=5)
        results["step5_ablation"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 5 (ablation) -- use --full-study to enable")
        results["step5_ablation"] = "skipped"

    # ── Step 6 ──────────────────────────────────────────────────────
    if args.full_study and not args.skip_imbalance:
        ok = run_step("step6_imbalance_study.py",
                      description="[6/9] Imbalance Strategy Comparison",
                      step_num=6)
        results["step6_imbalance"] = "ok" if ok else "failed"
    else:
        logger.info("Skipping Step 6 (imbalance) -- use --full-study to enable")
        results["step6_imbalance"] = "skipped"

    # ── Step 7 ──────────────────────────────────────────────────────
    ok = run_step("step7_dashboard.py",
                  description="[7/9] Summary Dashboard Generation",
                      step_num=7)
    results["step7_dashboard"] = "ok" if ok else "failed"

    # ── Step 8 ──────────────────────────────────────────────────────
    ok = run_step("step8_comprehensive_analysis.py",
                  description="[8/9] Comprehensive Analysis",
                      step_num=8)
    results["step8_analysis"] = "ok" if ok else "failed"

    # ── Step 10: Coverage Extension & FN Reduction ─────────────────
    if args.coverage:
        step10_run_dir = get_latest_run_dir()
        if step10_run_dir is None:
            logger.error("Step 10: run_dir를 찾을 수 없음")
            results["step10_coverage"] = "failed"
        else:
            step10_args = [
                "--run-dir", str(step10_run_dir),
                "--strategy", *args.strategy,
                "--fn-penalty", str(args.fn_penalty),
                "--n-clusters", str(args.n_clusters),
                "--n-bits", str(args.n_bits),
                "--top-k",  str(args.top_k),
            ]
            ok = run_step("step10_coverage_extension.py", args=step10_args,
                          description="[10] Coverage Extension & FN Reduction (A-J)",
                          step_num=10, total_steps=10)
            results["step10_coverage"] = "ok" if ok else "failed"
    else:
        logger.info(
            "Skipping Step 10 -- 화학공간 커버리지 확장 미실행\n"
            "  실행 예시:\n"
            "    python run_all.py --coverage\n"
            "    python run_all.py --coverage --strategy A B D --fn-penalty 5.0"
        )
        results["step10_coverage"] = "skipped"

    # ── Step 9: External Validation ─────────────────────────────────
    has_external = any([args.external_ames, args.external_invitro, args.external_invivo])
    run_step9 = has_external or args.split_comparison

    if run_step9:
        run_dir = get_latest_run_dir()
        if run_dir is None:
            logger.error("Step 9: run_dir를 찾을 수 없음 -- step3이 완료됐는지 확인")
            results["step9_external"] = "failed"
        else:
            logger.info(f"Step 9 run_dir: {run_dir}")

            step9_args = ["--run-dir", str(run_dir)]

            if args.external_ames:
                step9_args += ["--external-ames",    str(Path(args.external_ames).resolve())]
            if args.external_invitro:
                step9_args += ["--external-invitro", str(Path(args.external_invitro).resolve())]
            if args.external_invivo:
                step9_args += ["--external-invivo",  str(Path(args.external_invivo).resolve())]

            step9_args += ["--external-smiles-col", args.external_smiles_col]
            step9_args += ["--external-label-col",  args.external_label_col]
            step9_args += [
                "--thr-min",   str(args.thr_min),
                "--thr-max",   str(args.thr_max),
                "--thr-step",  str(args.thr_step),
                "--criterion", args.criterion,
                "--ad-threshold", str(args.ad_threshold),
                "--n-bits", str(args.n_bits),
                "--top-k",  str(args.top_k),
            ]
            if args.split_comparison: step9_args.append("--split-comparison")
            if args.fn_augmentation:  step9_args += ["--fn-augmentation", "--fn-rounds", str(args.fn_rounds)]
            if args.skip_literature:  step9_args.append("--skip-literature")

            ok = run_step("step9_external_benchmark.py", args=step9_args,
                          description="[9/9] External Validation & Benchmark",
                      step_num=9)
            results["step9_external"] = "ok" if ok else "failed"
    else:
        logger.info(
            "Skipping Step 9 -- 외부 검증셋 미지정\n"
            "  실행 예시:\n"
            "    python run_all.py --external-ames data\\eu_ntp_ames.csv\n"
            "    python run_all.py --external-ames data\\eu_ntp_ames.csv "
            "--fn-augmentation --fn-rounds 3 --criterion f1"
        )
        results["step9_external"] = "skipped"

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"\n{'#'*60}")
    logger.info(f"PIPELINE COMPLETE  ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    for step, status in results.items():
        symbol = "V" if status == "ok" else ("O" if status == "skipped" else "X")
        logger.info(f"  [{symbol}] {step}: {status}")
    logger.info(f"{'#'*60}")
    logger.info(f"Runs dir: {cfg.RUNS_DIR}")
    logger.info(f"Log:      {LOG_DIR / 'run_all.log'}")


if __name__ == "__main__":
    main()
