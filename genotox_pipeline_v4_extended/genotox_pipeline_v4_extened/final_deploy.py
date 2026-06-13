"""
final_deploy.py — 최종 배포 아티팩트 생성
===========================================
deploy_pipeline.py 실행 후, conformal calibration + AD reference를 추가 저장.

Usage:
    python final_deploy.py --deploy-dir output/deploy --data-dir data --output-dir models
"""

import sys, json, logging, argparse, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
lg = logging.getLogger("final")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from config import GLOBAL_SEED
from deploy_pipeline import (
    load_and_clean, apply_scenario_deploy, build_features_v4, make_model,
)
from step4_feature_extraction import extract_fingerprint_features

DEPLOY_ENDPOINTS = ["ames", "invitro_sampling", "invivo_sampling"]
CONFORMAL_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def run(deploy_dir: Path, data_dir: Path, output_dir: Path):
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    lg.info("=" * 60)
    lg.info("  FINAL DEPLOYMENT — artifact generation")
    lg.info("=" * 60)

    summary = {}

    for ep in DEPLOY_ENDPOINTS:
        ep_src = deploy_dir / ep
        meta_path = ep_src / "metadata.json"
        if not meta_path.exists():
            lg.warning(f"  [{ep}] Not found in deploy_dir, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        lg.info(f"\n{'─'*50}")
        lg.info(f"  [{ep}] {meta['model_type']} / {meta['scenario']} / {meta['feature_mode']}")

        ep_out = output_dir / ep
        ep_out.mkdir(parents=True, exist_ok=True)

        # ── 데이터 + Feature ──
        df, flags = load_and_clean(data_dir, ep)
        df = apply_scenario_deploy(df, flags, meta["scenario"])

        with open(ep_src / "feature_columns.json") as f:
            fcols = json.load(f)

        feat = build_features_v4(df, ep, meta["feature_mode"], meta["scenario"])
        feat = feat.reindex(columns=fcols, fill_value=0)
        X = feat.values.astype(np.float32)
        y = df["label"].values.astype(int)

        # ── 1. Model copy ──
        for fname in ["model.joblib", "metadata.json", "feature_columns.json",
                       "shap_values.csv", "shap_summary.png", "shap_bar.png",
                       "threshold_curve.png"]:
            src = ep_src / fname
            if src.exists():
                import shutil
                shutil.copy2(src, ep_out / fname)

        # ── 2. Conformal calibration ──
        lg.info(f"  Conformal calibration...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        oof_p0 = np.zeros(len(y))
        oof_p1 = np.zeros(len(y))

        for tr_idx, val_idx in skf.split(X, y):
            spw = (y[tr_idx]==0).sum() / max((y[tr_idx]==1).sum(), 1)
            mdl = make_model(meta["model_type"], spw)
            mdl.fit(X[tr_idx], y[tr_idx])
            proba = mdl.predict_proba(X[val_idx])
            oof_p0[val_idx] = proba[:, 0]
            oof_p1[val_idx] = proba[:, 1]

        nc_scores = np.where(y == 1, 1 - oof_p1, 1 - oof_p0)

        conformal = {"alphas": {}}
        for alpha in CONFORMAL_ALPHAS:
            q = float(np.quantile(nc_scores, 1 - alpha))
            # Evaluate
            set_0 = (1 - oof_p0) <= q
            set_1 = (1 - oof_p1) <= q
            uncertain = set_0 & set_1
            certain = ~uncertain & ~(~set_0 & ~set_1)
            coverage = certain.sum() / len(y)

            if certain.sum() > 10:
                cert_pred = np.where(set_1[certain] & ~set_0[certain], 1, 0)
                mcc = matthews_corrcoef(y[certain], cert_pred)
            else:
                mcc = 0.0

            conformal["alphas"][str(alpha)] = {
                "quantile": round(q, 6),
                "coverage": round(float(coverage), 4),
                "mcc_certain": round(float(mcc), 4),
            }
            lg.info(f"    α={alpha:.2f}: q={q:.4f} coverage={coverage:.0%} MCC={mcc:.3f}")

        conformal["default_alpha"] = 0.05
        conformal["nc_scores_percentiles"] = {
            f"p{p}": round(float(np.percentile(nc_scores, p)), 6)
            for p in [5, 10, 25, 50, 75, 90, 95]
        }
        with open(ep_out / "conformal.json", "w") as f:
            json.dump(conformal, f, indent=2)

        # ── 3. AD reference fingerprints ──
        lg.info(f"  AD reference fingerprints...")
        fp = extract_fingerprint_features(df, ep, n_bits=1024)
        fp_arr = fp.values.astype(np.float32)
        np.savez_compressed(ep_out / "ad_reference.npz", fingerprints=fp_arr)
        lg.info(f"    Saved: {fp_arr.shape} ({fp_arr.nbytes // 1024} KB)")

        # ── 4. Performance card ──
        best_alpha = 0.05
        best_conf = conformal["alphas"]["0.05"]
        perf = {
            "endpoint": ep,
            "model_type": meta["model_type"],
            "scenario": meta["scenario"],
            "feature_mode": meta["feature_mode"],
            "n_train": len(X),
            "n_pos": int(y.sum()),
            "threshold": meta["optimal_threshold"],
            "overall": {
                "mcc": meta["mcc_at_threshold"],
                "sensitivity": meta["sensitivity"],
                "specificity": meta["specificity"],
            },
            "conformal_alpha005": {
                "mcc": best_conf["mcc_certain"],
                "coverage": best_conf["coverage"],
            },
            "cv_mcc": meta["cv_mcc"],
        }
        with open(ep_out / "performance.json", "w") as f:
            json.dump(perf, f, indent=2)

        summary[ep] = perf

    # ── Global summary ──
    with open(output_dir / "models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # ── requirements.txt ──
    with open(output_dir / "requirements.txt", "w") as f:
        f.write("numpy>=1.24\npandas>=2.0\nscikit-learn>=1.3\n"
                "xgboost>=2.0\nlightgbm>=4.0\nrdkit>=2023.03\njoblib>=1.3\n")

    lg.info(f"\n{'='*60}")
    lg.info(f"  COMPLETE ({time.time()-t0:.0f}s)")
    lg.info(f"  Output: {output_dir}")
    lg.info(f"{'='*60}")
    for ep, p in summary.items():
        c = p["conformal_alpha005"]
        lg.info(f"  [{ep}] MCC={c['mcc']:.3f} (coverage={c['coverage']:.0%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deploy-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="models")
    a = p.parse_args()
    run(Path(a.deploy_dir), Path(a.data_dir), Path(a.output_dir))
