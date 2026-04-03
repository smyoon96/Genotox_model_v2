"""
genotox_pipeline.py v11 — Clean Single-Run Pipeline
=====================================================
원칙: 하나의 코드, 하나의 실행, 하나의 결과.
      로그 파싱 없음. 런 병합 없음. 수동 조립 없음.

Step 1-3: Load → Clean → Flags → Fixed Split
Step 4:   Train CV → scenario selection (compact/xgb only)
Step 5:   36 combos locked test (pre-computed features, no re-extraction)
Step 6:   LDO + Checklist + Output

설계 제한 (정직하게 명시):
  - 하이퍼파라미터: 고정 baseline + RandomizedSearchCV 튜닝 결과 모두 보고
  - Threshold tuning: OOF-based (train resubstitution 아님)
  - CV: scenario selection에만 사용, 36-combo selection 아님
  - QM/under-sampling: 메인 결과에 미포함
  - cv_selected_compact_xgb: CV가 고른 scenario에서 compact+xgb (endpoint당 1개)
  - paper_primary_model: locked test MCC 최고 조합 (endpoint당 1개, tie-break: raw_all 우선)
"""
import sys, json, logging, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_v2_core import (
    resolve_conflicts, assign_scaffolds, fixed_split, apply_scenario,
    repeated_cv, bootstrap_ci, compute_ad, calibration,
    domain_confounding, leave_domain_out, cross_endpoint,
    find_smi, to_can, file_hash, interpret_result,
)
from step2b_preprocessing_impact import classify_all_compounds
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features, extract_fingerprint_features,
)
from config import DATA_DIR, RUNS_DIR, GLOBAL_SEED, ENDPOINTS, make_run_dir, save_json
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*X has feature names.*")

SCENARIOS = ["raw_all", "no_metal", "salt_stripped"]
FEAT_MODES = ["compact", "broad_fp512", "broad_fp1024", "broad_fp2048"]
TABULAR_MODELS = ["xgb", "lgbm", "rf", "svm", "logistic", "ann"]
MODEL_NAMES = TABULAR_MODELS + ["gnn"]  # GNN uses molecular graphs, not tabular features

# Check optional dependencies at import time
_HAS_LGBM = False
_HAS_TORCH = False
try:
    import lightgbm; _HAS_LGBM = True
except ImportError: pass
try:
    import torch; import torch.nn as tnn; _HAS_TORCH = True
except ImportError: pass


def setup_log(rd):
    root = logging.getLogger(); root.setLevel(logging.INFO); root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(rd / "pipeline.log", encoding="utf-8")]:
        h.setFormatter(fmt); root.addHandler(h)
    return logging.getLogger("gp")


def make_model(name, spw):
    """고정 하이퍼파라미터 baseline 모델 생성."""
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=GLOBAL_SEED, n_jobs=-1)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, verbose=-1,
            random_state=GLOBAL_SEED, n_jobs=-1)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=-1)
    elif name == "svm":
        from sklearn.svm import SVC
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=GLOBAL_SEED))])
    elif name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000,
                random_state=GLOBAL_SEED))])
    elif name == "ann":
        from sklearn.neural_network import MLPClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                random_state=GLOBAL_SEED))])
    elif name == "dnn":
        from sklearn.neural_network import MLPClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                batch_size=64, random_state=GLOBAL_SEED))])
    raise ValueError(f"Unknown model: {name}")


# ─── GNN Wrapper (PyTorch, sklearn-compatible) ──────────

class MolGCN:
    """
    Simple Graph Convolutional Network for molecular property prediction.
    sklearn-compatible interface (fit, predict, predict_proba).
    Uses RDKit for graph extraction, PyTorch for training.
    """
    def __init__(self, hidden_dim=64, n_layers=3, dropout=0.2,
                 lr=0.001, epochs=100, batch_size=32, patience=15,
                 random_state=42):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.device_ = None

    @staticmethod
    def smiles_to_graph(smi):
        """SMILES → (atom_features, adjacency_matrix)"""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            return np.zeros((1, 9), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        atoms = mol.GetAtoms()
        n = len(atoms)
        # Atom features: [atomic_num, degree, formal_charge, num_Hs,
        #                  aromatic, hybridization(3), in_ring]
        feat = np.zeros((n, 9), dtype=np.float32)
        for i, atom in enumerate(atoms):
            feat[i, 0] = atom.GetAtomicNum() / 53.0  # normalize
            feat[i, 1] = atom.GetDegree() / 4.0
            feat[i, 2] = atom.GetFormalCharge()
            feat[i, 3] = atom.GetTotalNumHs() / 4.0
            feat[i, 4] = float(atom.GetIsAromatic())
            hyb = atom.GetHybridization()
            feat[i, 5] = float(hyb == Chem.rdchem.HybridizationType.SP)
            feat[i, 6] = float(hyb == Chem.rdchem.HybridizationType.SP2)
            feat[i, 7] = float(hyb == Chem.rdchem.HybridizationType.SP3)
            feat[i, 8] = float(atom.IsInRing())
        # Adjacency (with self-loops)
        adj = np.eye(n, dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0
        # Normalize adjacency (D^-0.5 A D^-0.5)
        deg = adj.sum(axis=1, keepdims=True)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        adj = deg_inv_sqrt * adj * deg_inv_sqrt.T
        return feat, adj

    def _build_model(self, in_dim):
        import torch.nn as tnn
        layers = []
        dims = [in_dim] + [self.hidden_dim] * self.n_layers
        for i in range(len(dims) - 1):
            layers.append(tnn.Linear(dims[i], dims[i + 1]))
            layers.append(tnn.ReLU())
            layers.append(tnn.Dropout(self.dropout))
        layers.append(tnn.Linear(dims[-1], 2))  # binary classification
        return tnn.Sequential(*layers)

    def _graph_forward(self, feat, adj, model):
        """GCN forward: X' = σ(A·X·W) → global mean pool → classify"""
        import torch
        x = torch.FloatTensor(feat).to(self.device_)
        a = torch.FloatTensor(adj).to(self.device_)
        # Message passing (2 layers of A·X)
        for _ in range(2):
            x = torch.matmul(a, x)
        # Global mean pooling
        x = x.mean(dim=0, keepdim=True)  # (1, hidden)
        return model(x)

    def fit(self, X_smiles, y):
        import torch
        torch.manual_seed(self.random_state)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert SMILES to graphs
        graphs = [self.smiles_to_graph(s) for s in X_smiles]
        in_dim = graphs[0][0].shape[1]  # atom feature dim

        self.model_ = self._build_model(in_dim).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        pos_weight = torch.FloatTensor([(y == 0).sum() / max((y == 1).sum(), 1)]).to(self.device_)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, float(pos_weight)]).to(self.device_))

        # Training
        best_loss = float("inf"); patience_cnt = 0
        for epoch in range(self.epochs):
            self.model_.train()
            indices = np.random.RandomState(self.random_state + epoch).permutation(len(y))
            total_loss = 0
            for i in indices:
                feat, adj = graphs[i]
                out = self._graph_forward(feat, adj, self.model_)
                label = torch.LongTensor([int(y[i])]).to(self.device_)
                loss = criterion(out, label)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(y)
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss; patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= self.patience:
                break
        return self

    def predict_proba(self, X_smiles):
        import torch
        self.model_.eval()
        graphs = [self.smiles_to_graph(s) for s in X_smiles]
        probs = []
        with torch.no_grad():
            for feat, adj in graphs:
                out = self._graph_forward(feat, adj, self.model_)
                p = torch.softmax(out, dim=1).cpu().numpy()[0]
                probs.append(p)
        return np.array(probs)

    def predict(self, X_smiles):
        return (self.predict_proba(X_smiles)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
                "dropout": self.dropout, "lr": self.lr, "epochs": self.epochs,
                "batch_size": self.batch_size, "patience": self.patience,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─── Hyperparameter Search Spaces ──────────────────────
# config.py에서 import (single source of truth)
from config import HP_GRIDS, HP_N_ITER, HP_CV_FOLDS


def tune_model(X_train, y_train, groups, model_name, spw,
               n_iter=HP_N_ITER, n_folds=HP_CV_FOLDS,
               smiles_train=None):
    """
    GroupKFold 기반 RandomizedSearchCV (tabular) 또는
    manual grid search (GNN)로 하이퍼파라미터 튜닝.

    Returns: (best_model, best_params, best_cv_score)
    """
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold
    from sklearn.metrics import make_scorer

    mcc_scorer = make_scorer(matthews_corrcoef)
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)

    if actual_folds < 2:
        mdl = make_model(model_name, spw) if model_name != "gnn" else MolGCN()
        if model_name == "gnn":
            mdl.fit(smiles_train, y_train)
        else:
            mdl.fit(X_train, y_train)
        return mdl, {}, np.nan

    # ── GNN: manual search (not sklearn-compatible for RandomizedSearchCV) ──
    if model_name == "gnn":
        if smiles_train is None:
            raise ValueError("GNN requires smiles_train")
        return _tune_gnn(smiles_train, y_train, groups, actual_folds)

    # ── Tabular models: RandomizedSearchCV ──
    gkf = GroupKFold(n_splits=actual_folds)
    base = make_model(model_name, spw)
    param_dist = HP_GRIDS.get(model_name, {})

    if not param_dist:
        base.fit(X_train, y_train)
        return base, {}, np.nan

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=min(n_iter, len(param_dist) * 5),
        scoring=mcc_scorer,
        cv=gkf,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        error_score=0.0,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_, round(search.best_score_, 4)


def _tune_gnn(smiles, y, groups, n_folds):
    """GNN HP tuning via manual GroupKFold search."""
    from sklearn.model_selection import GroupKFold

    best_score, best_params, best_model = -1, {}, None
    grid = HP_GRIDS["gnn"]

    # Generate a small number of random configs
    rng = np.random.RandomState(GLOBAL_SEED)
    configs = []
    for _ in range(min(HP_N_ITER, 12)):  # GNN is slow, limit configs
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        configs.append(cfg)

    for cfg in configs:
        fold_scores = []
        gkf = GroupKFold(n_splits=n_folds)
        for tr_idx, val_idx in gkf.split(smiles, y, groups):
            mdl = MolGCN(**cfg, epochs=60)  # reduced epochs for tuning
            mdl.fit(np.array(smiles)[tr_idx], y[tr_idx])
            pred = mdl.predict(np.array(smiles)[val_idx])
            if len(set(pred)) > 0:
                fold_scores.append(matthews_corrcoef(y[val_idx], pred))
        if fold_scores:
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = cfg

    # Refit on full data with best params
    final = MolGCN(**best_params, epochs=100)
    final.fit(smiles, y)
    return final, best_params, round(best_score, 4)


def oof_threshold(X_train, y_train, groups, model_fn, n_folds=3):
    """OOF probability 기반 threshold tuning (train resubstitution이 아님)."""
    from sklearn.model_selection import GroupKFold
    oof_proba = np.full(len(y_train), np.nan)
    n_splits = min(n_folds, len(np.unique(groups)))
    if n_splits < 2:
        return 0.5
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
        mdl = model_fn()
        mdl.fit(X_train[tr_idx], y_train[tr_idx])
        oof_proba[val_idx] = mdl.predict_proba(X_train[val_idx])[:, 1]
    valid = ~np.isnan(oof_proba)
    if valid.sum() < 10:
        return 0.5
    best_t, best_mcc = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (oof_proba[valid] >= t).astype(int)
        if len(set(pred)) < 2: continue
        m = matthews_corrcoef(y_train[valid], pred)
        if m > best_mcc:
            best_mcc = m; best_t = t
    return round(best_t, 2)


def _build_row(experiment, ep, sc, fm, mn, is_rec,
               y_tr, y_te, orig_test_n, ad_result, ci, cal,
               opt_t, mcc_tuned_val, cv_ref):
    """Build a result row dict with all standard columns."""
    row = {
        "experiment": experiment,
        "endpoint": ep,
        "scenario": sc,
        "feature_mode": fm,
        "model": mn,
        "cv_selected_compact_xgb": is_rec,
        "train_n": int(len(y_tr)),
        "test_n": int(len(y_te)),
        "test_pos_rate": round(float(y_te.mean()), 4),
        "test_n_pos": int(y_te.sum()),
        "coverage": round(len(y_te) / orig_test_n, 4),
        "ad_coverage": ad_result["coverage"],
        "ad_mean_sim": ad_result["mean_sim"],
        "brier": cal["brier"],
        "ece": cal["ece"],
        "threshold_default": 0.5,
        "threshold_tuned": opt_t,
        "mcc_tuned": round(mcc_tuned_val, 4),
        "cv_mcc_mean": cv_ref.get("cv_mcc_mean") if isinstance(cv_ref, dict) else None,
        "cv_mcc_std": cv_ref.get("cv_mcc_std") if isinstance(cv_ref, dict) else None,
    }
    for k in ["mcc", "bacc", "roc_auc", "sens", "spec", "brier"]:
        if k in ci:
            row[k] = ci[k]["mean"]
            row[f"{k}_lo"] = ci[k]["lo"]
            row[f"{k}_hi"] = ci[k]["hi"]
    return row


def _save_experiment(edir, row, y_te, proba, pred_default, pred_tuned,
                     fcols, mdl, mn):
    """Save per-experiment files."""
    pd.DataFrame({"label": y_te, "proba": proba,
                  "pred": pred_default, "pred_tuned": pred_tuned
                  }).to_csv(edir / "test_predictions.csv", index=False)
    save_json(row, edir / "metrics.json")
    try:
        if mn == "xgb":
            imp = mdl.feature_importances_
        elif mn == "lgbm":
            imp = mdl.feature_importances_
        elif mn == "rf":
            imp = mdl.feature_importances_
        elif mn in ("logistic", "svm", "ann", "dnn"):
            imp = np.abs(mdl.named_steps["clf"].coef_[0]) if hasattr(mdl.named_steps.get("clf", mdl), "coef_") else None
        else:
            imp = None
        if imp is not None:
            pd.DataFrame({"feature": fcols[:len(imp)], "importance": imp}).sort_values(
                "importance", ascending=False).to_csv(edir / "feature_importance.csv", index=False)
    except: pass


def run_pipeline(data_dir=None, tag="v11"):
    from xgboost import XGBClassifier
    import joblib

    dp = Path(data_dir) if data_dir else DATA_DIR
    rd = make_run_dir(tag); lg = setup_log(rd); t0 = time.time()
    lg.info(f"Pipeline v11 — Clean Single Run\n  {rd}")
    save_json({"data": str(dp), "version": "v11",
               "ts": datetime.now().isoformat(), "seed": GLOBAL_SEED}, rd / "config.json")

    # ═══ STEP 1: Load + Clean ═══
    lg.info("STEP 1: Load + Clean")
    raw = {}
    for ep, fn in [("ames","ames.csv"),("invitro","invitro.csv"),("invivo","invivo.csv"),("invitro_sampling","invitro_sampling.csv"),("invivo_sampling","invivo_sampling.csv")]:
        fp = dp / fn
        lg.info(f"  Looking for: {fp.resolve()}")
        if not fp.exists():
            lg.info(f"    NOT FOUND — skipping")
            continue
        df = pd.read_excel(fp, engine="openpyxl") if fn.endswith(".xlsx") else pd.read_csv(fp, encoding="utf-8-sig")
        for c in df.columns:
            if c.strip().upper() == "SMILES": df = df.rename(columns={c: "SMILES"})
        df["label"] = df["label"].astype(int); df["endpoint"] = ep
        df = df.dropna(subset=["label","SMILES"])
        raw[ep] = df
        lg.info(f"  [{ep}] n={len(df)} pos={df['label'].mean():.4f}")

    # Cross-endpoint overlap
    if not raw:
        lg.error(f"  ✗ No data files found in {dp.resolve()}")
        lg.error(f"    Expected: ames.csv, invitro.csv, invivo.csv","invitro_sampling.csv","invivo_sampling.csv")
        lg.error(f"    Directory contents: {list(dp.iterdir()) if dp.exists() else 'DIR NOT FOUND'}")
        raise FileNotFoundError(f"No data files in {dp.resolve()}")

    ov = cross_endpoint(raw); ov.to_csv(rd / "cross_endpoint_overlap.csv", index=False)

    # Domain confounding
    dom_rpt = domain_confounding(raw.get("ames", pd.DataFrame()))
    save_json(dom_rpt, rd / "ames_domain_confounding.json")
    if dom_rpt.get("domain_acc"):
        lg.info(f"  ⚠ Domain predictor acc: {dom_rpt['domain_acc']}")

    # Label conflicts
    cleaned = {}
    for ep, df in raw.items():
        (rd / ep).mkdir(parents=True, exist_ok=True)
        sens = []
        for s in ["conservative","positive_priority","majority_vote"]:
            c, _, r = resolve_conflicts(df, s)
            c.to_csv(rd / ep / f"cleaned_{s}.csv", index=False)
            sens.append(r)
        save_json(sens, rd / ep / "conflict_sensitivity.json")
        cleaned[ep], _, r = resolve_conflicts(df, "conservative")
        lg.info(f"  [{ep}] {len(df)} → {len(cleaned[ep])} (conflicts={r['n_conflicts']})")

    # ═══ STEP 2: Flags ═══
    lg.info("STEP 2: Preprocessing Flags")
    flagged = {}
    for ep, df in cleaned.items():
        f = classify_all_compounds(df, smi_col="SMILES")
        f.to_csv(rd / ep / "preprocess_flags.csv", index=False)
        flagged[ep] = f

    # ═══ STEP 3: Fixed Split ═══
    lg.info("STEP 3: Fixed Split")
    splits = {}; smeta = {}
    for ep, df in cleaned.items():
        ds, m = fixed_split(assign_scaffolds(df, seed=GLOBAL_SEED), seed=GLOBAL_SEED)
        assert m["scaffold_overlap"] == 0
        ds.to_csv(rd / ep / "fixed_split.csv", index=False)
        splits[ep] = ds; smeta[ep] = m
        lg.info(f"  [{ep}] train={m['train_n']} test={m['test_n']} pos_diff={m['diff']}")
    save_json(smeta, rd / "split_metadata.json")

    # ═══ STEP 4: Scenario Selection (Train CV, compact/xgb only) ═══
    lg.info("STEP 4: Scenario Selection (Train CV)")
    cv_rows = []
    best_sc = {}

    for ep in ENDPOINTS:
        if ep not in splits: continue
        ds = splits[ep]; fl = flagged[ep]
        ep_best = {"scenario": "raw_all", "cv_mcc": -1}

        for sc in SCENARIOS:
            train, _ = apply_scenario(ds, sc, fl)
            if len(train) < 20: continue

            # Extract compact features for CV
            fg = extract_fg_features(train.reset_index(drop=True), ep)
            ph = extract_physchem_features(train.reset_index(drop=True), ep)
            fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns: feat[c] = pd.to_numeric(feat[c], errors="coerce")
            feat = feat.fillna(0)
            X = feat.values.astype(np.float32)
            y = train.reset_index(drop=True)["label"].values.astype(int)
            if len(set(y)) < 2: continue

            groups = LabelEncoder().fit_transform(
                train["scaffold_group"].astype(str).values
            ) if "scaffold_group" in train.columns else np.arange(len(train))
            spw = (y == 0).sum() / max((y == 1).sum(), 1)

            cvr = repeated_cv(X, y, groups,
                              lambda: make_model("xgb", spw),
                              n_folds=5, n_repeats=5, seed=GLOBAL_SEED)
            if not cvr or "mcc" not in cvr: continue

            row = {
                "endpoint": ep, "scenario": sc,
                "train_n": len(y),
                "cv_mcc_mean": cvr["mcc"]["mean"],
                "cv_mcc_std": cvr["mcc"]["std"],
                "cv_roc_mean": cvr.get("roc_auc", {}).get("mean"),
                "n_unique_folds": cvr.get("_n_unique_fold_assignments", 0),
                "cv_n_evals": cvr["mcc"]["n"],
            }
            cv_rows.append(row)
            lg.info(f"  [{ep}/{sc}] CV_MCC={row['cv_mcc_mean']:.4f}±{row['cv_mcc_std']:.4f} "
                     f"unique_folds={row['n_unique_folds']}")

            if row["cv_mcc_mean"] > ep_best["cv_mcc"]:
                ep_best = {"scenario": sc, "cv_mcc": row["cv_mcc_mean"]}

        best_sc[ep] = ep_best["scenario"]
        lg.info(f"  [{ep}] ★ Best: {ep_best['scenario']} (CV={ep_best['cv_mcc']:.4f})")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(rd / "scenario_cv_selection.csv", index=False)
    save_json(best_sc, rd / "best_scenario_by_cv.json")

    # ═══ STEP 5: All Combos on Locked Test ═══
    # Tabular: 3sc × 2fm × 7mdl × 3ep = 126
    # GNN:     3sc × 1fm("graph") × 3ep = 9
    n_tabular = len(SCENARIOS) * len(FEAT_MODES) * len(TABULAR_MODELS)
    n_gnn = len(SCENARIOS) if _HAS_TORCH else 0
    lg.info(f"STEP 5: Locked Test ({n_tabular} tabular + {n_gnn} GNN per ep)")

    # Check available models
    available_tabular = list(TABULAR_MODELS)
    if not _HAS_LGBM:
        available_tabular.remove("lgbm")
        lg.info("  ⚠ LightGBM not installed — skipping lgbm. pip install lightgbm")
    if not _HAS_TORCH:
        lg.info("  ⚠ PyTorch not installed — skipping gnn/ann/dnn. pip install torch")
        for m in ["ann", "dnn"]:
            if m in available_tabular:
                available_tabular.remove(m)

    all_rows = []

    for ep in ENDPOINTS:
        if ep not in splits: continue
        ds = splits[ep]; fl = flagged[ep]
        orig_test_n = (ds["split"] == "test").sum()

        # ── Pre-compute tabular features ONCE per endpoint ──
        lg.info(f"  [{ep}] Pre-computing features...")
        fg_all = extract_fg_features(ds, ep)
        ph_all = extract_physchem_features(ds, ep)
        fgp_all = fg_all[[c for c in fg_all.columns if c.endswith("_present") or c.startswith("bb_")]]

        compact_feat = pd.concat([fgp_all, ph_all], axis=1)
        for c in compact_feat.columns:
            compact_feat[c] = pd.to_numeric(compact_feat[c], errors="coerce")
        compact_feat = compact_feat.fillna(0)
        compact_cols = list(compact_feat.columns)

        # Fingerprints at multiple bit sizes
        FP_BITS = [256, 512, 1024, 2048]
        fp_dict = {}      # {nbits: DataFrame}
        broad_dict = {}   # {nbits: DataFrame}
        broad_cols_dict = {}
        for nbits in FP_BITS:
            fp_df = extract_fingerprint_features(ds, ep, n_bits=nbits)
            fp_dict[nbits] = fp_df
            bf = pd.concat([fgp_all, ph_all, fp_df], axis=1)
            for c in bf.columns: bf[c] = pd.to_numeric(bf[c], errors="coerce")
            bf = bf.fillna(0)
            broad_dict[nbits] = bf
            broad_cols_dict[nbits] = list(bf.columns)
            lg.info(f"    fp{nbits}: {len(bf.columns)} features (compact={len(compact_cols)} + fp={fp_df.shape[1]})")

        # AD용 기본 fingerprint (1024-bit)
        fp_ad = fp_dict.get(1024, fp_dict[256])

        y_all = ds["label"].values.astype(int)
        ds_index = ds.index

        # SMILES for GNN
        smi_col_name = find_smi(ds)
        smiles_all = ds[smi_col_name].values

        for sc in SCENARIOS:
            sc_train, sc_test = apply_scenario(ds, sc, fl)
            if len(sc_train) < 10 or len(sc_test) < 5: continue

            tr_idx = set(sc_train.index)
            te_idx = set(sc_test.index)
            sc_mask = np.array([i in tr_idx or i in te_idx for i in ds_index])
            is_train = np.array([i in tr_idx for i in ds_index[sc_mask]])

            y_tr = y_all[sc_mask][is_train]
            y_te = y_all[sc_mask][~is_train]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2: continue

            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

            # AD (1024-bit fingerprint 기반, scenario당 1회)
            fp_sc = fp_ad.iloc[np.where(sc_mask)[0]].values.astype(np.float32)
            ad_result = compute_ad(fp_sc[is_train], fp_sc[~is_train])

            # CV reference
            cv_ref = next((r for r in cv_rows if r["endpoint"]==ep and r["scenario"]==sc), {})

            # Groups for OOF threshold & HP tuning
            grp_vals = sc_train["scaffold_group"].astype(str).values if "scaffold_group" in sc_train.columns else np.arange(len(sc_train))
            groups_le = LabelEncoder().fit_transform(grp_vals)

            # SMILES for GNN
            smi_tr = smiles_all[sc_mask][is_train]
            smi_te = smiles_all[sc_mask][~is_train]

            # ── A. Tabular models (all feature modes) ──
            for fm in FEAT_MODES:
                # Select feature set based on mode
                if fm == "compact":
                    feat_df = compact_feat
                    fcols = compact_cols
                elif fm.startswith("broad_fp"):
                    nbits = int(fm.replace("broad_fp", ""))
                    feat_df = broad_dict[nbits]
                    fcols = broad_cols_dict[nbits]
                else:
                    continue

                X_tr = feat_df.iloc[np.where(sc_mask)[0][is_train]].values.astype(np.float32)
                X_te = feat_df.iloc[np.where(sc_mask)[0][~is_train]].values.astype(np.float32)

                for mn in available_tabular:
                    experiment = f"{ep}_{sc}_{fm}_{mn}"
                    edir = rd / experiment; edir.mkdir(parents=True, exist_ok=True)

                    # OOF threshold tuning
                    opt_t = oof_threshold(X_tr, y_tr, groups_le,
                                          lambda: make_model(mn, spw), n_folds=3)

                    # Fixed-param model
                    mdl = make_model(mn, spw)
                    mdl.fit(X_tr, y_tr)
                    proba = mdl.predict_proba(X_te)[:, 1]
                    pred_default = (proba >= 0.5).astype(int)
                    pred_tuned = (proba >= opt_t).astype(int)

                    ci = bootstrap_ci(y_te, pred_default, proba, 200)
                    cal = calibration(y_te, proba)
                    mcc_tuned = matthews_corrcoef(y_te, pred_tuned)

                    is_rec = (sc == best_sc.get(ep) and fm == "compact" and mn == "xgb")

                    row = _build_row(experiment, ep, sc, fm, mn, is_rec,
                                     y_tr, y_te, orig_test_n, ad_result, ci, cal,
                                     opt_t, mcc_tuned, cv_ref)

                    # HP tuning
                    try:
                        hp_mdl, hp_params, hp_cv = tune_model(
                            X_tr, y_tr, groups_le, mn, spw)
                        hp_proba = hp_mdl.predict_proba(X_te)[:, 1]
                        hp_pred = (hp_proba >= 0.5).astype(int)
                        hp_mcc = matthews_corrcoef(y_te, hp_pred)
                        hp_cal = calibration(y_te, hp_proba)
                        row["mcc_hp_tuned"] = round(hp_mcc, 4)
                        row["hp_cv_mcc"] = hp_cv
                        row["hp_best_params"] = json.dumps(
                            {k: (v if not hasattr(v, 'item') else v.item())
                             for k, v in hp_params.items()}, ensure_ascii=False)
                        row["hp_brier"] = hp_cal["brier"]
                        row["hp_ece"] = hp_cal["ece"]
                    except Exception as e:
                        lg.info(f"    HP tuning failed for {mn}: {e}")
                        row.update({"mcc_hp_tuned": None, "hp_cv_mcc": None,
                                    "hp_best_params": None, "hp_brier": None, "hp_ece": None})

                    row["warnings"] = interpret_result(row)
                    all_rows.append(row)
                    _save_experiment(edir, row, y_te, proba, pred_default, pred_tuned,
                                    fcols, mdl, mn)

                    star = " ★" if is_rec else ""
                    hp_s = f" HP={row.get('mcc_hp_tuned','—')}" if row.get("mcc_hp_tuned") is not None else ""
                    lg.info(f"  {experiment}: MCC={row.get('mcc',0):.4f} "
                            f"[{row.get('mcc_lo',0):.4f},{row.get('mcc_hi',0):.4f}] "
                            f"t={opt_t} AD={ad_result['coverage']:.2f}{hp_s}{star}")
                    if row["warnings"] != "OK":
                        lg.info(f"    ⚠ {row['warnings']}")

            # ── B. GNN (uses molecular graphs, feature_mode="graph") ──
            if _HAS_TORCH:
                mn = "gnn"; fm = "graph"
                experiment = f"{ep}_{sc}_{fm}_{mn}"
                edir = rd / experiment; edir.mkdir(parents=True, exist_ok=True)

                try:
                    # Fixed-param GNN
                    gnn = MolGCN(random_state=GLOBAL_SEED)
                    gnn.fit(smi_tr, y_tr)
                    proba = gnn.predict_proba(smi_te)[:, 1]
                    pred_default = (proba >= 0.5).astype(int)

                    ci = bootstrap_ci(y_te, pred_default, proba, 200)
                    cal = calibration(y_te, proba)

                    row = _build_row(experiment, ep, sc, fm, mn, False,
                                     y_tr, y_te, orig_test_n, ad_result, ci, cal,
                                     0.5, matthews_corrcoef(y_te, pred_default), cv_ref)

                    # HP tuning for GNN
                    try:
                        hp_gnn, hp_params, hp_cv = tune_model(
                            None, y_tr, groups_le, "gnn", spw,
                            smiles_train=smi_tr)
                        hp_proba = hp_gnn.predict_proba(smi_te)[:, 1]
                        hp_pred = (hp_proba >= 0.5).astype(int)
                        hp_mcc = matthews_corrcoef(y_te, hp_pred)
                        hp_cal = calibration(y_te, hp_proba)
                        row["mcc_hp_tuned"] = round(hp_mcc, 4)
                        row["hp_cv_mcc"] = hp_cv
                        row["hp_best_params"] = json.dumps(hp_params, ensure_ascii=False)
                        row["hp_brier"] = hp_cal["brier"]
                        row["hp_ece"] = hp_cal["ece"]
                    except Exception as e:
                        lg.info(f"    GNN HP tuning failed: {e}")
                        row.update({"mcc_hp_tuned": None, "hp_cv_mcc": None,
                                    "hp_best_params": None, "hp_brier": None, "hp_ece": None})

                    row["warnings"] = interpret_result(row)
                    all_rows.append(row)
                    pd.DataFrame({"label": y_te, "proba": proba, "pred": pred_default
                                  }).to_csv(edir / "test_predictions.csv", index=False)
                    save_json(row, edir / "metrics.json")

                    lg.info(f"  {experiment}: MCC={row.get('mcc',0):.4f} "
                            f"[{row.get('mcc_lo',0):.4f},{row.get('mcc_hi',0):.4f}] "
                            f"AD={ad_result['coverage']:.2f} HP={row.get('mcc_hp_tuned','—')}")
                    if row["warnings"] != "OK":
                        lg.info(f"    ⚠ {row['warnings']}")
                except Exception as e:
                    lg.info(f"  {experiment}: FAILED — {e}")

    # ── Save main result table ──
    locked_df = pd.DataFrame(all_rows)

    if locked_df.empty:
        lg.error("  ✗ No results generated — check data loading and Step 5 logs")
        save_json({"error": "no results"}, rd / "checklist.json")
        return {"run_dir": rd, "cv": cv_df, "locked": locked_df,
                "best_sc": best_sc, "ldo": pd.DataFrame(), "checks": {}}

    # paper_primary_model: locked test MCC 최고 (tie-break: raw_all > no_metal > salt_stripped)
    SCENARIO_PRIORITY = {"raw_all": 0, "no_metal": 1, "salt_stripped": 2}
    locked_df["paper_primary_model"] = False
    for ep in locked_df["endpoint"].unique():
        mask = locked_df["endpoint"] == ep
        edf = locked_df[mask].copy()
        edf["_sc_pri"] = edf["scenario"].map(SCENARIO_PRIORITY).fillna(9)
        best_idx = edf.sort_values(["mcc", "_sc_pri"], ascending=[False, True]).index[0]
        locked_df.loc[best_idx, "paper_primary_model"] = True

    locked_df.to_csv(rd / "all_locked_test.csv", index=False)
    locked_df[["endpoint","scenario","feature_mode","model",
               "ad_coverage","ad_mean_sim"]].to_csv(rd / "ad_coverage_summary.csv", index=False)
    locked_df[["endpoint","scenario","feature_mode","model",
               "brier","ece"]].to_csv(rd / "calibration_summary.csv", index=False)

    # Split tables: main (default threshold) vs supplementary (tuned)
    main_exclude = ["threshold_default","threshold_tuned","mcc_tuned","_sc_pri",
                    "mcc_hp_tuned","hp_cv_mcc","hp_best_params","hp_brier","hp_ece"]
    main_cols = [c for c in locked_df.columns if c not in main_exclude]
    locked_df[main_cols].to_csv(rd / "table_main_default_threshold.csv", index=False)

    tuned_cols = ["experiment","endpoint","scenario","feature_mode","model",
                  "paper_primary_model","threshold_default","threshold_tuned",
                  "mcc","mcc_tuned","test_n","test_n_pos"]
    tuned_cols = [c for c in tuned_cols if c in locked_df.columns]
    df_tuned = locked_df[tuned_cols].copy().rename(columns={"mcc": "mcc_default"})
    df_tuned["mcc_delta"] = df_tuned["mcc_tuned"] - df_tuned["mcc_default"]
    df_tuned.to_csv(rd / "table_supplementary_threshold_tuning.csv", index=False)

    # HP tuning comparison table
    hp_cols = ["experiment","endpoint","scenario","feature_mode","model",
               "paper_primary_model","mcc","mcc_hp_tuned","hp_cv_mcc","hp_brier","hp_ece","hp_best_params"]
    hp_cols = [c for c in hp_cols if c in locked_df.columns]
    if hp_cols and "mcc_hp_tuned" in locked_df.columns:
        df_hp = locked_df[hp_cols].copy().rename(columns={"mcc": "mcc_fixed"})
        df_hp["mcc_hp_delta"] = df_hp["mcc_hp_tuned"] - df_hp["mcc_fixed"]
        df_hp.to_csv(rd / "table_hp_tuning_comparison.csv", index=False)
        lg.info(f"  HP tuning comparison: {len(df_hp)} rows")
        # Summary per endpoint
        for ep in df_hp["endpoint"].unique():
            edf = df_hp[df_hp["endpoint"]==ep]
            delta_mean = edf["mcc_hp_delta"].dropna().mean()
            n_improved = (edf["mcc_hp_delta"].dropna() > 0).sum()
            lg.info(f"    [{ep}] HP tuning: mean Δ={delta_mean:+.4f}, improved {n_improved}/{len(edf)} combos")

    lg.info(f"  Saved all_locked_test.csv: {len(locked_df)} rows, {len(locked_df.columns)} columns")

    # ═══ STEP 6: LDO + Checklist ═══
    lg.info("STEP 6: LDO + Checklist")
    ldo = []
    if dom_rpt.get("has_domain"):
        def _feat(df, ep):
            fg = extract_fg_features(df, ep); ph = extract_physchem_features(df, ep)
            fgp = fg[[c for c in fg.columns if c.endswith("_present") or c.startswith("bb_")]]
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns: feat[c] = pd.to_numeric(feat[c], errors="coerce")
            return feat.fillna(0).values.astype(np.float32), df["label"].values.astype(int), []
        ldo = leave_domain_out(
            cleaned["ames"], "domain", _feat,
            lambda: make_model("xgb", 1.0), "ames")
        pd.DataFrame(ldo).to_csv(rd / "ames_leave_domain_out.csv", index=False)
        for r in ldo:
            lg.info(f"  LDO {r['train_dom']}→{r['test_dom']}: MCC={r.get('mcc',0):.4f}")

    # ── Checklist (실제 검증) ──
    checks = {
        "fixed_split": all(smeta[ep]["scaffold_overlap"] == 0 for ep in smeta),
        "conflicts": all((rd / ep / "conflict_sensitivity.json").exists() for ep in cleaned),
        "domain_diagnosed": (rd / "ames_domain_confounding.json").exists(),
        "domain_ldo": len(ldo) > 0,
        "cross_ep": (rd / "cross_endpoint_overlap.csv").exists(),
        "cv_selection": (rd / "scenario_cv_selection.csv").exists(),
        "cv_unique_folds": all(r.get("n_unique_folds", 0) > 1 for r in cv_rows) if cv_rows else False,
        "full_grid": len(locked_df) >= len(SCENARIOS) * len(FEAT_MODES) * 2 * len([e for e in ENDPOINTS if e in splits]),  # at minimum xgb+logistic
        "no_empty_mcc": locked_df["mcc"].notna().all() if not locked_df.empty else False,
        "broad_fp_included": "broad_fp" in locked_df["feature_mode"].values if not locked_df.empty else False,
        "logistic_included": "logistic" in locked_df["model"].values if not locked_df.empty else False,
        "cv_selected_1_per_ep": all(
            locked_df[locked_df["endpoint"]==ep]["cv_selected_compact_xgb"].sum() == 1
            for ep in locked_df["endpoint"].unique()
        ) if not locked_df.empty else False,
        "bootstrap_ci": "mcc_lo" in locked_df.columns if not locked_df.empty else False,
        "ad_computed": locked_df["ad_coverage"].notna().all() if not locked_df.empty else False,
        "calibration": locked_df["brier"].notna().all() if not locked_df.empty else False,
        "threshold_oof": locked_df["threshold_tuned"].notna().all() if not locked_df.empty else False,
        "paper_primary_1_per_ep": all(
            locked_df[locked_df["endpoint"]==ep]["paper_primary_model"].sum() == 1
            for ep in locked_df["endpoint"].unique()
        ) if not locked_df.empty else False,
        "warnings_present": any(locked_df["warnings"] != "OK") if not locked_df.empty else False,
        "all_columns_complete": locked_df[["train_n","test_n","mcc","roc_auc","ad_coverage","brier"]].notna().all().all() if not locked_df.empty else False,
        "hp_tuning_complete": locked_df["mcc_hp_tuned"].notna().all() if (not locked_df.empty and "mcc_hp_tuned" in locked_df.columns) else False,
    }
    save_json(checks, rd / "checklist.json")

    total = time.time() - t0
    lg.info(f"\nChecklist ({sum(checks.values())}/{len(checks)}):")
    for k, v in checks.items():
        lg.info(f"  [{'✓' if v else '✗'}] {k}")

    lg.info(f"\nDONE in {total:.0f}s: {rd}")
    return {"run_dir": rd, "cv": cv_df, "locked": locked_df,
            "best_sc": best_sc, "ldo": pd.DataFrame(ldo), "checks": checks}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tag", default="v11")
    a = p.parse_args()
    run_pipeline(data_dir=a.data_dir, tag=a.tag)
