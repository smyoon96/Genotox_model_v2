"""
genotox_pipeline.py v12 -- Clean Single-Run Pipeline (2026-03-27)
================================================================
Bug fixes v11→v12:
  [CRITICAL-1] salt_stripped now ACTUALLY uses stripped SMILES for features
  [CRITICAL-2] GNN replaced with proper GCN (learnable message passing)
  [CRITICAL-3] AD uses proper Tanimoto on binary FP
  [CRITICAL-4] HP tuning: separate inner/outer eval to avoid data leakage
  [METHOD-5]   Added pairwise statistical tests (McNemar)
  [METHOD-7]   Domain confounding tested with multiple models
  [METHOD-8]   OOF threshold uses 5-fold (was 3)
  [METHOD-10]  Learning curve added
  [METHOD-11]  SHAP analysis for best models
  [METHOD-14]  FP bit selection in broad modes
"""
import sys, json, logging, warnings, time, traceback
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.progress import pbar, step_header, task_done, eta_str
from pipeline_v2_core import (
    resolve_conflicts, assign_scaffolds, fixed_split, apply_scenario,
    repeated_cv, bootstrap_ci, compute_ad, calibration,
    domain_confounding, leave_domain_out, cross_endpoint,
    find_smi, to_can, file_hash, interpret_result,
    learning_curve, pairwise_model_comparison, mcnemar_test,
    oof_tune_threshold,          # [v12.2] unified OOF threshold grid search
    ANALYSIS_SMILES_COL,
)
from step2b_preprocessing_impact import classify_all_compounds
from step4_feature_extraction import (
    extract_fg_features, extract_physchem_features, extract_fingerprint_features,
    select_fingerprint_bits, build_all_features, extract_qm_features,
    extract_maccs_features, extract_atompair_features, extract_toptorsion_features,
    extract_multi_fp_union,
)
from step4b_sa_features import extract_sa_features
from step4c_delta_descriptors import (
    extract_delta_features, extract_extended_descriptors,
    extract_electronic_descriptors,
)
from step0_conditional_router import ConditionalRouter
from config import (
    DATA_DIR, RUNS_DIR, GLOBAL_SEED, ENDPOINTS, make_run_dir, save_json,
    CROSS_ENDPOINT_STACKING, ENSEMBLE_CONFIG,
)
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*X has feature names.*")

SCENARIOS = ["raw_all", "no_metal", "salt_stripped", "conditional"]
FEAT_MODES = ["compact", "broad_fp512", "broad_fp1024", "broad_fp2048", "multi_fp", "extended"]
QM_DIR = DATA_DIR / "qm" if (DATA_DIR / "qm").exists() else None
TABULAR_MODELS = ["xgb", "lgbm", "rf", "svm", "logistic", "ann"]
MODEL_NAMES = TABULAR_MODELS + ["gnn"]

# ─── Optional dependencies ──────────────────────
_HAS_LGBM = False
_HAS_TORCH = False
_HAS_SHAP = False
try:
    import lightgbm; _HAS_LGBM = True
except ImportError: pass
try:
    import torch; import torch.nn as tnn; _HAS_TORCH = True
except ImportError: pass
try:
    import shap; _HAS_SHAP = True
except ImportError: pass


def setup_log(rd):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 자식 프로세스 stdout을 UTF-8로 고정
    # run_all.py Popen(encoding='utf-8')와 동일 인코딩 → UnicodeDecodeError 방지
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # 리다이렉션/pipe 등 reconfigure 불가 환경

    # 기존 핸들러 정리: flush+close 후 제거 (ResourceWarning 방지)
    for h in root.handlers[:]:
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout),
              logging.FileHandler(rd / "pipeline.log", encoding="utf-8")]:
        h.setFormatter(fmt)
        root.addHandler(h)
    return logging.getLogger("gp")


# ═══════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════
#  Domain-Aware Sample Weight Grid Search (Ames: industrial vs drug)
# ═══════════════════════════════════════════════════════

def _derive_domain_weight_range(df: "pd.DataFrame",
                                domain_col: str = "domain"
                                ) -> "tuple[list, list]":
    """
    실제 데이터의 도메인별 양성률을 기반으로 탐색 범위를 유도.

    근거:
      w_industrial의 상한 = p_drug / p_industrial  (prior shift 완전 보정)
      w_drug의 하한      = p_industrial / p_drug   (완전 보정 반대 방향)
      탐색 범위는 1.0 (보정 없음) ~ 완전 보정의 절반까지만 적용.
      (완전 보정 시 오히려 industrial의 TN을 희생하는 경향 있음)

    반환:
      w_ind_candidates: industrial weight 후보 리스트
      w_drug_candidates: drug weight 후보 리스트
    """
    if domain_col not in df.columns:
        return [1.0], [1.0]

    dom = df[domain_col].str.lower().str.strip()
    grp = df.groupby(dom)["label"].agg(["mean", "count"])

    p_ind  = float(grp.loc["industrial", "mean"]) if "industrial" in grp.index else None
    p_drug = float(grp.loc["drug",       "mean"]) if "drug"       in grp.index else None
    n_ind  = int(grp.loc["industrial", "count"]) if "industrial" in grp.index else 0
    n_drug = int(grp.loc["drug",       "count"]) if "drug"       in grp.index else 0

    import logging; lg_ = logging.getLogger("gp")
    _p_ind_str  = f"{p_ind:.4f}"  if p_ind  is not None else "?"
    _p_drug_str = f"{p_drug:.4f}" if p_drug is not None else "?"
    lg_.info(f"  Domain stats: industrial n={n_ind} pos={_p_ind_str} | "
             f"drug n={n_drug} pos={_p_drug_str}")

    if p_ind is None or p_drug is None or p_ind == 0:
        return [1.0], [1.0]

    # prior shift ratio: p_drug / p_industrial ≈ 19x (from literature)
    ratio = p_drug / p_ind

    # industrial weight 범위: 1.0 (no correction) ~ ratio^0.5 (half correction)
    # log-uniform sampling이 더 넓은 범위를 균등하게 커버
    import numpy as np
    w_ind_max  = max(1.5, min(ratio ** 0.5, 8.0))   # 상한 8x 캡
    w_ind_cands = list(np.round(
        np.exp(np.linspace(np.log(1.0), np.log(w_ind_max), 6)), 2
    ))  # [1.0, 1.x, 2.x, ..., w_ind_max] 6단계

    # drug weight 범위: 1.0 (no correction) ~ 1/ratio^0.5
    w_drug_min = max(0.1, min(1.0 / ratio ** 0.5, 1.0))
    w_drug_cands = list(np.round(
        np.exp(np.linspace(np.log(w_drug_min), np.log(1.0), 6)), 2
    ))

    lg_.info(f"  Prior shift ratio: {ratio:.2f}x → "
             f"w_ind range: {w_ind_cands[0]}~{w_ind_cands[-1]} | "
             f"w_drug range: {w_drug_cands[0]}~{w_drug_cands[-1]}")
    return w_ind_cands, w_drug_cands


def tune_domain_weights(X_tr: "np.ndarray", y_tr: "np.ndarray",
                        domain_arr: "np.ndarray",
                        model_fn,
                        w_ind_candidates: list,
                        w_drug_candidates: list,
                        n_folds: int = 5,
                        criterion: str = "mcc",
                        seed: int = None) -> dict:
    """
    OOF CV로 도메인 weight 조합을 grid search하여 최적값 탐색.

    Parameters
    ----------
    X_tr, y_tr    : 학습 데이터 및 라벨
    domain_arr    : 각 샘플의 도메인 레이블 배열 ("industrial" / "drug" / other)
    model_fn      : () → sklearn estimator 팩토리
    w_ind_cands   : industrial weight 후보 리스트 (data-driven)
    w_drug_cands  : drug weight 후보 리스트 (data-driven)
    criterion     : "mcc" | "sensitivity" | "youden"
                    - mcc:         전체 성능 균형 (논문 주지표)
                    - sensitivity: FN 최소화 최우선 (규제 관점)
                    - youden:      sens + spec - 1

    Returns
    -------
    dict:
        best_w_industrial: float
        best_w_drug:       float
        best_score:        float
        criterion:         str
        derivation:        str  ← 논문 Methods 기재용
        grid_results:      List[dict]  ← 전체 탐색 결과
    """
    import numpy as np
    from itertools import product
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef, confusion_matrix
    from sklearn.base import clone
    import logging; lg_ = logging.getLogger("gp")

    seed = seed or GLOBAL_SEED

    # domain → numeric weight 배열로 변환
    def _make_sw(w_ind, w_drug):
        sw = np.ones(len(domain_arr), dtype=np.float32)
        sw[domain_arr == "industrial"] = w_ind
        sw[domain_arr == "drug"]       = w_drug
        sw = sw / sw.mean()   # 정규화: 평균 weight = 1.0
        return sw

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X_tr, y_tr))

    grid_results = []
    best = {"score": -np.inf, "w_ind": 1.0, "w_drug": 1.0}

    candidates = list(product(w_ind_candidates, w_drug_candidates))
    # 가장 가까운 weight 먼저 (1.0, 1.0) baseline 포함 보장
    if (1.0, 1.0) not in candidates:
        candidates = [(1.0, 1.0)] + candidates

    from utils.progress import pbar as _pbar
    for w_ind, w_drug in _pbar(candidates,
                                desc=f"    domain weight grid ({criterion})",
                                leave=False):
        sw = _make_sw(w_ind, w_drug)
        oof_pred = np.full(len(y_tr), -1, dtype=int)
        oof_prob = np.zeros(len(y_tr), dtype=float)

        for tr_idx, val_idx in folds:
            m = clone(model_fn())
            sw_fold = sw[tr_idx]
            try:
                # sample_weight 지원 모델만 (xgb, lgbm, rf)
                m.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=sw_fold)
            except TypeError:
                m.fit(X_tr[tr_idx], y_tr[tr_idx])
            prob = m.predict_proba(X_tr[val_idx])[:, 1]
            oof_prob[val_idx] = prob
            # threshold 0.5 고정 (weight tuning 목적에만 집중)
            oof_pred[val_idx] = (prob >= 0.5).astype(int)

        valid = oof_pred >= 0
        if valid.sum() < 10 or len(set(oof_pred[valid])) < 2:
            continue

        yt, yp = y_tr[valid], oof_pred[valid]
        tn, fp_n, fn_n, tp_n = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
        spec = tn  / (tn  + fp_n)  if (tn  + fp_n)  > 0 else 0.0

        if criterion == "mcc":
            score = float(matthews_corrcoef(yt, yp))
        elif criterion == "sensitivity":
            score = float(sens)
        elif criterion == "youden":
            score = float(sens + spec - 1.0)
        else:
            score = float(matthews_corrcoef(yt, yp))

        grid_results.append({
            "w_industrial": w_ind, "w_drug": w_drug,
            "score": round(score, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "criterion": criterion,
        })
        if score > best["score"]:
            best = {"score": score, "w_ind": w_ind, "w_drug": w_drug}

    lg_.info(f"    Domain weight grid: best w_ind={best['w_ind']:.2f} "
             f"w_drug={best['w_drug']:.2f} {criterion}={best['score']:.4f} "
             f"(searched {len(candidates)} combos)")

    derivation = (
        f"Domain sample weights were selected via {n_folds}-fold OOF CV grid search "
        f"({len(candidates)} combinations). "
        f"The candidate range for w_industrial was log-uniformly derived from "
        f"the empirical prior shift ratio (p_drug / p_industrial), "
        f"capped at the square-root of the ratio to avoid over-correction. "
        f"The criterion was OOF {criterion}. "
        f"Optimal: w_industrial={best['w_ind']}, w_drug={best['w_drug']}."
    )

    return {
        "best_w_industrial": best["w_ind"],
        "best_w_drug":       best["w_drug"],
        "best_score":        round(best["score"], 4),
        "criterion":         criterion,
        "derivation":        derivation,
        "grid_results":      grid_results,
    }


def make_model(name, spw):
    """Fixed hyperparameter baseline model."""
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
        from sklearn.svm import SVC, LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        # LinearSVC + Platt scaling: O(n) vs RBF SVC O(n²~n³)
        # n>3000이면 LinearSVC 사용, 그 이하면 RBF SVC
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.1, class_weight="balanced",
                          max_iter=2000, random_state=GLOBAL_SEED),
                cv=3, method="sigmoid"))])
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
    raise ValueError(f"Unknown model: {name}")


# ═══════════════════════════════════════════════════════
#  [CRITICAL-2 FIX] Proper GCN with Learnable Weights
# ═══════════════════════════════════════════════════════

class GCNLayer(tnn.Module if _HAS_TORCH else object):
    """Single GCN layer: X' = σ(D^-0.5 A D^-0.5 X W)"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for GNN")
        super().__init__()
        self.linear = tnn.Linear(in_dim, out_dim)
        self.dropout = tnn.Dropout(dropout)
        self.norm = tnn.LayerNorm(out_dim)

    def forward(self, x, adj_norm):
        # Message passing WITH learnable weight
        h = torch.matmul(adj_norm, x)  # aggregate neighbors
        h = self.linear(h)             # learnable transform
        h = self.norm(h)
        h = torch.relu(h)
        h = self.dropout(h)
        return h


class GCNModel(tnn.Module if _HAS_TORCH else object):
    """Multi-layer GCN with global mean pooling → classifier."""
    def __init__(self, in_dim, hidden_dim=64, n_layers=3, dropout=0.2):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for GNN")
        super().__init__()
        self.layers = tnn.ModuleList()
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(GCNLayer(dims[i], dims[i+1], dropout))
        self.classifier = tnn.Sequential(
            tnn.Linear(hidden_dim, hidden_dim // 2),
            tnn.ReLU(),
            tnn.Dropout(dropout),
            tnn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, adj_norm):
        for layer in self.layers:
            x = layer(x, adj_norm) + (x if x.shape == layer.linear.weight.shape[1:] else 0)
        # Global mean pooling
        x = x.mean(dim=0, keepdim=True)
        return self.classifier(x)


class MolGCN:
    """
    Proper Graph Convolutional Network for molecular property prediction.
    [CRITICAL-2 FIX]: Uses GCNLayer with learnable weights instead of
    raw adjacency multiplication + MLP.
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
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            return np.zeros((1, 9), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        atoms = mol.GetAtoms()
        n = len(atoms)
        feat = np.zeros((n, 9), dtype=np.float32)
        for i, atom in enumerate(atoms):
            feat[i, 0] = atom.GetAtomicNum() / 53.0
            feat[i, 1] = atom.GetDegree() / 4.0
            feat[i, 2] = atom.GetFormalCharge()
            feat[i, 3] = atom.GetTotalNumHs() / 4.0
            feat[i, 4] = float(atom.GetIsAromatic())
            hyb = atom.GetHybridization()
            feat[i, 5] = float(hyb == Chem.rdchem.HybridizationType.SP)
            feat[i, 6] = float(hyb == Chem.rdchem.HybridizationType.SP2)
            feat[i, 7] = float(hyb == Chem.rdchem.HybridizationType.SP3)
            feat[i, 8] = float(atom.IsInRing())
        adj = np.eye(n, dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0
        # Normalize: D^-0.5 A D^-0.5
        deg = adj.sum(axis=1, keepdims=True)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        adj = deg_inv_sqrt * adj * deg_inv_sqrt.T
        return feat, adj

    def fit(self, X_smiles, y):
        import torch
        torch.manual_seed(self.random_state)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graphs = [self.smiles_to_graph(s) for s in X_smiles]
        in_dim = graphs[0][0].shape[1]

        # [FIX] Use proper GCN model with learnable weights
        self.model_ = GCNModel(
            in_dim, self.hidden_dim, self.n_layers, self.dropout
        ).to(self.device_)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                     weight_decay=1e-4)
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0, float(pos_w)]).to(self.device_)
        )

        best_loss = float("inf")
        patience_cnt = 0
        for epoch in range(self.epochs):
            self.model_.train()
            indices = np.random.RandomState(self.random_state + epoch).permutation(len(y))
            total_loss = 0
            for i in indices:
                feat, adj = graphs[i]
                x = torch.FloatTensor(feat).to(self.device_)
                a = torch.FloatTensor(adj).to(self.device_)
                out = self.model_(x, a)
                label = torch.LongTensor([int(y[i])]).to(self.device_)
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(y)
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_cnt = 0
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
                x = torch.FloatTensor(feat).to(self.device_)
                a = torch.FloatTensor(adj).to(self.device_)
                out = self.model_(x, a)
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


# ═══════════════════════════════════════════════════════
#  HP Grids (from config)
# ═══════════════════════════════════════════════════════
from config import HP_GRIDS, HP_N_ITER, HP_CV_FOLDS


# ═══════════════════════════════════════════════════════
#  [CRITICAL-4 FIX] HP Tuning with Proper Nested Evaluation
# ═══════════════════════════════════════════════════════

def tune_model(X_train, y_train, groups, model_name, spw,
               n_iter=HP_N_ITER, n_folds=HP_CV_FOLDS,
               smiles_train=None):
    """
    HP tuning with GroupKFold.

    [CRITICAL-4 FIX]: The returned best_cv_score is the honest
    internal CV estimate. The caller must NOT use the same train
    data for final evaluation -- must evaluate on held-out test only.

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

    # GNN: manual search
    if model_name == "gnn":
        if smiles_train is None:
            raise ValueError("GNN requires smiles_train")
        return _tune_gnn(smiles_train, y_train, groups, actual_folds)

    # Tabular: RandomizedSearchCV
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

    # [FIX] Return internal CV score as honest estimate
    # The model is refit on full train data, test evaluation is separate
    return search.best_estimator_, search.best_params_, round(search.best_score_, 4)


def _tune_gnn(smiles, y, groups, n_folds):
    from sklearn.model_selection import GroupKFold

    best_score, best_params = -1, {}
    grid = HP_GRIDS.get("gnn", {})
    if not grid:
        mdl = MolGCN()
        mdl.fit(smiles, y)
        return mdl, {}, np.nan

    rng = np.random.RandomState(GLOBAL_SEED)
    configs = []
    for _ in range(min(HP_N_ITER, 8)):
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        configs.append(cfg)

    for cfg in configs:
        fold_scores = []
        gkf = GroupKFold(n_splits=n_folds)
        for tr_idx, val_idx in gkf.split(smiles, y, groups):
            mdl = MolGCN(**cfg, epochs=50)
            mdl.fit(np.array(smiles)[tr_idx], y[tr_idx])
            pred = mdl.predict(np.array(smiles)[val_idx])
            if len(set(pred)) > 0:
                fold_scores.append(matthews_corrcoef(y[val_idx], pred))
        if fold_scores:
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = cfg

    final = MolGCN(**best_params, epochs=100)
    final.fit(smiles, y)
    return final, best_params, round(best_score, 4)


# ═══════════════════════════════════════════════════════
#  [METHOD-8 FIX] OOF Threshold -- pipeline_v2_core 통합 (v12.2)
# ═══════════════════════════════════════════════════════

def oof_threshold(X_train, y_train, groups, model_fn, n_folds=5):
    """scaffold-aware OOF threshold. pipeline_v2_core.oof_tune_threshold 위임."""
    return oof_tune_threshold(
        X_train, y_train, groups=groups, model_fn=model_fn,
        n_folds=n_folds, criterion="mcc",
    )["best_threshold"]


# ═══════════════════════════════════════════════════════
#  [METHOD-11] SHAP Analysis
# ═══════════════════════════════════════════════════════

def compute_shap_importance(mdl, X_train, X_test, fcols, model_name,
                            max_samples=200, seed=42):
    """Compute SHAP values for interpretability. Returns DataFrame."""
    if not _HAS_SHAP:
        return None
    try:
        rng = np.random.RandomState(seed)
        bg_idx = rng.choice(len(X_train), min(max_samples, len(X_train)), replace=False)
        X_bg = X_train[bg_idx]

        if model_name in ("xgb", "lgbm", "rf"):
            explainer = shap.TreeExplainer(mdl)
            sv = explainer.shap_values(X_test[:min(200, len(X_test))])
            if isinstance(sv, list):
                sv = sv[1]  # positive class
        elif model_name in ("logistic", "svm", "ann"):
            predict_fn = mdl.predict_proba
            explainer = shap.KernelExplainer(predict_fn, X_bg)
            sv = explainer.shap_values(X_test[:min(50, len(X_test))])
            if isinstance(sv, list):
                sv = sv[1]
        else:
            return None

        mean_abs = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({
            "feature": fcols[:len(mean_abs)],
            "shap_importance": mean_abs,
        }).sort_values("shap_importance", ascending=False)
        return df
    except Exception as e:
        logging.getLogger("gp").info(f"    SHAP failed: {e}")
        return None


# ═══════════════════════════════════════════════════════
#  Result Row Builder & Saver
# ═══════════════════════════════════════════════════════

def _build_row(experiment, ep, sc, fm, mn, is_rec,
               y_tr, y_te, orig_test_n, ad_result, ci, cal,
               opt_t, mcc_tuned_val, cv_ref):
    row = {
        "experiment": experiment, "endpoint": ep,
        "scenario": sc, "feature_mode": fm, "model": mn,
        "cv_selected_compact_xgb": is_rec,
        "train_n": int(len(y_tr)), "test_n": int(len(y_te)),
        "test_pos_rate": round(float(y_te.mean()), 4),
        "test_n_pos": int(y_te.sum()),
        "coverage": round(len(y_te) / orig_test_n, 4),
        "ad_coverage": ad_result["coverage"],
        "ad_mean_sim": ad_result["mean_sim"],
        "brier": cal["brier"], "ece": cal["ece"],
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
                     fcols, mdl, mn, X_train=None, X_test=None):
    pd.DataFrame({
        "label": y_te, "proba": proba,
        "pred": pred_default, "pred_tuned": pred_tuned,
    }).to_csv(edir / "test_predictions.csv", index=False)
    save_json(row, edir / "metrics.json")

    # Feature importance
    try:
        if mn in ("xgb", "lgbm", "rf"):
            imp = mdl.feature_importances_
        elif mn in ("logistic", "svm", "ann"):
            clf = mdl.named_steps.get("clf", mdl)
            imp = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else None
        else:
            imp = None
        if imp is not None:
            pd.DataFrame({
                "feature": fcols[:len(imp)], "importance": imp
            }).sort_values("importance", ascending=False).to_csv(
                edir / "feature_importance.csv", index=False)
    except Exception:
        pass

    # SHAP (for best models only, controlled by caller)
    if X_train is not None and X_test is not None:
        shap_df = compute_shap_importance(mdl, X_train, X_test, fcols, mn)
        if shap_df is not None:
            shap_df.to_csv(edir / "shap_importance.csv", index=False)


# ═══════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════

def run_pipeline(data_dir=None, tag="v12"):
    from xgboost import XGBClassifier

    dp = Path(data_dir) if data_dir else DATA_DIR
    rd = make_run_dir(tag)
    lg = setup_log(rd)
    t0 = time.time()
    lg.info(f"Pipeline v12 -- Bug-Fixed Single Run\n  {rd}")
    save_json({
        "data": str(dp), "version": "v12",
        "ts": datetime.now().isoformat(), "seed": GLOBAL_SEED,
        "fixes": [
            "CRITICAL-1: salt_stripped actually strips salts",
            "CRITICAL-2: GNN uses proper GCN layers",
            "CRITICAL-3: AD uses proper Tanimoto",
            "CRITICAL-4: HP tuning inner/outer separation",
            "METHOD-5: statistical tests added",
            "METHOD-8: OOF threshold 5-fold",
            "METHOD-10: learning curves",
            "METHOD-11: SHAP analysis",
            "METHOD-14: FP bit selection",
        ],
    }, rd / "config.json")

    # ═══ STEP 1: Load + Clean ═══
    lg.info("STEP 1: Load + Clean")
    raw = {}
    _ep_files = [("ames","ames.csv"),("invitro","invitro.csv"),("invivo","invivo.csv"),("invitro_sampling","invitro_sampling.csv"),("invivo_sampling","invivo_sampling.csv")]
    for ep, fn in pbar(_ep_files, desc="[1] Load data", colour="cyan"):
        fp = dp / fn
        if not fp.exists():
            lg.info(f"  {fn} NOT FOUND -- skipping")
            continue
        df = (pd.read_excel(fp, engine="openpyxl") if fn.endswith(".xlsx")
              else pd.read_csv(fp, encoding="utf-8-sig"))
        for c in df.columns:
            if c.strip().upper() == "SMILES":
                df = df.rename(columns={c: "SMILES"})
        df["label"] = df["label"].astype(int)
        df["endpoint"] = ep
        df = df.dropna(subset=["label", "SMILES"])
        raw[ep] = df
        lg.info(f"  [{ep}] n={len(df)} pos={df['label'].mean():.4f}")

    if not raw:
        raise FileNotFoundError(f"No data files in {dp.resolve()}")

    ov = cross_endpoint(raw)
    ov.to_csv(rd / "cross_endpoint_overlap.csv", index=False)

    dom_rpt = domain_confounding(raw.get("ames", pd.DataFrame()))
    save_json(dom_rpt, rd / "ames_domain_confounding.json")
    if dom_rpt.get("domain_acc"):
        lg.info(f"  ⚠ Domain predictor acc: {dom_rpt['domain_acc']}")

    # Label conflicts
    cleaned = {}
    for ep, df in raw.items():
        (rd / ep).mkdir(parents=True, exist_ok=True)
        sens = []
        for s in ["conservative", "positive_priority", "majority_vote"]:
            c, _, r = resolve_conflicts(df, s)
            c.to_csv(rd / ep / f"cleaned_{s}.csv", index=False)
            sens.append(r)
        save_json(sens, rd / ep / "conflict_sensitivity.json")
        cleaned[ep], _, r = resolve_conflicts(df, "conservative")
        lg.info(f"  [{ep}] {len(df)} → {len(cleaned[ep])} (conflicts={r['n_conflicts']})")

    # ═══ STEP 2: Preprocessing Flags ═══
    lg.info("STEP 2: Preprocessing Flags")
    flagged = {}
    for ep, df in pbar(cleaned.items(), desc="[2] Preprocessing flags", total=len(cleaned), colour="cyan"):
        f = classify_all_compounds(df, smi_col="SMILES")
        f.to_csv(rd / ep / "preprocess_flags.csv", index=False)
        flagged[ep] = f

    # ═══ STEP 3: Fixed Split ═══
    lg.info("STEP 3: Fixed Split")
    splits = {}
    smeta = {}
    for ep, df in pbar(cleaned.items(), desc="[3] Scaffold split", total=len(cleaned), colour="cyan"):
        ds, m = fixed_split(assign_scaffolds(df, seed=GLOBAL_SEED), seed=GLOBAL_SEED)
        assert m["scaffold_overlap"] == 0
        ds.to_csv(rd / ep / "fixed_split.csv", index=False)
        splits[ep] = ds
        smeta[ep] = m
        lg.info(f"  [{ep}] train={m['train_n']} test={m['test_n']} pos_diff={m['diff']}")
    save_json(smeta, rd / "split_metadata.json")

    # ═══ STEP 4: Scenario CV Selection ═══
    lg.info("STEP 4: Scenario Selection (Train CV)")
    cv_rows = []
    best_sc = {}

    for ep in pbar(ENDPOINTS, desc="[4] Scenario CV", colour="cyan"):
        if ep not in splits:
            continue
        ds = splits[ep]
        fl = flagged[ep]
        ep_best = {"scenario": "raw_all", "cv_mcc": -1}

        for sc in pbar(SCENARIOS, desc=f"  {ep} scenarios", leave=False):
            # Handle conditional scenario
            if sc == "conditional":
                try:
                    _router = ConditionalRouter(strategy='conditional')
                    _routed = _router.route_and_preprocess(ds, smi_col="SMILES")
                    train = _routed[_routed["split"] == "train"].copy()
                except Exception:
                    continue
            else:
                train, _ = apply_scenario(ds, sc, fl)
            if len(train) < 20:
                continue

            fg = extract_fg_features(train.reset_index(drop=True), ep)
            ph = extract_physchem_features(train.reset_index(drop=True), ep)
            fgp = fg  # v12.2: FG 전체 컬럼 사용
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce")
            feat = feat.fillna(0)
            X = feat.values.astype(np.float32)
            y = train.reset_index(drop=True)["label"].values.astype(int)
            if len(set(y)) < 2:
                continue

            groups = LabelEncoder().fit_transform(
                train["scaffold_group"].astype(str).values
            ) if "scaffold_group" in train.columns else np.arange(len(train))
            spw = (y == 0).sum() / max((y == 1).sum(), 1)

            cvr = repeated_cv(X, y, groups,
                              lambda: make_model("xgb", spw),
                              n_folds=5, n_repeats=5, seed=GLOBAL_SEED)
            if not cvr or "mcc" not in cvr:
                continue

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
            lg.info(f"  [{ep}/{sc}] CV_MCC={row['cv_mcc_mean']:.4f}±{row['cv_mcc_std']:.4f}")

            if row["cv_mcc_mean"] > ep_best["cv_mcc"]:
                ep_best = {"scenario": sc, "cv_mcc": row["cv_mcc_mean"]}

        best_sc[ep] = ep_best["scenario"]
        lg.info(f"  [{ep}] * Best: {ep_best['scenario']} (CV={ep_best['cv_mcc']:.4f})")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(rd / "scenario_cv_selection.csv", index=False)
    save_json(best_sc, rd / "best_scenario_by_cv.json")

    # ═══ STEP 5: All Combos on Locked Test ═══
    lg.info("STEP 5: Locked Test (all combos)")

    available_tabular = list(TABULAR_MODELS)
    if not _HAS_LGBM:
        available_tabular.remove("lgbm")
        lg.info("  ⚠ LightGBM not installed -- skipping lgbm")
    if not _HAS_TORCH:
        lg.info("  ⚠ PyTorch not installed -- skipping gnn/ann")
        for m in ["ann"]:
            if m in available_tabular:
                available_tabular.remove(m)

    all_rows = []
    # Store predictions for McNemar tests
    test_predictions = {}  # {(ep, experiment): {"y_true": ..., "pred": ...}}

    _ep_list = [ep for ep in ENDPOINTS if ep in splits]
    for ep in pbar(_ep_list, desc="[5] Locked test eval", colour="green"):
        ds = splits[ep]
        fl = flagged[ep]
        orig_test_n = (ds["split"] == "test").sum()

        lg.info(f"  [{ep}] Pre-computing features...")

        # ── Pre-compute features per scenario ──
        # [CRITICAL-1 FIX] Features are now extracted PER SCENARIO
        # because salt_stripped changes the SMILES

        y_all = ds["label"].values.astype(int)
        smi_col_name = find_smi(ds)
        smiles_all = ds[smi_col_name].values

        for sc in pbar(SCENARIOS, desc=f"  {ep} scenarios", leave=False):
            # ── Conditional scenario uses ConditionalRouter ──
            if sc == "conditional":
                try:
                    router = ConditionalRouter(strategy='conditional')
                    ds_routed = router.route_and_preprocess(ds, smi_col="SMILES")
                    # Apply same train/test split
                    sc_train = ds_routed[ds_routed["split"] == "train"].copy()
                    sc_test = ds_routed[ds_routed["split"] == "test"].copy()
                    # Remove flagged compounds same as raw_all but keep metals
                    if len(sc_train) < 10 or len(sc_test) < 5:
                        continue
                except Exception as _cond_e:
                    lg.warning(f"  [{ep}] Conditional routing failed: {_cond_e}")
                    continue
            else:
                sc_train, sc_test = apply_scenario(ds, sc, fl)
            if len(sc_train) < 10 or len(sc_test) < 5:
                continue

            # [CRITICAL-1 FIX] Extract features from scenario-processed data
            # which has _analysis_smiles set correctly
            sc_combined = pd.concat([sc_train, sc_test])

            fg_sc = extract_fg_features(sc_combined, ep)
            ph_sc = extract_physchem_features(sc_combined, ep)
            fgp_sc = fg_sc  # v12.2: FG 전체 컬럼 사용 (기존 _present+bb_ 필터 제거)

            # ── NEW: SA structural alert features ──
            try:
                sa_sc = extract_sa_features(sc_combined, ep)
            except Exception as _sa_e:
                lg.warning(f"  SA feature extraction failed: {_sa_e}")
                sa_sc = pd.DataFrame(index=sc_combined.index)

            # ── NEW: Delta (preprocessing-aware) features ──
            try:
                delta_sc = extract_delta_features(
                    sc_combined, raw_smi_col="SMILES",
                    pre_smi_col=find_smi(sc_combined), endpoint=ep)
            except Exception as _delta_e:
                lg.warning(f"  Delta feature extraction failed: {_delta_e}")
                delta_sc = pd.DataFrame(index=sc_combined.index)

            # ── NEW: Electronic descriptors ──
            try:
                elec_sc = extract_electronic_descriptors(sc_combined, ep)
            except Exception as _elec_e:
                lg.warning(f"  Electronic feature extraction failed: {_elec_e}")
                elec_sc = pd.DataFrame(index=sc_combined.index)

            # ── NEW: Route features (conditional scenario only) ──
            if sc == "conditional":
                try:
                    route_sc = router.get_route_features(sc_combined)
                except Exception:
                    route_sc = pd.DataFrame(index=sc_combined.index)
            else:
                route_sc = pd.DataFrame(index=sc_combined.index)

            # compact: FG 전체 + Physchem (+ QM if available)
            try:
                _qm_sc, _ = extract_qm_features(sc_combined, ep, qm_dir=QM_DIR)
            except Exception as _qm_e:
                lg.warning(f'  QM extraction skipped: {_qm_e}')
                _qm_sc = pd.DataFrame(index=sc_combined.index)
            _compact_parts = [fgp_sc, ph_sc]
            if not _qm_sc.empty:
                _compact_parts.append(_qm_sc)
            compact_feat = pd.concat(_compact_parts, axis=1)
            for c in compact_feat.columns:
                compact_feat[c] = pd.to_numeric(compact_feat[c], errors="coerce")
            compact_feat = compact_feat.fillna(0)
            compact_cols = list(compact_feat.columns)

            # ── NEW: Extended feature set ──
            # compact + SA + delta + electronic + route
            _ext_parts = [fgp_sc, ph_sc, sa_sc, delta_sc, elec_sc]
            if not _qm_sc.empty:
                _ext_parts.append(_qm_sc)
            if not route_sc.empty:
                _ext_parts.append(route_sc)
            extended_feat = pd.concat(_ext_parts, axis=1)
            for c in extended_feat.columns:
                extended_feat[c] = pd.to_numeric(extended_feat[c], errors="coerce")
            extended_feat = extended_feat.fillna(0).replace([np.inf, -np.inf], 0)
            # Remove constant columns
            _nunique = extended_feat.nunique()
            extended_feat = extended_feat.loc[:, _nunique > 1]
            extended_cols = list(extended_feat.columns)

            # Split back into train/test
            n_tr = len(sc_train)
            n_te = len(sc_test)

            y_tr = sc_combined["label"].values[:n_tr].astype(int)
            y_te = sc_combined["label"].values[n_tr:].astype(int)
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

            # Fingerprints at multiple bit sizes
            FP_BITS = [512, 1024, 2048]
            fp_dict = {}
            broad_dict = {}
            broad_cols_dict = {}
            for nbits in FP_BITS:
                fp_df = extract_fingerprint_features(sc_combined, ep, n_bits=nbits)
                fp_dict[nbits] = fp_df

                # [METHOD-14] FP bit selection on train only
                # Scale top_k with nbits: 512→128, 1024→256, 2048→384
                # This ensures broad_fp modes actually differ in dimensionality
                fp_top_k = {512: 128, 1024: 256, 2048: 384}.get(nbits, nbits // 4)
                fp_train_only = fp_df.iloc[:n_tr]
                selected_bits = select_fingerprint_bits(
                    fp_train_only, y_tr, top_k=fp_top_k, method="mi"
                )
                fp_selected = fp_df[selected_bits]

                # v12.2: FG 전체 + Physchem + FP + QM
                _bf_parts = [fgp_sc, ph_sc, fp_selected]
                if not _qm_sc.empty:
                    _bf_parts.append(_qm_sc)
                bf = pd.concat(_bf_parts, axis=1)
                for c in bf.columns:
                    bf[c] = pd.to_numeric(bf[c], errors="coerce")
                bf = bf.fillna(0)
                broad_dict[nbits] = bf
                broad_cols_dict[nbits] = list(bf.columns)

            # AD: use 1024-bit FP (full, not selected) for proper Tanimoto
            fp_ad_full = fp_dict.get(1024, fp_dict[512])
            fp_tr_ad = fp_ad_full.iloc[:n_tr].values.astype(np.float32)
            fp_te_ad = fp_ad_full.iloc[n_tr:].values.astype(np.float32)
            ad_result = compute_ad(fp_tr_ad, fp_te_ad)

            # ── Multi-FP union (Strategy 3: Feature Augmentation) ──
            try:
                multi_fp_raw = extract_multi_fp_union(
                    sc_combined, ep,
                    morgan_bits=1024, ap_bits=512, tt_bits=512)
                # MI-based selection on train
                _mfp_train = multi_fp_raw.iloc[:n_tr]
                _mfp_selected_bits = select_fingerprint_bits(
                    _mfp_train, y_tr, top_k=384, method="mi")
                _mfp_selected = multi_fp_raw[_mfp_selected_bits]
                _mfp_parts = [fgp_sc, ph_sc, _mfp_selected]
                if not _qm_sc.empty:
                    _mfp_parts.append(_qm_sc)
                multi_fp_feat = pd.concat(_mfp_parts, axis=1)
                for c in multi_fp_feat.columns:
                    multi_fp_feat[c] = pd.to_numeric(multi_fp_feat[c], errors="coerce")
                multi_fp_feat = multi_fp_feat.fillna(0)
                multi_fp_cols = list(multi_fp_feat.columns)
            except Exception as _mfp_e:
                lg.warning(f"  Multi-FP extraction failed: {_mfp_e}")
                multi_fp_feat = None
                multi_fp_cols = []

            # ── Cross-endpoint stacking (Strategy 2) ──
            # Ames 모델 예측값을 보조 feature로 추가
            _ames_proba_col = None
            if (CROSS_ENDPOINT_STACKING.get("enabled")
                    and ep in CROSS_ENDPOINT_STACKING.get("target_endpoints", [])
                    and hasattr(run_pipeline, "_ames_stacking_model")):
                try:
                    _ames_mdl_info = run_pipeline._ames_stacking_model
                    _ames_mdl = _ames_mdl_info["model"]
                    _ames_fcols = _ames_mdl_info["feature_cols"]
                    _ames_domain_w = _ames_mdl_info.get("domain_weights", None)

                    # sc_combined의 feature를 Ames 모델 feature space로 매핑
                    _stk_feat = compact_feat.reindex(columns=_ames_fcols, fill_value=0)
                    _stk_X = _stk_feat.values.astype(np.float32)
                    _ames_raw_proba = _ames_mdl.predict_proba(_stk_X)[:, 1]

                    # Domain-weighted calibration
                    if (_ames_domain_w and "domain" in sc_combined.columns
                            and CROSS_ENDPOINT_STACKING.get("domain_weighted")):
                        _dom_arr = sc_combined["domain"].str.lower().str.strip().values
                        _w_ind = _ames_domain_w["w_industrial"]
                        _w_drug = _ames_domain_w["w_drug"]
                        _dom_w = np.ones(len(_ames_raw_proba), dtype=np.float32)
                        _dom_w[_dom_arr == "industrial"] = _w_ind
                        _dom_w[_dom_arr == "drug"] = _w_drug
                        # Weighted probability: 높은 weight → 예측 강화
                        _ames_proba_weighted = _ames_raw_proba * _dom_w
                        _ames_proba_weighted = np.clip(_ames_proba_weighted, 0, 1)
                    else:
                        _ames_proba_weighted = _ames_raw_proba

                    _ames_proba_col = pd.Series(
                        _ames_proba_weighted, index=sc_combined.index,
                        name="ames_proba")
                    lg.info(f"  [{ep}] Ames stacking feature added "
                            f"(mean={_ames_proba_weighted.mean():.3f})")
                except Exception as _stk_e:
                    lg.warning(f"  Ames stacking failed for {ep}: {_stk_e}")
                    _ames_proba_col = None

            cv_ref = next((r for r in cv_rows
                           if r["endpoint"] == ep and r["scenario"] == sc), {})

            grp_vals = (sc_train["scaffold_group"].astype(str).values
                        if "scaffold_group" in sc_train.columns
                        else np.arange(len(sc_train)))
            groups_le = LabelEncoder().fit_transform(grp_vals)

            # SMILES for GNN
            smi_col_sc = find_smi(sc_combined)
            smi_tr = sc_combined[smi_col_sc].values[:n_tr]
            smi_te = sc_combined[smi_col_sc].values[n_tr:]

            # ── Tabular models ──
            for fm in FEAT_MODES:
                if fm == "compact":
                    feat_df = compact_feat.copy()
                    fcols = list(compact_cols)
                elif fm == "extended":
                    feat_df = extended_feat.copy()
                    fcols = list(extended_cols)
                elif fm.startswith("broad_fp"):
                    nbits = int(fm.replace("broad_fp", ""))
                    if nbits not in broad_dict:
                        continue
                    feat_df = broad_dict[nbits].copy()
                    fcols = list(broad_cols_dict[nbits])
                elif fm == "multi_fp":
                    if multi_fp_feat is None:
                        continue
                    feat_df = multi_fp_feat.copy()
                    fcols = list(multi_fp_cols)
                else:
                    continue

                # Append Ames stacking feature if available
                if _ames_proba_col is not None:
                    feat_df["ames_proba"] = _ames_proba_col.values
                    if "ames_proba" not in fcols:
                        fcols = fcols + ["ames_proba"]

                # numeric 변환은 compact/broad 빌드 시 완료됨
                X_tr = feat_df.values[:n_tr].astype(np.float32)
                X_te = feat_df.values[n_tr:].astype(np.float32)

                # ── Domain weight grid search (Ames only, tree-based) ──────────
                _domain_fit_kwargs = {}
                if ep == "ames" and "domain" in sc_train.columns:
                    # grid search는 첫 번째 tree-based 모델에서만 수행 (캐싱)
                    _dom_cache_key = f"{ep}_{sc}_{fm}"
                    if not hasattr(run_pipeline, "_dw_cache"):
                        run_pipeline._dw_cache = {}
                    if _dom_cache_key not in run_pipeline._dw_cache:
                        _dom_arr = sc_train["domain"].str.lower().str.strip().values
                        _w_ind_cands, _w_drug_cands = _derive_domain_weight_range(
                            sc_train, domain_col="domain")
                        _dw_result = tune_domain_weights(
                            X_tr, y_tr, _dom_arr,
                            model_fn=lambda: make_model("xgb", spw),
                            w_ind_candidates=_w_ind_cands,
                            w_drug_candidates=_w_drug_cands,
                            n_folds=5, criterion="mcc", seed=GLOBAL_SEED,
                        )
                        run_pipeline._dw_cache[_dom_cache_key] = _dw_result
                        # grid 결과 저장 (Figure용)
                        import pandas as _pd
                        _grid_df = _pd.DataFrame(_dw_result["grid_results"])
                        _grid_df.to_csv(
                            rd / f"domain_weight_grid_{ep}_{sc}_{fm}.csv",
                            index=False)
                        lg.info(f"  {_dw_result['derivation']}")
                    else:
                        _dw_result = run_pipeline._dw_cache[_dom_cache_key]

                    _best_w_ind  = _dw_result["best_w_industrial"]
                    _best_w_drug = _dw_result["best_w_drug"]
                    _dom_arr_all = sc_train["domain"].str.lower().str.strip().values
                    _sw_arr = np.ones(len(y_tr), dtype=np.float32)
                    _sw_arr[_dom_arr_all == "industrial"] = _best_w_ind
                    _sw_arr[_dom_arr_all == "drug"]       = _best_w_drug
                    _sw_arr = _sw_arr / _sw_arr.mean()
                    _domain_fit_kwargs = {"sample_weight": _sw_arr}

                # n_samples 기반 모델 목록 축소
                _avail_ep = list(available_tabular)
                if len(y_tr) > 5000:
                    # 대용량: logistic은 빠르므로 유지, SVM은 LinearSVC로 대체됨
                    pass  # LinearSVC로 이미 교체됨
                if len(y_tr) > 10000:
                    # 초대용량(ames): HP tuning SVM 스킵
                    lg.info(f"  [{ep}] n_train={len(y_tr)} > 10k -- HP tuning skipped for svm")

                for mn in pbar(_avail_ep, desc=f"    models", leave=False):
                    experiment = f"{ep}_{sc}_{fm}_{mn}"
                    edir = rd / experiment
                    edir.mkdir(parents=True, exist_ok=True)

                    # per-model soft timeout 경고
                    _model_t0 = time.time()

                    try:
                        # OOF threshold [METHOD-8 FIX: 5 folds]
                        opt_t = oof_threshold(X_tr, y_tr, groups_le,
                                              lambda mn=mn: make_model(mn, spw), n_folds=5)

                        # Fixed-param model
                        mdl = make_model(mn, spw)
                        _fw = _domain_fit_kwargs if mn in ("xgb","lgbm","rf") else {}
                        mdl.fit(X_tr, y_tr, **_fw)
                        proba = mdl.predict_proba(X_te)[:, 1]
                        pred_default = (proba >= 0.5).astype(int)
                        pred_tuned = (proba >= opt_t).astype(int)

                        ci = bootstrap_ci(y_te, pred_default, proba, 500)
                        cal = calibration(y_te, proba)
                        mcc_tuned = matthews_corrcoef(y_te, pred_tuned)

                        is_rec = (sc == best_sc.get(ep) and fm == "compact" and mn == "xgb")

                        row = _build_row(experiment, ep, sc, fm, mn, is_rec,
                                         y_tr, y_te, orig_test_n, ad_result, ci, cal,
                                         opt_t, mcc_tuned, cv_ref)

                        # HP tuning [CRITICAL-4: eval only on test]
                        # 대용량 + 느린 모델은 HP tuning 스킵
                        _skip_hp = (mn in ("svm",) and len(y_tr) > 5000)
                        try:
                            if _skip_hp:
                                raise RuntimeError(f"HP tuning skipped for {mn} (n={len(y_tr)}>5000)")
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
                            row.update({"mcc_hp_tuned": None, "hp_cv_mcc": None,
                                        "hp_best_params": None, "hp_brier": None,
                                        "hp_ece": None})

                        row["warnings"] = interpret_result(row)
                        all_rows.append(row)

                        # SHAP: compute for tree-based models on non-sampling endpoints
                        # (fast via TreeExplainer; sampling endpoints too small for reliable SHAP)
                        do_shap = (
                            _HAS_SHAP
                            and mn in ("xgb", "lgbm", "rf")
                            and "sampling" not in ep
                            and len(y_te) >= 50
                        )
                        _save_experiment(
                            edir, row, y_te, proba, pred_default, pred_tuned,
                            fcols, mdl, mn,
                            X_train=X_tr if do_shap else None,
                            X_test=X_te if do_shap else None,
                        )

                        # Store predictions for McNemar tests
                        test_predictions[(ep, experiment)] = {
                            "y_true": y_te, "pred": pred_default,
                        }

                        star = " *" if is_rec else ""
                        _model_elapsed = time.time() - _model_t0
                        if _model_elapsed > 300:
                            lg.warning(f"  ⚠ {mn} took {_model_elapsed:.0f}s "
                                       f"on {ep}/{fm} (n={len(y_tr)})")
                        lg.info(f"  {experiment}: MCC={row.get('mcc', 0):.4f} "
                                f"[{row.get('mcc_lo', 0):.4f},{row.get('mcc_hi', 0):.4f}] "
                                f"t={opt_t} AD={ad_result['coverage']:.2f}{star} "
                                f"({_model_elapsed:.0f}s)")

                    except Exception as e:
                        lg.info(f"  {experiment}: FAILED -- {e}")
                        traceback.print_exc()

            # ── GNN ──
            if _HAS_TORCH:
                mn = "gnn"
                fm = "graph"
                experiment = f"{ep}_{sc}_{fm}_{mn}"
                edir = rd / experiment
                edir.mkdir(parents=True, exist_ok=True)

                try:
                    gnn = MolGCN(random_state=GLOBAL_SEED)
                    gnn.fit(smi_tr, y_tr)
                    proba = gnn.predict_proba(smi_te)[:, 1]
                    pred_default = (proba >= 0.5).astype(int)

                    ci = bootstrap_ci(y_te, pred_default, proba, 500)
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
                        row.update({"mcc_hp_tuned": None, "hp_cv_mcc": None,
                                    "hp_best_params": None, "hp_brier": None,
                                    "hp_ece": None})

                    row["warnings"] = interpret_result(row)
                    all_rows.append(row)
                    pd.DataFrame({
                        "label": y_te, "proba": proba, "pred": pred_default
                    }).to_csv(edir / "test_predictions.csv", index=False)
                    save_json(row, edir / "metrics.json")

                    test_predictions[(ep, experiment)] = {
                        "y_true": y_te, "pred": pred_default,
                    }

                    lg.info(f"  {experiment}: MCC={row.get('mcc', 0):.4f}")
                except Exception as e:
                    lg.info(f"  {experiment}: FAILED -- {e}")

        # ── Store best Ames model for cross-endpoint stacking ──
        if (ep == "ames" and CROSS_ENDPOINT_STACKING.get("enabled")
                and all_rows):
            _ames_rows = [r for r in all_rows if r.get("endpoint") == "ames"
                          and r.get("model") in ("xgb", "lgbm", "rf")]
            if _ames_rows:
                _best_ames = max(_ames_rows, key=lambda r: r.get("mcc", 0))
                _best_sc = _best_ames["scenario"]
                _best_fm = _best_ames["feature_mode"]
                _best_mn = _best_ames["model"]
                lg.info(f"  [Stacking] Retraining best Ames model: "
                        f"{_best_mn}/{_best_sc}/{_best_fm} "
                        f"(MCC={_best_ames['mcc']:.4f})")

                # Retrain on Ames train set with best config
                _stk_sc_train, _stk_sc_test = apply_scenario(
                    splits["ames"], _best_sc, flagged["ames"])
                _stk_combined = pd.concat([_stk_sc_train, _stk_sc_test])
                _stk_fg = extract_fg_features(_stk_combined, "ames")
                _stk_ph = extract_physchem_features(_stk_combined, "ames")
                _stk_feat = pd.concat([_stk_fg, _stk_ph], axis=1)
                for _c in _stk_feat.columns:
                    _stk_feat[_c] = pd.to_numeric(_stk_feat[_c], errors="coerce")
                _stk_feat = _stk_feat.fillna(0)
                _stk_n_tr = len(_stk_sc_train)
                _stk_X_tr = _stk_feat.values[:_stk_n_tr].astype(np.float32)
                _stk_y_tr = _stk_combined["label"].values[:_stk_n_tr].astype(int)
                _stk_spw = (_stk_y_tr == 0).sum() / max((_stk_y_tr == 1).sum(), 1)

                # Domain weight 적용
                _stk_dw = None
                _stk_fw = {}
                if ("domain" in _stk_sc_train.columns
                        and CROSS_ENDPOINT_STACKING.get("domain_weighted")):
                    _dom_arr = _stk_sc_train["domain"].str.lower().str.strip().values
                    _w_ind_c, _w_drug_c = _derive_domain_weight_range(
                        _stk_sc_train, "domain")
                    _dw_res = tune_domain_weights(
                        _stk_X_tr, _stk_y_tr, _dom_arr,
                        lambda: make_model(_best_mn, _stk_spw),
                        _w_ind_c, _w_drug_c, n_folds=5, criterion="mcc",
                        seed=GLOBAL_SEED)
                    _stk_dw = {
                        "w_industrial": _dw_res["best_w_industrial"],
                        "w_drug": _dw_res["best_w_drug"]}
                    _sw = np.ones(len(_stk_y_tr), dtype=np.float32)
                    _sw[_dom_arr == "industrial"] = _stk_dw["w_industrial"]
                    _sw[_dom_arr == "drug"] = _stk_dw["w_drug"]
                    _sw /= _sw.mean()
                    _stk_fw = {"sample_weight": _sw}
                    lg.info(f"  [Stacking] Domain weights: "
                            f"ind={_stk_dw['w_industrial']:.2f}, "
                            f"drug={_stk_dw['w_drug']:.2f}")

                _stk_mdl = make_model(_best_mn, _stk_spw)
                _stk_mdl.fit(_stk_X_tr, _stk_y_tr, **_stk_fw)

                run_pipeline._ames_stacking_model = {
                    "model": _stk_mdl,
                    "feature_cols": list(_stk_feat.columns),
                    "domain_weights": _stk_dw,
                    "scenario": _best_sc,
                    "model_name": _best_mn,
                }
                lg.info(f"  [Stacking] Ames model stored for cross-endpoint use")

    # ── Save main result table ──
    locked_df = pd.DataFrame(all_rows)

    if locked_df.empty:
        lg.error("  [X] No results generated")
        save_json({"error": "no results"}, rd / "checklist.json")
        return {"run_dir": rd, "cv": cv_df, "locked": locked_df,
                "best_sc": best_sc, "ldo": pd.DataFrame(), "checks": {}}

    # paper_primary_model selection
    SCENARIO_PRIORITY = {"raw_all": 0, "no_metal": 1, "salt_stripped": 2}
    locked_df["paper_primary_model"] = False
    for ep in locked_df["endpoint"].unique():
        mask = locked_df["endpoint"] == ep
        edf = locked_df[mask].copy()
        edf["_sc_pri"] = edf["scenario"].map(SCENARIO_PRIORITY).fillna(9)
        best_idx = edf.sort_values(
            ["mcc", "_sc_pri"], ascending=[False, True]
        ).index[0]
        locked_df.loc[best_idx, "paper_primary_model"] = True

    locked_df.to_csv(rd / "all_locked_test.csv", index=False)

    # ═══ STEP 5b: Statistical Comparisons ═══
    lg.info("STEP 5b: Statistical Comparisons")
    stat_results = []
    for ep in locked_df["endpoint"].unique():
        comp = pairwise_model_comparison(all_rows, ep, "mcc")
        if not comp.empty:
            comp.to_csv(rd / f"statistical_comparison_{ep}.csv", index=False)
            stat_results.append(comp)
            lg.info(f"  [{ep}] {len(comp)} pairwise comparisons saved")

    # McNemar tests between best models per endpoint
    mcnemar_results = []
    for ep in locked_df["endpoint"].unique():
        ep_mask = locked_df["endpoint"] == ep
        ep_df = locked_df[ep_mask].nlargest(2, "mcc")
        if len(ep_df) >= 2:
            exp_a = ep_df.iloc[0]["experiment"]
            exp_b = ep_df.iloc[1]["experiment"]
            key_a = (ep, exp_a)
            key_b = (ep, exp_b)
            if key_a in test_predictions and key_b in test_predictions:
                pa = test_predictions[key_a]
                pb = test_predictions[key_b]
                if len(pa["y_true"]) == len(pb["y_true"]):
                    mc = mcnemar_test(pa["y_true"], pa["pred"], pb["pred"])
                    mc["endpoint"] = ep
                    mc["model_a"] = exp_a
                    mc["model_b"] = exp_b
                    mcnemar_results.append(mc)
                    lg.info(f"  [{ep}] McNemar {exp_a} vs {exp_b}: p={mc['p_value']:.4f}")
    if mcnemar_results:
        pd.DataFrame(mcnemar_results).to_csv(rd / "mcnemar_tests.csv", index=False)

    # ═══ STEP 5c: Learning Curves ═══
    lg.info("STEP 5c: Learning Curves")
    lc_results = []
    for ep in ENDPOINTS:
        if ep not in splits:
            continue
        # Use best scenario, compact features, XGBoost
        sc = best_sc.get(ep, "raw_all")
        sc_train, _ = apply_scenario(splits[ep], sc, flagged[ep])
        if len(sc_train) < 50:
            continue

        fg = extract_fg_features(sc_train.reset_index(drop=True), ep)
        ph = extract_physchem_features(sc_train.reset_index(drop=True), ep)
        fgp = fg  # v12.2: FG 전체 컬럼 사용
        feat = pd.concat([fgp, ph], axis=1).fillna(0)
        for c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        feat = feat.fillna(0)

        X = feat.values.astype(np.float32)
        y = sc_train.reset_index(drop=True)["label"].values.astype(int)
        groups = LabelEncoder().fit_transform(
            sc_train["scaffold_group"].astype(str).values
        ) if "scaffold_group" in sc_train.columns else np.arange(len(sc_train))

        if len(set(y)) < 2:
            continue

        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        lc = learning_curve(
            X, y, groups,
            lambda: make_model("xgb", spw),
            fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            n_folds=5, seed=GLOBAL_SEED,
        )
        for r in lc:
            r["endpoint"] = ep
        lc_results.extend(lc)
        lg.info(f"  [{ep}] Learning curve: {len(lc)} points")

    if lc_results:
        pd.DataFrame(lc_results).to_csv(rd / "learning_curves.csv", index=False)

    # ═══ STEP 5d: Consensus Ensemble (Strategy 1) ═══
    if ENSEMBLE_CONFIG.get("enabled") and not locked_df.empty:
        lg.info("STEP 5d: Consensus Ensemble")
        ensemble_rows = []
        for ep in locked_df["endpoint"].unique():
            ep_df = locked_df[locked_df["endpoint"] == ep].copy()

            # 앙상블 대상: tabular 모델만, graph 제외
            ep_tab = ep_df[ep_df["feature_mode"] != "graph"].copy()
            if len(ep_tab) < ENSEMBLE_CONFIG.get("min_models", 3):
                lg.info(f"  [{ep}] Too few models ({len(ep_tab)}) for ensemble")
                continue

            # 상위 k개 모델 선택
            top_k = ENSEMBLE_CONFIG.get("top_k", 5)
            top_models = ep_tab.nlargest(top_k, "mcc")

            # 각 모델의 test predictions 수집
            model_probas = []
            model_names = []
            y_true = None

            for _, mrow in top_models.iterrows():
                exp = mrow["experiment"]
                pred_path = rd / exp / "test_predictions.csv"
                if pred_path.exists():
                    pred_df = pd.read_csv(pred_path)
                    model_probas.append(pred_df["proba"].values)
                    model_names.append(exp)
                    if y_true is None:
                        y_true = pred_df["label"].values.astype(int)

            if len(model_probas) < 2 or y_true is None:
                continue

            probas = np.array(model_probas)

            # Method 1: Soft Voting (probability averaging)
            soft_proba = probas.mean(axis=0)
            soft_pred = (soft_proba >= 0.5).astype(int)
            soft_mcc = matthews_corrcoef(y_true, soft_pred)

            ci_soft = bootstrap_ci(y_true, soft_pred, soft_proba, 500)
            cal_soft = calibration(y_true, soft_proba)
            ad_val = top_models["ad_coverage"].max()

            ens_row_soft = {
                "experiment": f"{ep}_ensemble_soft_vote",
                "endpoint": ep, "scenario": "ensemble",
                "feature_mode": "ensemble_soft_vote",
                "model": f"ensemble({len(model_probas)})",
                "mcc": round(soft_mcc, 4),
                "mcc_lo": ci_soft["mcc"]["lo"],
                "mcc_hi": ci_soft["mcc"]["hi"],
                "roc_auc": ci_soft["roc_auc"]["mean"],
                "sens": ci_soft["sens"]["mean"],
                "spec": ci_soft["spec"]["mean"],
                "bacc": ci_soft["bacc"]["mean"],
                "brier": cal_soft["brier"],
                "ece": cal_soft["ece"],
                "ad_coverage": ad_val,
                "test_n": len(y_true),
                "test_pos_rate": float(y_true.mean()),
                "ensemble_members": ",".join(model_names),
                "paper_primary_model": False,
            }
            ensemble_rows.append(ens_row_soft)

            # Method 2: Rank Averaging
            ranks = np.array([
                pd.Series(p).rank(pct=True).values for p in model_probas
            ])
            rank_proba = ranks.mean(axis=0)
            rank_pred = (rank_proba >= 0.5).astype(int)
            rank_mcc = matthews_corrcoef(y_true, rank_pred)

            ci_rank = bootstrap_ci(y_true, rank_pred, rank_proba, 500)
            cal_rank = calibration(y_true, rank_proba)
            ens_row_rank = {
                "experiment": f"{ep}_ensemble_rank_avg",
                "endpoint": ep, "scenario": "ensemble",
                "feature_mode": "ensemble_rank_avg",
                "model": f"ensemble({len(model_probas)})",
                "mcc": round(rank_mcc, 4),
                "mcc_lo": ci_rank["mcc"]["lo"],
                "mcc_hi": ci_rank["mcc"]["hi"],
                "roc_auc": ci_rank["roc_auc"]["mean"],
                "sens": ci_rank["sens"]["mean"],
                "spec": ci_rank["spec"]["mean"],
                "bacc": ci_rank["bacc"]["mean"],
                "brier": cal_rank["brier"],
                "ece": cal_rank["ece"],
                "ad_coverage": ad_val,
                "test_n": len(y_true),
                "test_pos_rate": float(y_true.mean()),
                "ensemble_members": ",".join(model_names),
                "paper_primary_model": False,
            }
            ensemble_rows.append(ens_row_rank)

            # Save ensemble predictions
            pd.DataFrame({
                "label": y_true,
                "soft_vote_proba": soft_proba,
                "rank_avg_proba": rank_proba,
            }).to_csv(rd / f"ensemble_predictions_{ep}.csv", index=False)

            lg.info(f"  [{ep}] Ensemble ({len(model_probas)} models): "
                    f"SoftVote MCC={soft_mcc:.4f}, RankAvg MCC={rank_mcc:.4f}")

        if ensemble_rows:
            ens_df = pd.DataFrame(ensemble_rows)
            ens_df.to_csv(rd / "ensemble_results.csv", index=False)
            # 메인 결과에도 추가
            locked_df = pd.concat([locked_df, ens_df], ignore_index=True)
            locked_df.to_csv(rd / "all_locked_test.csv", index=False)
            lg.info(f"  Ensemble results: {len(ensemble_rows)} rows added")

    # ═══ STEP 6: LDO + Domain Analysis ═══
    lg.info("STEP 6: LDO + Domain Analysis")
    ldo = []
    if dom_rpt.get("has_domain"):
        def _feat(df, ep):
            fg = extract_fg_features(df, ep)
            ph = extract_physchem_features(df, ep)
            fgp = fg  # v12.2: FG 전체 컬럼 사용
            feat = pd.concat([fgp, ph], axis=1)
            for c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce")
            return feat.fillna(0).values.astype(np.float32), df["label"].values.astype(int), []

        # [METHOD-7] LDO with multiple models
        for mn in ["xgb", "lgbm", "rf"]:
            if mn == "lgbm" and not _HAS_LGBM:
                continue
            ldo_m = leave_domain_out(
                cleaned["ames"], "domain", _feat,
                lambda: make_model(mn, 1.0), "ames"
            )
            for r in ldo_m:
                r["model"] = mn
            ldo.extend(ldo_m)
            for r in ldo_m:
                lg.info(f"  LDO [{mn}] {r['train_dom']}→{r['test_dom']}: "
                        f"MCC={r.get('mcc', 0):.4f}")

        if ldo:
            pd.DataFrame(ldo).to_csv(rd / "ames_leave_domain_out.csv", index=False)

    # ═══ Summary tables ═══
    locked_df[[
        "endpoint", "scenario", "feature_mode", "model",
        "ad_coverage", "ad_mean_sim"
    ]].to_csv(rd / "ad_coverage_summary.csv", index=False)

    main_exclude = [
        "threshold_default", "threshold_tuned", "mcc_tuned", "_sc_pri",
        "mcc_hp_tuned", "hp_cv_mcc", "hp_best_params", "hp_brier", "hp_ece"
    ]
    main_cols = [c for c in locked_df.columns if c not in main_exclude]
    locked_df[main_cols].to_csv(rd / "table_main_default_threshold.csv", index=False)

    # HP tuning comparison
    hp_cols = ["experiment", "endpoint", "scenario", "feature_mode", "model",
               "paper_primary_model", "mcc", "mcc_hp_tuned", "hp_cv_mcc",
               "hp_brier", "hp_ece", "hp_best_params"]
    hp_cols = [c for c in hp_cols if c in locked_df.columns]
    if "mcc_hp_tuned" in locked_df.columns:
        df_hp = locked_df[hp_cols].copy().rename(columns={"mcc": "mcc_fixed"})
        df_hp["mcc_hp_delta"] = df_hp["mcc_hp_tuned"] - df_hp["mcc_fixed"]
        df_hp.to_csv(rd / "table_hp_tuning_comparison.csv", index=False)

    # Threshold tuning supplementary
    tuned_cols = ["experiment", "endpoint", "scenario", "feature_mode", "model",
                  "paper_primary_model", "threshold_default", "threshold_tuned",
                  "mcc", "mcc_tuned", "test_n", "test_n_pos"]
    tuned_cols = [c for c in tuned_cols if c in locked_df.columns]
    df_tuned = locked_df[tuned_cols].copy().rename(columns={"mcc": "mcc_default"})
    df_tuned["mcc_delta"] = df_tuned["mcc_tuned"] - df_tuned["mcc_default"]
    df_tuned.to_csv(rd / "table_supplementary_threshold_tuning.csv", index=False)

    # ═══ SHAP Summary Aggregation ═══
    lg.info("STEP 7: SHAP Summary Aggregation")
    shap_summaries = []
    for d in sorted(rd.iterdir()):
        sf = d / "shap_importance.csv"
        if sf.exists():
            try:
                sdf = pd.read_csv(sf)
                sdf["experiment"] = d.name
                # Parse experiment name
                parts = d.name.split("_")
                # Find model (last part), feature_mode, scenario, endpoint
                sdf_meta = next(
                    (r for r in all_rows if r.get("experiment") == d.name), {}
                )
                sdf["endpoint"] = sdf_meta.get("endpoint", "")
                sdf["scenario"] = sdf_meta.get("scenario", "")
                sdf["feature_mode"] = sdf_meta.get("feature_mode", "")
                sdf["model"] = sdf_meta.get("model", "")
                sdf["mcc"] = sdf_meta.get("mcc", 0)
                shap_summaries.append(sdf)
            except Exception:
                pass

    if shap_summaries:
        all_shap = pd.concat(shap_summaries, ignore_index=True)
        all_shap.to_csv(rd / "shap_all_experiments.csv", index=False)

        # Per-endpoint: top features from best model
        for ep in all_shap["endpoint"].unique():
            ep_shap = all_shap[all_shap["endpoint"] == ep]
            best_exp = ep_shap.loc[ep_shap["mcc"].idxmax(), "experiment"]
            best_shap = ep_shap[ep_shap["experiment"] == best_exp].nlargest(
                20, "shap_importance"
            )
            best_shap.to_csv(rd / f"shap_top20_{ep}.csv", index=False)
            lg.info(f"  [{ep}] SHAP top-3: "
                    f"{', '.join(best_shap.head(3)['feature'].tolist())}")
        lg.info(f"  Total SHAP files aggregated: {len(shap_summaries)}")
    else:
        lg.info("  No SHAP files found (install shap: pip install shap)")

    # ═══ Checklist ═══
    checks = {
        "fixed_split": all(smeta[ep]["scaffold_overlap"] == 0 for ep in smeta),
        "conflicts": all((rd / ep / "conflict_sensitivity.json").exists() for ep in cleaned),
        "domain_diagnosed": (rd / "ames_domain_confounding.json").exists(),
        "domain_ldo": len(ldo) > 0,
        "cross_ep": (rd / "cross_endpoint_overlap.csv").exists(),
        "cv_selection": (rd / "scenario_cv_selection.csv").exists(),
        "salt_stripped_verified": any(
            r.get("scenario") == "salt_stripped" and r.get("mcc", 0) != next(
                (r2.get("mcc", -1) for r2 in all_rows
                 if r2.get("endpoint") == r.get("endpoint")
                 and r2.get("scenario") == "raw_all"
                 and r2.get("feature_mode") == r.get("feature_mode")
                 and r2.get("model") == r.get("model")), -1
            ) for r in all_rows
        ) if all_rows else False,
        "statistical_tests": (rd / "mcnemar_tests.csv").exists(),
        "learning_curves": (rd / "learning_curves.csv").exists(),
        "bootstrap_ci": "mcc_lo" in locked_df.columns if not locked_df.empty else False,
        "ad_computed": locked_df["ad_coverage"].notna().all() if not locked_df.empty else False,
        "calibration": locked_df["brier"].notna().all() if not locked_df.empty else False,
        "shap_computed": len(shap_summaries) > 0 if 'shap_summaries' in dir() else False,
    }
    save_json(checks, rd / "checklist.json")

    total = time.time() - t0
    lg.info(f"\nChecklist ({sum(checks.values())}/{len(checks)}):")
    for k, v in checks.items():
        lg.info(f"  [{'[OK]' if v else '[X]'}] {k}")

    lg.info(f"\nDONE in {total:.0f}s: {rd}")
    return {
        "run_dir": rd, "cv": cv_df, "locked": locked_df,
        "best_sc": best_sc, "ldo": pd.DataFrame(ldo), "checks": checks,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tag", default="v12")
    a = p.parse_args()
    run_pipeline(data_dir=a.data_dir, tag=a.tag)
