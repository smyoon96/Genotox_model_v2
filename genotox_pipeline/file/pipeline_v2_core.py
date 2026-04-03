"""
pipeline_v2_core.py — v8 최종 (2026-03-19)
모든 치명적/주요 문제 수정. 함수명 통일.
"""
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MakeScaffoldGeneric
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix)
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)
SEED = 42
METAL_NUMS = frozenset({3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,33,
    37,38,39,40,41,42,44,45,46,47,48,49,50,55,56,72,73,74,75,76,77,78,79,80,81,82,83})

def find_smi(df):
    for c in ["canonical_smiles","SMILES_raw","SMILES","_can","smiles"]:
        if c in df.columns: return c
    raise KeyError(f"No SMILES column in {list(df.columns)}")

def to_can(s):
    if pd.isna(s) or str(s).strip()=="": return None
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def file_hash(path):
    """SHA-256 of file for reproducibility."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def resolve_conflicts(df, strategy="conservative"):
    sc = find_smi(df); df = df.copy(); df["_can"] = df[sc].apply(to_can)
    valid = df[df["_can"].notna()].copy()
    grp = valid.groupby("_can")["label"].agg(["nunique","count","sum"])
    csmi = set(grp[grp["nunique"]>1].index)
    nc = valid[~valid["_can"].isin(csmi)]
    cr = valid[valid["_can"].isin(csmi)]
    if strategy=="conservative": cleaned=nc.copy()
    elif strategy=="positive_priority":
        res=[]
        for cs in csmi: r=cr[cr["_can"]==cs].iloc[0].copy(); r["label"]=1; res.append(r)
        cleaned=pd.concat([nc,pd.DataFrame(res)],ignore_index=True) if res else nc.copy()
    elif strategy=="majority_vote":
        res=[]
        for cs in csmi:
            sub=cr[cr["_can"]==cs]; np_=(sub["label"]==1).sum(); nn_=(sub["label"]==0).sum()
            if np_==nn_: continue
            r=sub.iloc[0].copy(); r["label"]=1 if np_>nn_ else 0; res.append(r)
        cleaned=pd.concat([nc,pd.DataFrame(res)],ignore_index=True) if res else nc.copy()
    else: raise ValueError(strategy)
    cleaned=cleaned.drop_duplicates(subset=["_can"],keep="first")
    rpt={"strategy":strategy,"n_original":len(df),"n_conflicts":len(csmi),"n_after":len(cleaned),
         "pos_rate_before":round(df["label"].mean(),4),"pos_rate_after":round(cleaned["label"].mean(),4) if len(cleaned)>0 else 0}
    return cleaned, cr, rpt

def assign_scaffolds(df, n_clusters=50, seed=SEED):
    sc=find_smi(df); df=df.copy(); scaffolds=[]; stypes=[]
    for smi in df[sc]:
        mol=Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None: scaffolds.append("INVALID"); stypes.append("invalid"); continue
        try:
            s=MurckoScaffoldSmiles(mol=mol,includeChirality=False)
            if s and s.strip():
                sm=Chem.MolFromSmiles(s)
                scaffolds.append(Chem.MolToSmiles(MakeScaffoldGeneric(sm),canonical=True) if sm else s)
                stypes.append("cyclic")
            else: scaffolds.append(f"__ACYC_{len(scaffolds)}"); stypes.append("acyclic")
        except: scaffolds.append(f"__ERR_{len(scaffolds)}"); stypes.append("error")
    df["scaffold"]=scaffolds; df["scaffold_type"]=stypes
    amask=df["scaffold_type"]=="acyclic"; na=amask.sum()
    if na > n_clusters:
        fps=[]
        for smi in df.loc[amask,sc]:
            mol=Chem.MolFromSmiles(str(smi))
            if mol:
                fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                a=np.zeros(1024,dtype=np.float32); DataStructs.ConvertToNumpyArray(fp,a); fps.append(a)
            else: fps.append(np.zeros(1024,dtype=np.float32))
        nc=min(n_clusters,na//2)
        if nc>=2:
            labels=AgglomerativeClustering(n_clusters=nc,metric="euclidean",linkage="ward").fit_predict(np.array(fps))
            df.loc[amask,"scaffold"]=[f"ACYC_C{l}" for l in labels]
    df["scaffold_group"]=df["scaffold"]; return df

def fixed_split(df, ratio=0.8, seed=SEED):
    if "scaffold_group" not in df.columns: df=assign_scaffolds(df,seed=seed)
    rng=np.random.RandomState(seed); scfs=defaultdict(list)
    for idx,row in df.iterrows(): scfs[row["scaffold_group"]].append(idx)
    items=list(scfs.items()); rng.shuffle(items); items.sort(key=lambda x:len(x[1]),reverse=True)
    n_target=int(len(df)*ratio); train_idx=[]; test_idx=[]
    for s,idxs in items:
        if len(train_idx)+len(idxs)<=n_target: train_idx.extend(idxs)
        else: test_idx.extend(idxs)
    df=df.copy(); df["split"]="unassigned"
    df.loc[train_idx,"split"]="train"; df.loc[test_idx,"split"]="test"
    tr=df[df["split"]=="train"]; te=df[df["split"]=="test"]
    bal={"train_n":len(tr),"test_n":len(te),
         "train_pos":round(tr["label"].mean(),4),"test_pos":round(te["label"].mean(),4),
         "diff":round(abs(tr["label"].mean()-te["label"].mean()),4),
         "scaffold_overlap":len(set(tr["scaffold_group"])&set(te["scaffold_group"])),
         "n_groups":df["scaffold_group"].nunique()}
    return df, bal

def apply_scenario(df_split, scenario, flags=None):
    df=df_split.copy()
    if scenario=="raw_all": pass
    elif scenario=="no_metal":
        if flags is not None and "has_metal" in flags.columns:
            m=flags.reindex(df.index)["has_metal"].fillna(False).values; df=df[~m].copy()
    elif scenario=="salt_stripped":
        if flags is not None and "smiles_canonical" in flags.columns:
            sc=find_smi(df); can=flags.reindex(df.index)["smiles_canonical"]
            v=can.notna(); df.loc[v,sc]=can[v].values
    elif scenario=="metal_as_feature":
        if flags is not None and "has_metal" in flags.columns:
            df["has_metal_feature"]=flags.reindex(df.index)["has_metal"].fillna(False).astype(int).values
    return df[df["split"]=="train"].copy(), df[df["split"]=="test"].copy()

# ═══════════════════════════════════════════════════
#  TRUE Repeated Scaffold CV
#  수정: np.arange(N) % n_folds (position-based)
#  perm % n_folds는 group identity 의존 → 항상 같은 fold (버그)
# ═══════════════════════════════════════════════════

def repeated_cv(X, y, groups, model_fn, n_folds=3, n_repeats=3, seed=SEED):
    ug = np.unique(groups); ng = len(ug); af = min(n_folds, ng)
    if af < 2: return {}
    all_m = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(seed + rep * 7919)
        perm = rng.permutation(ng)
        # POSITION-BASED fold assignment (NOT perm % af)
        fold_assign = np.arange(ng) % af
        g2f = dict(zip(ug[perm], fold_assign))
        sf = np.array([g2f[g] for g in groups])
        for f in range(af):
            vm = sf==f; tm = ~vm
            if vm.sum()==0 or tm.sum()==0: continue
            Xt,yt = X[tm],y[tm]; Xv,yv = X[vm],y[vm]
            if len(set(yt))<2 or len(set(yv))<2: continue
            mdl = model_fn(); mdl.fit(Xt,yt)
            yp = mdl.predict_proba(Xv)[:,1]; yd = (yp>=0.5).astype(int)
            mcc = matthews_corrcoef(yv,yd)
            try: roc = roc_auc_score(yv,yp)
            except: roc = np.nan
            all_m.append({"rep":rep,"fold":f,"mcc":mcc,"roc_auc":roc,
                          "bacc":balanced_accuracy_score(yv,yd),"n_val":len(yv)})
    if not all_m: return {}
    mdf = pd.DataFrame(all_m)
    # Verify fold uniqueness
    fold_sigs = set()
    for rep in range(n_repeats):
        rng2 = np.random.RandomState(seed + rep * 7919)
        p2 = rng2.permutation(ng)
        g2f2 = dict(zip(ug[p2], np.arange(ng) % af))
        fold_sigs.add(tuple(g2f2[g] for g in ug[:min(10,ng)]))
    result = {"_n_unique_fold_assignments": len(fold_sigs)}
    for m in ["mcc","roc_auc","bacc"]:
        v = mdf[m].dropna()
        if len(v)>0:
            result[m]={"mean":round(v.mean(),4),"std":round(v.std(),4),
                       "lo":round(v.quantile(0.025),4),"hi":round(v.quantile(0.975),4),"n":len(v)}
    return result

def bootstrap_ci(yt, yp, ypr, n=200, seed=SEED):
    rng=np.random.RandomState(seed); sz=len(yt)
    boots={"mcc":[],"bacc":[],"roc_auc":[],"sens":[],"spec":[],"brier":[]}
    for _ in range(n):
        idx=rng.choice(sz,sz,replace=True)
        yt_,yp_,ypr_=yt[idx],yp[idx],ypr[idx]
        if len(set(yt_))<2: continue
        tn,fp,fn,tp=confusion_matrix(yt_,yp_,labels=[0,1]).ravel()
        boots["mcc"].append(matthews_corrcoef(yt_,yp_))
        boots["bacc"].append(balanced_accuracy_score(yt_,yp_))
        boots["sens"].append(tp/(tp+fn) if (tp+fn)>0 else 0)
        boots["spec"].append(tn/(tn+fp) if (tn+fp)>0 else 0)
        boots["brier"].append(brier_score_loss(yt_,ypr_))
        try: boots["roc_auc"].append(roc_auc_score(yt_,ypr_))
        except: pass
    out={}
    for m,vals in boots.items():
        if vals:
            a=np.array(vals)
            out[m]={"mean":round(a.mean(),4),"lo":round(np.percentile(a,2.5),4),"hi":round(np.percentile(a,97.5),4)}
    return out

def compute_ad(tr_fp, te_fp, threshold=0.3, max_tr=300, seed=42):
    rng=np.random.RandomState(seed)
    tr=tr_fp[rng.choice(len(tr_fp),min(max_tr,len(tr_fp)),replace=False)] if len(tr_fp)>max_tr else tr_fp
    ms=np.zeros(len(te_fp))
    for i in range(len(te_fp)):
        sims=[]
        for j in range(len(tr)):
            inter=np.sum(te_fp[i]*tr[j]); union=np.sum(te_fp[i])+np.sum(tr[j])-inter
            sims.append(inter/union if union>0 else 0)
        ms[i]=max(sims) if sims else 0
    in_ad=ms>=threshold
    return {"coverage":round(in_ad.mean(),4),"n_in":int(in_ad.sum()),"n_out":int((~in_ad).sum()),"mean_sim":round(ms.mean(),4)}

def calibration(yt, ypr, bins=10):
    brier=brier_score_loss(yt,ypr); edges=np.linspace(0,1,bins+1); ece=0.0
    for i in range(bins):
        mask=(ypr>=edges[i])&(ypr<edges[i+1])
        if mask.sum()==0: continue
        ece+=mask.sum()/len(yt)*abs(yt[mask].mean()-ypr[mask].mean())
    return {"brier":round(brier,4),"ece":round(ece,4)}

def domain_confounding(df, col="domain"):
    if col not in df.columns: return {"has_domain":False}
    rpt={"has_domain":True,"domains":{}}
    for d,g in df.groupby(col):
        rpt["domains"][d]={"n":len(g),"pos":int((g["label"]==1).sum()),"pos_rate":round(g["label"].mean(),4)}
    doms=list(rpt["domains"].keys())
    if len(doms)==2:
        r0=rpt["domains"][doms[0]]["pos_rate"]; r1=rpt["domains"][doms[1]]["pos_rate"]
        p0=1 if r0>0.5 else 0; p1=1 if r1>0.5 else 0
        cor=sum(1 for _,row in df.iterrows() if (p0 if row[col]==doms[0] else p1)==row["label"])
        rpt["domain_acc"]=round(cor/len(df),4); rpt["enrichment"]=round(max(r0,r1)/max(min(r0,r1),0.001),1)
    return rpt

def leave_domain_out(df, col, feat_fn, model_fn, ep):
    if col not in df.columns: return []
    doms=df[col].unique(); results=[]
    for td in doms:
        train=df[df[col]!=td].copy(); test=df[df[col]==td].copy()
        if len(train)<10 or len(test)<10 or len(set(train["label"]))<2 or len(set(test["label"]))<2: continue
        Xr,yr,_=feat_fn(train,ep); Xe,ye,_=feat_fn(test,ep)
        mdl=model_fn(); mdl.fit(Xr,yr)
        yp=mdl.predict_proba(Xe)[:,1]; yd=(yp>=0.5).astype(int)
        ci=bootstrap_ci(ye,yd,yp,n=500)
        results.append({"train_dom":",".join([d for d in doms if d!=td]),"test_dom":td,
            "train_n":len(yr),"test_n":len(ye),"test_pos":round(ye.mean(),4),
            **{k:v["mean"] for k,v in ci.items()},**{f"{k}_lo":v["lo"] for k,v in ci.items()},
            **{f"{k}_hi":v["hi"] for k,v in ci.items()}})
    return results

def cross_endpoint(datasets):
    maps={}
    for ep,df in datasets.items():
        sc=find_smi(df); m={}
        for _,row in df.iterrows():
            cs=to_can(row[sc])
            if cs: m[cs]=int(row["label"])
        maps[ep]=m
    results=[]; eps=list(datasets.keys())
    for i in range(len(eps)):
        for j in range(i+1,len(eps)):
            e1,e2=eps[i],eps[j]; m1,m2=maps[e1],maps[e2]
            ov=set(m1)&set(m2); conc=sum(1 for s in ov if m1[s]==m2[s])
            kappa=np.nan
            if len(ov)>1:
                try:
                    from sklearn.metrics import cohen_kappa_score
                    kappa=cohen_kappa_score([m1[s] for s in ov],[m2[s] for s in ov])
                except: pass
            results.append({"ep1":e1,"ep2":e2,"overlap":len(ov),
                "overlap_pct":round(len(ov)/max(len(m2),1)*100,1),
                "concordant":conc,"discordant":len(ov)-conc,
                "kappa":round(kappa,4) if not np.isnan(kappa) else None})
    return pd.DataFrame(results)

def interpret_result(row):
    """invivo 등 소표본에서 과대해석 방지를 위한 자동 주석"""
    notes = []
    if row.get("test_n",0) > 0:
        n_pos = round(row["test_n"] * row.get("test_pos_rate", row.get("test_pos",0)))
        if n_pos < 15:
            notes.append(f"CAUTION: only ~{n_pos} positives in test set — interpret as exploratory")
    mcc = row.get("mcc",0); cv_mcc = row.get("cv_mcc_mean")
    if cv_mcc is not None and not pd.isna(cv_mcc):
        gap = abs(mcc - cv_mcc)
        if gap > 0.2:
            notes.append(f"WARNING: test-CV gap={gap:.3f} — possible split luck or CV instability")
    return "; ".join(notes) if notes else "OK"

# Aliases
smi_col = find_smi

# ─── Threshold Tuning (train-based) ───────────

def tune_threshold(y_train_true, y_train_proba):
    """Train set에서 MCC 최대화 threshold를 찾는다. Test에서는 이 값을 그대로 적용."""
    best_t, best_mcc = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (y_train_proba >= t).astype(int)
        if len(set(pred)) < 2: continue
        m = matthews_corrcoef(y_train_true, pred)
        if m > best_mcc:
            best_mcc = m; best_t = t
    return round(best_t, 2), round(best_mcc, 4)
