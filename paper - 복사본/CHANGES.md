# Revision Summary

이 폴더는 peer review 의견을 반영한 수정본입니다. 원본 대비 변경된 점과 후속 작업이 필요한 항목을 정리합니다.

---

## v3 추가 (v12.1 raw 결합 + Table 5 CI + Supplementary CSV 패키지)

`20260330_101505_v12.1.zip`에서 발견된 raw outputs를 반영해 추가 보강했습니다.

### 핵심 변경

1. **Table 5에 95% bootstrap CI 추가** (`3. results.tex`).
   - `ames_leave_domain_out.csv`에서 추출한 `mcc_lo` / `mcc_hi` 값 그대로 반영.
   - 표 캡션에 "MCC values are reported with 95% bootstrap confidence intervals (500 iterations)" 명시.
   - 표 footer에서 Table S9를 raw source로 가리키도록 cross-reference 추가.

2. **Methods의 bootstrap CI 설명 확장** (`2. methods.tex`).
   - 기존에는 Level 1만 bootstrap CI라고 적혀 있었음.
   - 이제 Level 3에서도 동일하게 보고된다는 점 명시: "As in Level 1, MCC and the full panel of secondary metrics… are reported with 95% bootstrap CIs (500 iterations) on the LDO test partition."

3. **Supplementary Table S9 신설** (`9. supplementary.tex`).
   - 모든 알고리즘 × 두 LDO 방향에 대한 raw metrics (MCC, BAcc, ROC-AUC, sens, spec, Brier) + 각각의 95% CI 표기.
   - Brier score asymmetry (drug→ind 0.117–0.127 vs ind→drug 0.241–0.464)에 대한 footer comment로 prior shift 직관적으로 전달.
   - "RF의 cross-source 우위는 point-estimate noise가 아니다" 라는 통계적 진술을 CI 비분리(non-overlap)로 직접 정당화.
   - Raw CSV: `ames_leave_domain_out.csv`.

4. **Supplementary CSV 파일명을 v12.1 원본명으로 갱신** (`9. supplementary.tex`).
   - 기존: `table_s2_full_results.csv`, `table_s3_hp_tuning.csv`, `table_s4_ad_coverage.csv`, `table_s5_cross_endpoint.csv`, `table_s6_shap.csv`.
   - 갱신: `table_main_default_threshold.csv`, `table_hp_tuning_comparison.csv`, `ad_coverage_summary.csv`, `cross_endpoint_overlap.csv`, `shap_all_experiments.csv` (+ per-endpoint top-20, statistical comparisons, learning curves, locked cascade, CV selection state).
   - Submission 시 v12.1 zip의 파일을 그대로 첨부 가능.

### 별도 deliverable: `supplementary_csvs.zip`

19개의 v12.1 CSV + 사용자의 `table_s8_ldo_threshold_analysis.csv` + `README.md` (각 파일이 paper의 어느 표/그림에 매핑되는지 명시).

`README.md`에 특히 두 가지 주의 명시:
- `table_supplementary_threshold_tuning.csv`는 in-domain threshold tuning이지 Table S7의 source가 아님 — 혼동 방지를 위한 explicit disclaimer.
- `ames_leave_domain_out.csv`는 v12.1에 이미 존재했으나 원본 draft에서는 CI가 surface되지 않았다는 설명, 이번 revision에서 authoritative source로 격상됨을 명시.

### 풀린 의문점

- ✅ Table 5의 raw source: `ames_leave_domain_out.csv` (정확히 일치).
- ✅ 메인 본문 Table 2/3 raw: `table_main_default_threshold.csv` (정확히 일치).
- ⚠️ Table S7 (FP-1024 LDO threshold) raw: v12.1에 직접 매핑되는 파일 없음. 별도 분석이거나 다른 파이프라인. README에 disclosure.
- ⚠️ Table S8 (compact LDO threshold, reduced partition) raw: `table_s8_ldo_threshold_analysis.csv` (사용자 별도 계산). Reduced N (5027/5821)의 출처는 여전히 TODO로 남음.

### 컴파일 확인

`pdflatex` × 3 + `bibtex` 통과. 32페이지, undefined citation 0건. Table 5의 CI 형식과 Table S9 행이 PDF에 정상 렌더링.

---

## v2 추가 (Table S8 복원)

이전 버전에 있던 `table_s8_ldo_threshold_analysis.csv`를 본문에 복원했습니다. 이 표가 추가되면서 일부 진술이 정량적으로 강화되었습니다.

### 새로 들어간 부분

- **`9. supplementary.tex`**: 새 subsection "Table S8. LDO threshold-tuning analysis on the locked-configuration features (gap decomposition)" 추가. Table S7 바로 뒤에 위치. 여섯 개 행 (XGB/LGBM/RF × 두 방향), gap decomposition 칼럼 (Δ_total, Δ_prior, Δ_residual, % explained) 포함. Plateau range는 oracle MCC ± 0.030으로 표기 (CSV 원본 데이터에서 plateau_lo/plateau_hi 그대로 사용).
- **`4. discussion.tex`**: Threshold decomposition 단락 전면 재작성. 기존에는 S7 결과만 인용하면서 "modest improvements"라는 정성적 진술이었는데, 이제 S8을 헤드라인으로 승격해 **"prior shift가 LDO gap의 30.9–35.3%를 설명, 나머지 ~67%는 residual"**이라는 정량 진술로 교체. RF robustness note도 S7/S8/Table 5 세 분석을 모두 비교하는 형태로 재작성. 결론: Tables 5와 S8이 broadly consistent (RF ind→drug 0.325 vs 0.350), S7만 다름 → "feature-conditional" 해석.
- **`0. abstract.tex`**: "oracle MCC plateauing at 0.27–0.36" → "prior shift accounts for roughly one-third (30.9–35.3%) of the LDO performance gap, leaving a substantial residual that no recalibration can eliminate". 더 정확하고 정량적.
- **`5. conclusion.tex`**: 같은 방식으로 "only part of this gap" → "roughly one-third (30.9–35.3%) of the LDO gap" 정량화.

### 알려진 불일치 (TODO 처리)

S8과 Table 5는 **default threshold에서 LDO 방향성이 반대**:

| | drug→ind | ind→drug |
|---|---|---|
| Table 5 (full N: 6416/7144) | XGB 0.234 (쉬움) | XGB 0.122 (어려움) |
| S8 (reduced N: 5027/5821) | XGB 0.118 (어려움) | XGB 0.389 (쉬움) |

S8의 train_n이 Table 5와 다르므로 **다른 partition** (cross-source dedup 또는 scaffold-aware split이 적용된 듯) 입니다. S8의 supplementary 본문에 "Methodological note on partitioning"으로 이 점을 disclose해 두었고, TODO에 정확한 rule을 코드에서 확인하라고 표시했습니다.

이 불일치를 reviewer가 짚을 가능성이 높은데, 두 가지 해결책 중 하나를 선택해야 합니다:
1. (이상적) Table 5 또는 S8 중 하나를 다른 partition으로 재계산해서 둘이 같은 partition을 쓰게 만들기.
2. (현실적) 위 methodological note를 충분한 disclosure로 두고, 두 표가 다른 사용 목적임을 명시 (Table 5 = default-threshold 메인 결과, S8 = 동일 feature/threshold-tuning 분해).

저는 두 번째를 default로 두는 LaTeX를 작성했지만, 첫 번째가 가능하면 그쪽이 훨씬 깨끗합니다.

### 영향 받은 다른 부분

이전 v1에서 적었던 "Table S7 vs Table 5 RF robustness 불일치" TODO는 S8이 들어오면서 부분적으로 해결됐습니다. S8의 RF ind→drug = 0.350이 Table 5의 0.325와 거의 일치하므로, "RF cross-source robustness가 진짜다"라는 결론은 유지됩니다. 단, S7과 Table 5/S8 사이의 불일치는 여전히 "feature-conditional" 해석으로 남습니다.

---

## v1 변경사항 (이전 라운드)

### 1. 직접 수정한 항목 (텍스트로 처리 가능)

#### main.tex
- **제목 변경**: "Data Source Heterogeneity, Not Algorithm Choice, Drives Performance Gaps in QSAR for Mutagenicity" → **"Accounting for Data Source Heterogeneity in Ames Mutagenicity QSAR Development and Evaluation"**.
  - 이전 제목은 본문의 cross-source 결과 (RF 0.325 vs XGB 0.122)와 충돌함. Reviewer가 거의 확실히 짚을 항목.
  - 이 제목은 사용자가 이전에 finalize했던 버전으로 복귀시킴.
- `\input{6. data_availability.tex}` 주석 해제.

#### 0. abstract.tex
- "increasingly relied upon" → "are accepted as supporting evidence under" (정확한 표현).
- 225 configurations 계산 명확화: "seven algorithms, five feature representations" 표현이 7×5×3 ≠ 225라서 오해 소지가 있었음. "six tabular algorithms paired with four feature representations, plus a graph convolutional network with graph features, under three preprocessing scenarios"로 수정.
- Cascade에서 빠져있던 0.470 (scaffold-split test) 추가.
- Algorithm 부분: "uneven robustness"만 적던 것을 "in-domain 차이는 작지만 cross-source에서는 크다"는 두 단계 진술로 교체. 새 제목과 일관됨.
- "as a reporting standard" → "as a useful addition to current reporting practices" (단일 연구로 standard 제안하는 톤 다운).

#### 1. introduction.tex
- Contribution (ii)와 (iii) 재서술.
  - (ii): prior shift 한정구를 결과 문장 안으로 통합.
  - (iii): "in-domain 차이는 작지만 cross-source 차이는 크다"를 명시. 새 제목과 일관됨.

#### 2. methods.tex
- **Drug-source data 출처 명시**: Hansen 2009 + Honma 2019 + Furuhama 2023, 그리고 "intentionally enriched for known mutagens for evaluation purposes"라고 enrichment 사실을 적시.
- **Klimisch filter 비대칭성 명시적 disclosure**: 산업 화학물질에만 적용되었다는 점을 본문과 limitation 양쪽에 명시.
- **Locked configuration 정당화 단락 추가**: XGB 선택 이유 (most widely reported in recent Ames benchmarks), commitment timing ("before LDO experiments were run"), Level 3에서 알고리즘 sensitivity 검사한다는 점 명시.
- **Scaffold clustering 디테일 추가**: average linkage, 1024-bit Morgan FP, 1−Tanimoto distance, threshold 0.6 (사실 확인 필요 — 아래 TODO 참조).
- **Hyperparameter search 디테일**: `n_iter=50`, 5 folds, scaffold groups, 그리고 Table S3 참조.
- **Conflict resolution sensitivity 언급** (TODO로 표시).
- **Cross-source duplicate handling 언급** (TODO로 표시).

#### 3. results.tex
- **GCN 비교 공정성 캐비어 추가**: GCN-XGB 차이가 representation과 algorithm을 conflate한다는 점 명시.
- **Naive source classifier 함의 확장**: 단순한 숫자 보고에서 "published Ames model performance가 source membership을 학습하고 있을 수 있다"는 explicit한 함의 진술로 확장.
- **In-vivo "unreliable" 표현 톤 다운**: 모델 한계와 sample size 한계를 분리.
- **In-vitro CA 단락 신설**: 메인 본문에서 거의 다뤄지지 않았던 in-vitro 결과에 대해 한 단락 추가.

#### 4. discussion.tex
- **Selection bias 가설 수정**: "drug candidates are designed for biological activity" 논리 약함을 "Hansen benchmark는 known mutagen으로 의도적으로 enriched"라는 더 정확한 메커니즘으로 교체.
- **Lipinski/Veber 인용 제거**: drug-likeness 논문이라 "chemical space difference" 근거로 부적절. TODO로 대체 인용 후보 적시.
- **Table S7 vs Table 5 불일치 명시적 다루기** (v2에서 S8 추가로 부분 해결됨).
- **In-vivo subsection 톤 다운**.
- **Practical Recommendations 톤 다운**: "reporting standards are proposed" → "useful additions to current reporting practices".

#### 5. conclusion.tex
- 새 제목과 일관되도록 재서술.
- Naive source classifier 함의 한 문장 추가.
- "minimum reporting standard" → "useful complement to existing practices".

#### 6. data_availability.tex
- 더 구체적인 산출물 명시.
- Author email 채움 (`hpjeon@ks.ac.kr`).

#### 9. supplementary.tex
- Table S7 footnote 강화.

---

## 2. 후속 작업이 필요한 TODO 항목

각 .tex 파일에서 `% TODO[smyoon]:`로 검색 가능합니다.

### 분석/사실 확인 필요

1. **★ Table S8 partition rule 확인** (`9. supplementary.tex`)
   - Reduced N (5027/5821)이 어떤 rule로 만들어졌는지 코드에서 확인 후 본문 채우기.
   - Cross-source dedup인지, scaffold-aware partitioning인지, 다른 규칙인지.
   - 이상적으로는 Table 5와 S8을 같은 partition에서 재계산해 둘이 일치하도록 만들기.

2. **Cross-source duplicate count [N]** (`2. methods.tex`)

3. **Conflict resolution sensitivity analysis** (`2. methods.tex`)
   - 실제 안 했으면 그 문장 삭제 후 limitation으로 옮기기.

4. **Scaffold clustering parameters** (`2. methods.tex`)
   - average linkage, distance threshold 0.6은 추정값. 코드와 대조.

### 형식적 보완

5. **Lipinski/Veber 대체 인용** (`4. discussion.tex`)

6. **GitHub URL과 Zenodo DOI** (`6. data_availability.tex`)

7. **Acknowledgments** (`6. data_availability.tex`)

---

## 3. 컴파일 확인

`pdflatex` → `bibtex` → `pdflatex` × 2 통과. 31페이지, undefined citation 경고 없음. figures/ 안의 PDF는 placeholder이므로 실제 그림으로 교체 필요.

