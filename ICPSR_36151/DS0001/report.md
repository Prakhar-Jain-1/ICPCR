# Index Reference — Complete Guide to Every Index in This Codebase

> **How to read this document:** Each section covers one index or variable group.
> For every index you will find: **(1) What it is**, **(2) What it measures and why**,
> **(3) What a high vs low score means practically**, and **(4) Exactly how it is calculated**
> with formulas and variable-level detail.

---

## 1. Physical Health Index (PHI) — `PHYSICAL_HEALTH_IDX`

**What it is:** The primary outcome variable. A composite score of physical illness burden built from three sub-scores covering acute illness, chronic disease, and functional ability.

**What it measures and why:** PHI operationalises the idea that health is not binary — the difference between someone ill for 2 days and someone with 20 days of sickness, active hypertension, and difficulty walking is enormous. PHI quantifies that gradient on a common scale. It is grounded in Sen's capability approach: health as the ability to achieve functioning across all three dimensions simultaneously.

**What a score means:**
- **0.00** = No recorded health events (most of the general population)
- **0.05** = Typical score for a sick person in the general population (UP mean)
- **0.13** = Mean PHI in the sick subgroup (UP) — moderate chronic + acute burden
- **0.50+** = Severe illness across multiple dimensions (rare, < 5% of sick subgroup)
- Higher always means worse health burden

**Formula:** `PHI = 0.30 × STM_score + 0.40 × CDI_score + 0.30 × ADL_score`

| Sub-score | Weight | IHDS-II variables | Construction |
|-----------|--------|-------------------|-------------|
| STM_score | **30%** | SM3, SM4, SM5, SM11 | `0.40×(SM3/30) + 0.20×(SM11/30) + 0.20×SM4 + 0.20×SM5` |
| CDI_score | **40%** | MB3,5,6,7,10,12,14,17 | `Σ(2×active + 1×cured) / (2 × n_vars)` |
| ADL_score | **30%** | AD3, AD4, AD5, AD6, AD8, AD9 | `Σ ordinal(0/1/2) / 12` |

CDI gets the highest weight (40%) because chronic disease drives the largest share of disability-adjusted life years in India's NCD transition. STM and ADL share the remainder — they capture dimensions that chronic disease variables alone miss.

**Key results:** UP mean = 0.051 (full pop.), 0.131 (sick subgroup). GJ = 0.020 / 0.093. MH = 0.020 / 0.109. The sick subgroup (PHI > 0) is 39% of UP, 21.7% of GJ, 17.8% of MH.

---

## 2. Short-Term Morbidity Score (STM_score)

**What it is:** The acute illness sub-score within PHI. A *flow* measure — it captures illness burden in the past 30 days, not a permanent stock of conditions.

**What it measures and why:** Acute illness is seasonal and episodic. It captures current health crises, fever episodes, and disability days that chronic disease variables miss because those are recorded only when a condition has been diagnosed. STM is sensitive to recent healthcare access — someone who received effective treatment last week has a lower STM than someone whose fever continued untreated for three weeks.

**What a score means:**
- **0.00** = No illness, fever, cough, or disability days in the past 30 days
- **0.20** = Roughly 1 week ill with fever (e.g. 7 days ill + fever flag)
- **1.00** = Maximum: 30 sick days, 30 disability days, fever AND cough (theoretical maximum)
- UP mean = 0.142, GJ = 0.037, MH = 0.045 (full population) — UP's advantage here is stark

**Formula:** `0.40×(SM3/30) + 0.20×(SM11/30) + 0.20×SM4 + 0.20×SM5`

- **SM3**: Days ill in past 30 days (continuous, clipped at 30 before dividing)
- **SM11**: Disability days — unable to carry out usual activities (continuous, 0–30)
- **SM4**: Fever in past 30 days (binary, 0/1)
- **SM5**: Cough in past 30 days (binary, 0/1)

Duration (SM3) gets double the weight of each symptom flag because it captures the severity gradient continuously — a person ill for 20 days is far more burdened than one ill for 2 days, and a binary fever flag cannot distinguish them.

---

## 3. Chronic Disease Index (CDI_score)

**What it is:** The chronic disease sub-score within PHI. A *stock* measure — it captures the accumulated burden of diagnosed long-term conditions, not recent episodes.

**What it measures and why:** In India's epidemiological transition, NCDs (hypertension, diabetes, heart disease, asthma) have overtaken infectious disease as the dominant source of DALYs. CDI captures this persistent, compounding burden. Unlike STM, a high CDI score is unlikely to resolve within weeks — it reflects years of disease accumulation.

**What a score means:**
- **0.00** = No chronic conditions diagnosed
- **0.10** = One active chronic condition (e.g. hypertension)
- **0.30** = Two–three active conditions with some cured conditions (high burden)
- **0.56** = Maximum observed (UP) — multiple active tier-1 conditions simultaneously
- Higher = more chronic disease burden

**Formula:** `Σ(2 × active + 1 × cured) / (2 × n_vars)`

Each MB variable is coded: **0** = No condition, **1** = Cured, **2** = Active. The raw sum is divided by `2 × n_vars` (the theoretical maximum if all conditions were active) to normalise to [0,1].

**Why "Cured" = 1 (not 0):** In India, "cured" of TB, hypertension, or diabetes rarely means the condition is fully resolved. It typically means a drug course was completed or medication was prescribed at some point. Setting Cured=0 would systematically undercount burden among people who accessed healthcare and would create a perverse incentive in the index — someone who never sought care (No=0) would appear equally healthy to someone with managed hypertension (Cured=1). Assigning half-weight preserves this distinction.

**Variables included:**
- Tier 1 (core burden): MB5=Hypertension, MB7=Diabetes, MB6=Heart disease, MB10=Asthma, MB17=Other long-term
- Tier 2 (substantive): MB3=Cataract, MB4=TB, MB12=Paralysis, MB14=Mental illness

---

## 4. Activities of Daily Living Score (ADL_score)

**What it is:** The functional limitation sub-score within PHI. A *capability* measure — it assesses what a person can actually do, not what conditions they have been diagnosed with.

**What it measures and why:** Two people with identical chronic disease burdens may lead entirely different functional lives depending on age, prior treatment, and socioeconomic support. A 70-year-old farmer with hypertension who can still walk 1km differs fundamentally from one who cannot leave the house. The ADL score operationalises Sen's capability approach — health is the ability to achieve functioning — and is the only PHI sub-score that directly captures independence.

**What a score means:**
- **0.00** = No difficulty with any of the 6 activities
- **0.17** = Difficulty with one activity (e.g. walking 1km with difficulty, no others)
- **0.50** = Significant limitation across three activities
- **1.00** = Unable to perform all 6 activities (maximum limitation)
- UP mean = 0.009, GJ = 0.014, MH = 0.010 (full pop.) — ADL rare at population level but concentrated in elderly and disabled

**Formula:** `Σ ordinal(0/1/2) / 12`

Each AD variable follows the WHO WHODAS 2.0 ordinal scale:
- **0** = No difficulty
- **1** = Can do with difficulty
- **2** = Unable to do

Activities: **AD3** = Walk 1km, **AD4** = Go to toilet, **AD5** = Dress/undress, **AD6** = Hear conversation, **AD8** = See distant objects, **AD9** = See near objects.

The ordinal structure is preserved rather than binarised — "with difficulty" is not the same as "unable". AD7 (speaking) was excluded due to lowest prevalence (0.9%) and near-zero variance contribution.

---

## 5. Healthcare Access Sub-index (HAS) — `HEALTH_ACCESS_IDX`

**What it is:** One of three pillars of the MHI. HAS measures systemic barriers that prevent ill people from getting care — financial catastrophe, untreated morbidity, and geographic distance to provider.

**What it measures and why:** Being sick is not the same as being able to get treatment. HAS captures the gap between illness and care, independently of how sick someone is. A person with high PHI but low HAS is very ill but well-served. A person with low PHI but high HAS faces barriers even for minor illness. PHI alone would miss this — Gujarat's lower PHI coexisted with the highest untreated morbidity (20.9%), a pattern invisible without HAS.

**What a score means:**
- **0.00** = No access barriers — all three dimensions = 0
- **0.08** = UP mean: mostly geographic barriers (moderate distance to provider)
- **0.13** = GJ mean: highest — driven by high untreated morbidity AND geographic distance
- **0.11** = MH mean: intermediate
- **0.67** = Maximum possible (all three barriers simultaneously present)
- Higher = worse access

**Formula:** `HAS = (ACC_dist_score + ACC_gap_score + ACC_fin_score) / 3`

| Component | Variable | How coded |
|-----------|---------|-----------|
| `ACC_dist_score` | **SM14B** (location of first provider) | Ordinal: Home/Same village=0.0, Another village=0.25, Other town=0.50, District town=0.75, Metro/Abroad=1.0 |
| `ACC_gap_score` | **UNTREATED** flag | 1 if ill (PHI>0) but no treatment sought for acute OR chronic condition |
| `ACC_fin_score` | **CAT_EXP** flag | 1 if OOP expenditure ≥ 10% of annual household consumption (WHO threshold) |

**SM14B fix (updated from previous version):** The geographic barrier previously used SM16 (which codes treatment *type*, not location). This has been corrected — SM14B captures the actual location of the first provider visited, giving a meaningful access-distance proxy. This change increased HAS values substantially across all states (UP: 0.045→0.078, GJ: 0.108→0.133, MH: 0.087→0.111) and raised all MHI scores accordingly.

**Key finding:** Gujarat's HAS (0.133) is highest despite having lower PHI than UP. The 20.9% untreated morbidity rate — 1 in 5 sick Gujaratis sought no treatment at all — is the key driver. This is the central finding that justifies the multidimensional index over PHI alone.

---

## 6. Social Vulnerability Index (SVI) — `SOCIAL_VULNERABILITY_IDX`

**What it is:** The third pillar of the MHI. SVI quantifies structural social disadvantage from gender and caste — the conversion factors that determine how the same income and illness translate into different health outcomes for different people.

**What it measures and why:** A SC/ST woman faces barriers to healthcare-seeking that a forward-caste man does not, even at identical income levels. These are not individual characteristics but structural positions embedded in social hierarchies. The SVI makes these structural modifiers explicit and quantifiable, grounding the MHI in the capability framework's recognition that conversion factors — not just resources — determine outcomes.

**What a score means:**
- **0.00** = Forward-caste male (minimum structural vulnerability)
- **0.25** = Forward-caste female OR SC/ST male (one dimension of disadvantage)
- **0.50** = OBC female OR SC/ST male — the most common score (median in all states)
- **0.75** = SC/ST female (both gender and caste disadvantage)
- **1.00** = Theoretical maximum (not attainable with current encoding)
- UP mean = 0.462, GJ = 0.449, MH = 0.419 — Maharashtra has slightly lower caste vulnerability

**Formula:** `SVI = (SOC_gender_score + SOC_caste_score) / 2`

- **SOC_gender_score** (from RO3): Female = 1.0, Male = 0.0
- **SOC_caste_score** (from ID13): Brahmin/Forward = 0.0, OBC/Other = 0.5, SC/ST = 1.0

**Missing caste** is filled with 0.5 (OBC neutral default) rather than 0 or NaN — to avoid attributing forward-caste status to unknown individuals and to prevent row deletion that would introduce selection bias.

---

## 7. Out-of-Pocket Cost Variables

**What they are:** A set of derived financial burden variables computed from IHDS-II medical expenditure questions. These are used as outcomes in cost regression (Cell 14) and as components of HAS (CAT_EXP feeds ACC_fin_score).

**Why they matter:** Being ill has two costs — the physical burden (PHI) and the financial consequence (OOP). These are partially independent: a rich person and a poor person may have identical PHI but face completely different financial catastrophe risk. The OOP variables quantify this financial dimension.

| Variable | Formula | Recall period | What it measures |
|----------|---------|---------------|-----------------|
| `OOP_STM` | SM18 + SM20 + SM21 | **30 days** | Doctor/hospital fees + medicines + travel for acute illness episode |
| `OOP_MB` | MB25 + MB27 + MB28 | **12 months** | Doctor fees + medicines + travel for chronic/major condition |
| `OOP_TOTAL` | OOP_STM + OOP_MB | Mixed | Combined financial burden across distinct episode types |
| `OOP_SHARE` | OOP_TOTAL / COTOTAL × 100 | Annual | Medical spending as % of annual household consumption |
| `CAT_EXP` | 1 if OOP_SHARE ≥ 10% | Annual | WHO catastrophic health expenditure binary flag |
| `UNTREATED` | 1 if ill but no care | — | Ill (PHI>0) but SM14A missing/invalid AND MB19=0 |
| `DAYS_LOST` | SM17 + MB18 | Mixed | Disability/absence days — proxy for indirect economic cost |

**Critical note on OOP_TOTAL:** SM/STM variables use a 30-day recall; MB variables use a 12-month recall. Adding them into OOP_TOTAL combines two different time windows into a single measure of "total recorded medical burden from distinct episode types." It should not be interpreted as total spending in a single period.

**Key results:** OOP_MB (chronic, 12-month) dwarfs OOP_STM (acute, 30-day) by 10–14× across all states. Private providers dominate: UP 90.3%, MH 86.1%, GJ 74.4%. CAT_EXP rates: GJ 11.3%, MH 10.4%, UP 8.9% — Gujarat has the highest catastrophic rate despite the lowest PHI.

---

## 8. Multidimensional Health Index (MHI) — `MULTIDIMENSIONAL_HEALTH_IDX`

**What it is:** The final composite index — the "triple burden" score. Integrates physical illness burden (PHI), access barriers (HAS), and structural disadvantage (SVI) into a single measure of overall health vulnerability.

**What it measures and why:** No single dimension tells the full story. PHI ranks GJ and MH identically on physical health (~0.020 full-pop mean). But when HAS is added, Gujarat's high untreated morbidity raises its composite score. When SVI is added, UP's higher caste and gender vulnerability reinforces its overall disadvantage. MHI is the only measure that captures all three simultaneously and ranks individuals by their overall health situation — not just how sick they are today.

**What a score means:**
- **0.00** = Best possible: no illness, no barriers, minimum structural vulnerability (theoretical)
- **0.131** = Salaried worker mean — best group (lowest across all three pillars)
- **0.187–0.201** = State means for sick subgroup (range across UP/GJ/MH)
- **0.210** = Inactive worker mean — worst group (driven by high SVI)
- **0.59** = Maximum observed (MH) — severe illness + barriers + social vulnerability all present
- Higher always = worse triple burden

**Formula:** `MHI = 0.50 × PHI + 0.25 × HAS + 0.25 × SVI`

**Updated results after SM14B fix:**

| State | N | PHI (50%) | HAS (25%) | SVI (25%) | MHI |
|-------|---|-----------|-----------|-----------|-----|
| Uttar Pradesh | 8,404 | 0.1309 | 0.0783 | 0.4622 | **0.2006** |
| Gujarat | 2,030 | 0.0933 | 0.1326 | 0.4485 | **0.1920** |
| Maharashtra | 2,846 | 0.1093 | 0.1110 | 0.4191 | **0.1872** |

**MHI by Occupation (pooled sick subgroup):**

| Group | PHI | HAS | SVI | MHI | Rank |
|-------|-----|-----|-----|-----|------|
| Inactive | 0.1238 | 0.0879 | 0.5061 | **0.2104** | 1 — worst |
| Agricultural | 0.1142 | 0.1194 | 0.3324 | **0.1700** | 2 |
| Wage | 0.1244 | 0.0965 | 0.2973 | **0.1607** | 3 |
| Self-employed | 0.1043 | 0.0988 | 0.2491 | **0.1392** | 4 |
| Salaried | 0.0944 | 0.1062 | 0.2482 | **0.1358** | 5 — best |

**The Inactive finding:** Inactive workers are worst not because they are sickest (PHI=0.124, comparable to Wage=0.124) but because SVI=0.506 — the highest of any group. Inactive is heavily female (housework) and SC/ST-weighted. This directly supports the capability framework: economic inactivity is not health-protective in India; it is a marker of structural exclusion.

**Why 50/25/25 weights:** PHI is the primary outcome — actual illness. HAS and SVI are structural modifiers that determine how illness is experienced, treated, and financially survived. The 2:1:1 ratio reflects this theoretical hierarchy. Sensitivity analysis with equal weights (33/33/33) produces Spearman ρ > 0.95 with the primary weighting.

---

## 9. PHI_INT (Rank-Based Inverse Normal Transform)

**What it is:** A statistically transformed version of PHI used only in the full-population OLS regression (Cell 12). Not used in the sick-subgroup analyses (Cells 13–16).

**Why it exists:** Raw PHI is zero-inflated — 61–82% of the general population scores exactly 0. Running OLS directly on this distribution violates normality assumptions and produces unreliable standard errors. PHI_INT maps every observation's rank to the corresponding standard normal quantile, transforming the spike at zero into a smooth left tail while perfectly preserving rank ordering (Spearman ρ = 1.0 with raw PHI by construction).

**Formula:** `PHI_INT = Φ⁻¹( (rank − 0.5) / n )`

Where Φ⁻¹ is the inverse of the standard normal CDF. Ties (all the zero-scorers) receive average rank. NaNs are preserved. The −0.5 continuity correction prevents Φ⁻¹(0) = −∞ at the lowest rank.

**When NOT to use PHI_INT:** In sick-subgroup regressions (Cell 13 onwards), PHI > 0 by definition, so the distribution is already right-skewed but continuous and tractable — PHI_INT is unnecessary and the raw PHI scale is more interpretable.

**Reference:** Beasley et al. (2009), *Genetic Epidemiology* 33(7):589-594.

---

## Summary: Index Hierarchy and Data Flow

```
IHDS-II raw CSVs
      ↓ [Cell 5: clean_dataframe()]
cleaned_*.csv
      ↓ [Cells 7-8: build_*_score(), build_physical_health_index()]
STM_score + CDI_score + ADL_score → PHI [Cells 9, 11]
      ↓ [Cell 14: add_medical_costs()]
OOP_STM, OOP_MB, OOP_SHARE, CAT_EXP, UNTREATED
      ↓ [Cell 15: build_access_index()]  — uses SM14B (fixed)
ACC_dist + ACC_gap + ACC_fin → HAS
      ↓ [Cell 16: build_social_vulnerability_index()]
SOC_gender + SOC_caste → SVI
      ↓ [Cell 17: build_multidimensional_index()]
0.50×PHI + 0.25×HAS + 0.25×SVI → MHI (final_mhi_dataset_*.csv)
```

All indices are oriented **higher = worse** throughout the pipeline.
