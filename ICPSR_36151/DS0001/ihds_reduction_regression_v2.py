#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IHDS reduction + regression pipeline (compat version)
- Avoids sklearn.mean_squared_error(squared=...) for older sklearn
- Handles OneHotEncoder 'sparse_output' vs 'sparse'
- Falls back from get_feature_names_out to get_feature_names
"""
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

# --------------------- User knobs ---------------------
CSV_PATHS = [
    # e.g. "/path/to/IHDS_II_individual.csv",
    "states_datasets/data_subset_Uttar Pradesh 09.csv",
    "states_datasets/data_subset_Gujarat 24.csv",
    "states_datasets/data_subset_Maharashtra 27.csv",
    # e.g. "/path/to/IHDS_II_household.csv",
]

MERGE_KEYS = ["STATEID", "DISTID", "PSUID", "HHID", "IDPERSON"]

TARGET_CANDIDATES = ["INCOMEPC", "INCOME", "INCEARN"]

PC1_KEEP_THRESHOLD = 0.60
CUMVAR_TARGET = 0.70
MAX_COMPONENTS = 2
INTRA_BLOCK_CORR_DROP = 0.90
MAX_CATS = 20

# ----- Thematic lists -----
Basic_info_list = ["RO3", "RO4", "RO5", "RO6", "RO7", "RO8", "RO9", "RO10"]

Farm = ["FM1", "FM36Y", "FM37", "FM38", "FMHOURS", "WKFARM", "FM39AY"]

Animal_work = ["AN1", "AN5Y", "AN6", "AN7Y"]

Non_farm = ["NF1", "NFBN1", "NF9", "NF12", "NF13", "NF15Y", "NFBN21", "NF29", "NF32", "NF33", "NF35Y", "NFBN41", "NF49", "NF52", "NF53", "NFDAYS", "NFHOURS", "NF55Y"]

Income = ["IN11S1", "IN13S1", "IN11S2", "IN13S2", "IN11S3", "IN13S3", "IN11S4", "IN13S4", "IN11S5", "IN13S5", "IN11S6", "IN13S6", "IN11S7",
           "IN13S7", "IN11S8", "IN13S8", "IN18", "IN19", "IN20", "IN21", "IN22", "IN23", "IN24"]

Education = ["ED2", "ED3", "ED4", "ED5", "ED6", "EDUC7", "EDUNDER1", "ED7", "ED8", "ED9", "ED10", "ED11", "ED12", "ED13"]

Technology = ["MM7Y", "MM8", "MM9", "MM12Y", "MM13", "MM14"]

Teacher_and_school = ["TA3", "TA4", "TA5", "TA6", "TA8A", "TA8B", "TA9A", "TA9B", "TA10A", "TA10B"]

College_school = ["CS3", "CS3Y", "CS4", "CS5", "CS6", "CS7", "CS8", "CS9", "CS10", "CS11", "CS12", "CS13", "CS16", "CS17",
                   "CS18", "CS19", "CS20", "CS21", "CS22", "CS23", "CS24", "CS25", "CS26", "CS27", "CS28"]

Child_and_school = ["CH2", "CH3", "CH4A", "CH4B", "CH5", "CH6", "CH7", "CH8", "CH9", "CH10", "CH11", "CH12", "CH13", "CH14", "CH15", "CH16", "CH17", "CH18", "CH19", "CH20", "CH22"]

Short_term_Morbidity = ["SM3", "SM4", "SM5", "SM6", "SM7", "SM8", "SM9", "SM10", "SM11", "SM12", "SM14A", "SM14B", "SM15A", "SM15B", "SM16", "SM17", "SM18", "SM19", "SM20", "SM21", "SM22"]

Major_Morbidity =["MB3", "MB4", "MB5", "MB6", "MB7", "MB8", "MB9", "MB10", "MB11", "MB12", "MB13", "MB14", "MB15", "MB16", "MB17", 
                  "MB18", "MB19", "MB21A", "MB21B", "MB22A", "MB22B", "MB23", "MB24", "MB25", "MB26", "MB27", "MB28", "MB29"]

Activity_difficulty = ["AD3", "AD4", "AD5", "AD6", "AD7", "AD8", "AD9"]

Tobacco_and_other = ["TO3", "TO4", "TO5", "TO6"]

Anthropometry = ["AP2", "AP3", "AP5", "AP6", "AP7", "AP8", "AP9"]

Eligible_women = ["EW3Y"]

Urban = ["URBAN2011", "URBAN4_2011", "METRO", "METRO6", "POVLINE2005", "POVLINE2012","DEFLATOR"]

Household_details = ["NPERSONS", "EWELIGIBLE", "EWQELIGIBLE","MHEADAGE", "FHEADAGE", "NADULTM", "NADULTF", "NCHILDM", "NCHILDF", "NTEENM", "NTEENF", "NELDERM", "NELDERF", "NMARRIEDM", 
                     "NMARRIEDF", "NWKNONAG", "NWKAGLAB", "NWKSALARY", "NWKBUSINESS", "NWKFARM", "NWKANIMAL", "NWKNREGA", "NWKNREGA4", "NWKNONNREGA", "NWKANY5",
                     "NNR", "HHEDUC", "HHEDUCM", "HHEDUCF"]

Caste_and_Religion = ["ID11", "ID13", "GROUPS"]

Buiseness = ["NF5", "NF25", "NF45"]

Household_financial = ["COTOTAL", "COPC", "ASSETS", "ASSETS2005", "INCCROP", "INCAGPROP", "INCANIMAL", "INCAG", "INCBUS", "INCOTHER", "INCEARN", "INCBENEFITS", "INCREMIT", "INCOME", "INCOMEPC","RSUNEARN"]

Work_participation = ["WKANIMAL", "WKBUSINESS","WKNREGA", "WKDAYS", "WKHOURS", "WKANY5"]

WorkSpace = ["WS3NM", "WS4", "WS5", "WS7", "WS7MONTHS", "WS8", "WS8YEAR", "WS9", "WS10", "WS10ANNUAL", "WSEARN", "WSEARNHOURLY", "WS11", "WS11MEALS", "WS11HOUSE",
              "WS11MEALSRS", "WS11HOUSERS", "WS12", "WS13", "WS14", "WS15", "WS7AGLAB", "WS8AGLAB", "WSEARNAGLAB", "WKAGLAB", "WS7NONAG", "WS8NONAG", "WSEARNNONAG",
                "WKNONAG", "WS7SALARY", "WS8SALARY", "WSEARNSALARY", "WKSALARY", "WS7NREGA", "WS8NREGA", "WSEARNNREGA"]

WSEARN = ["WSEARNAGLAB", "WSEARNNONAG", "WSEARNSALARY", "WSEARNNREGA", "WSEARNANNUAL", "WSEARN"]

Income_Household = ["INCNONAG", "INCAGLAB", "INCSALARY", "INCNREGA", "INCNONNREGA"]

Migrants_data = ["MG4", "MG5", "MG6", "MG7", "MG8", "MG9NM", "MG10", "MG11","MGYEAR5", "NMIG5", "MGMONTHS5", "MGYEAR1", "NMIG1", "MGMONTHS1"]

Regression_list = [("Farm", Farm), ("Animal_work", Animal_work), ("Non_farm", Non_farm), ("Education", Education), ("Technology", Technology), 
                   ("Teacher_and_school", Teacher_and_school), ("College_school", College_school), ("Child_and_school", Child_and_school),
                   ("Short_term_Morbidity", Short_term_Morbidity), ("Major_Morbidity", Major_Morbidity), ("Activity_difficulty", Activity_difficulty), 
                   ("Tobacco_and_other", Tobacco_and_other), ("Anthropometry", Anthropometry), ("WorkSpace", WorkSpace), ("WSEARN", WSEARN), 
                   ("Income_Household", Income_Household)]

Control_blocks = [("Basic_info", Basic_info_list), ("Household_details", Household_details), ("Caste_and_Religion", Caste_and_Religion), ("Urban", Urban)]

# --------------------- Helpers ---------------------

def load_and_merge(csv_paths: List[str], merge_keys: List[str]) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError("Please set CSV_PATHS to point at your IHDS CSV files.")
    dfs = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    dfs = [df.rename(columns={c: c.strip() for c in df.columns}) for df in dfs]
    dfs = [df.replace(r"^\s*$", np.nan, regex=True) for df in dfs]
    base = dfs[0]
    for other in dfs[1:]:
        common = [k for k in merge_keys if k in base.columns and k in other.columns]
        if not common:
            base = pd.concat([base, other], axis=0, ignore_index=True, sort=False)
        else:
            base = base.merge(other, on=common, how="outer", suffixes=("", "_y"))
            dupes = [c for c in base.columns if c.endswith("_y")]
            for c in dupes:
                base[c.replace("_y", "")] = base[c.replace("_y", "")].combine_first(base[c])
            base.drop(columns=dupes, inplace=True)
    return base

def coerce_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": 1, "y": 1, "1": 1, 1: 1, "true": 1, True: 1, "male": 1,
        "no": 0, "n": 0, "0": 0, 0: 0, "false": 0, False: 0, "female": 0
    }
    if series.dtype == "O":
        s = series.astype(str).str.strip().str.lower()
        unique = set(s.dropna().unique())
        if unique.issubset(set(mapping.keys())):
            return s.map(mapping).astype("float64")
    return pd.to_numeric(series, errors="ignore")

def drop_high_corr(df_num: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df_num.shape[1] <= 1:
        return df_num
    corr = df_num.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df_num.drop(columns=to_drop, errors="ignore")

def fit_block_pca(df: pd.DataFrame, cols: List[str], block_name: str) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame(index=df.index), {}, pd.DataFrame()
    X = df[available].copy()
    X = X.replace(r"^\s*$", np.nan, regex=True)
    for c in X.columns:
        X[c] = coerce_binary(X[c])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]
    X = X.dropna(axis=1, how="all")
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique[nunique > 1].index]
    if X.shape[1] == 0:
        return pd.DataFrame(index=df.index), {}, pd.DataFrame()
    X_imp = X.copy()
    for c in X_imp.columns:
        med = X_imp[c].median()
        X_imp[c] = X_imp[c].fillna(med)
    X_imp = drop_high_corr(X_imp, INTRA_BLOCK_CORR_DROP)
    if X_imp.shape[1] == 0:
        return pd.DataFrame(index=df.index), {}, pd.DataFrame()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp.values)
    pca = PCA(n_components=min(MAX_COMPONENTS, X_imp.shape[1]))
    pcs = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=X_imp.columns, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    if evr[0] >= PC1_KEEP_THRESHOLD:
        kept = 1
    else:
        cum = np.cumsum(evr)
        kept = int(np.argmax(cum >= CUMVAR_TARGET) + 1) if any(cum >= CUMVAR_TARGET) else min(MAX_COMPONENTS, pca.n_components_)
    kept = max(1, min(kept, pca.n_components_))
    block_feats = {}
    for k in range(kept):
        block_feats[f"{block_name}_PC{k+1}"] = pcs[:, k]
    block_df = pd.DataFrame(block_feats, index=df.index)
    return block_df, block_feats, loadings

def choose_target(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    fallback = [c for c in df.columns if re.match(r"IN(C|COME)", c)]
    if fallback:
        return fallback[0]
    raise ValueError("No target variable found. Please set TARGET_CANDIDATES or edit code.")

def limit_categories(s: pd.Series, max_cats: int = MAX_CATS) -> pd.Series:
    vc = s.value_counts(dropna=True)
    if len(vc) <= max_cats:
        return s
    keep = set(vc.index[:max_cats])
    return s.where(s.isin(keep), other="Other")

def build_design_matrix(df: pd.DataFrame,
                        indices_by_block: Dict[str, List[str]],
                        fallback_vars: List[str],
                        controls: List[Tuple[str, List[str]]]
                        ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    parts = []
    feature_names = []
    for block, idx_cols in indices_by_block.items():
        parts.append(df[idx_cols])
        feature_names.extend(idx_cols)
    if fallback_vars:
        parts.append(df[fallback_vars])
        feature_names.extend(fallback_vars)
    control_cols = []
    for name, cols in controls:
        control_cols.extend([c for c in cols if c in df.columns])
    if control_cols:
        parts.append(df[control_cols])
        feature_names.extend(control_cols)
    if not parts:
        raise ValueError("No features available to build the design matrix.")
    X = pd.concat(parts, axis=1)
    return X, feature_names, control_cols

def separate_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in X.columns:
        # First attempt to coerce column into numeric
        coerced = pd.to_numeric(X[c], errors="coerce")
        # If at least half the non-NA values are valid numbers, treat as numeric
        if coerced.notna().sum() >= 0.5 * len(coerced.dropna()):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def main():
    outdir = Path("ihds_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(CSV_PATHS, MERGE_KEYS)
    df.to_csv(outdir / "clean_merged.csv", index=False)

    indices_by_block = {}
    fallback_vars_all = []
    for block_name, cols in Regression_list:
        block_df, block_feats, loadings = fit_block_pca(df, cols, block_name)
        if not block_df.empty:
            df = pd.concat([df, block_df], axis=1)
            indices_by_block[block_name] = list(block_df.columns)
            if not loadings.empty:
                loadings.to_csv(outdir / f"block_component_loadings_{block_name}.csv")
        else:
            avail = [c for c in cols if c in df.columns]
            if avail:
                numeric_avail = []
                for c in avail:
                    coerced = pd.to_numeric(df[c], errors="coerce")
                    if coerced.notna().sum() > 0 and coerced.nunique(dropna=True) > 1:
                        df[c + "_num"] = coerced
                        numeric_avail.append(c + "_num")
                if numeric_avail:
                    kept = drop_high_corr(df[numeric_avail], INTRA_BLOCK_CORR_DROP).columns.tolist()
                    fallback_vars_all.extend(kept)

    X, feature_names, control_cols = build_design_matrix(df, indices_by_block, fallback_vars_all, Control_blocks)

    target = choose_target(df, TARGET_CANDIDATES)
    y = pd.to_numeric(df[target], errors="coerce")
    keep_rows = y.notna()
    X = X.loc[keep_rows].copy()
    y = y.loc[keep_rows].astype(float)

    X = X.replace(r"^\s*$", np.nan, regex=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = coerce_binary(X[c])
    num_cols, cat_cols = separate_types(X)
    for c in num_cols:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
    for c in cat_cols:
        X[c] = limit_categories(X[c])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("reg", LinearRegression())
    ])

    model.fit(X, y)

    # --- Start: Corrected and Streamlined Post-processing ---

    # Get coefficients and correct feature names
    reg = model.named_steps["reg"]
    coef = reg.coef_
    
    try:
        final_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    except AttributeError:
        final_feature_names = model.named_steps["preprocessor"].get_feature_names()

    # Create and save the coefficients DataFrame
    coef_df = pd.DataFrame({
        "feature": final_feature_names,
        "coef": coef
    }).sort_values("coef", ascending=False)
    coef_df.to_csv(outdir / "regression_coefficients.csv", index=False)

    # Calculate prediction metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    try:
        rmse = mean_squared_error(y, y_pred, squared=False)
    except TypeError: # For older sklearn versions
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # --- End: Corrected Block ---
    
    # Save supplementary output files
    pd.DataFrame({
        "block": list(indices_by_block.keys()), 
        "indices": [",".join(v) for v in indices_by_block.values()]
    }).to_csv(outdir / "block_indexes.csv", index=False)

    with open(outdir / "X_feature_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_feature_names))

    with open(outdir / "regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Rows used: {len(y)}\n")
        f.write(f"Blocks with indices: {list(indices_by_block.keys())}\n")
        f.write(f"Fallback variables kept: {fallback_vars_all}\n")
        f.write(f"R^2: {r2:.4f}\nRMSE: {rmse:.4f}\n")

    print("Done. Outputs written to:", outdir)
    print("Remember to set CSV_PATHS at the top of this script.")

if __name__ == "__main__":
    main()