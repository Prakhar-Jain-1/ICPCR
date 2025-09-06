
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# ==============================
# User-specified variable groups
# (from your message)
# ==============================
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

Regression_list = [Farm, Animal_work, Non_farm, Education, Technology, Teacher_and_school, College_school, Child_and_school,
                   Short_term_Morbidity, Major_Morbidity, Activity_difficulty, Tobacco_and_other, Anthropometry, 
                   WorkSpace, WSEARN, Income_Household]

# ==============================
# Utility functions
# ==============================

def read_csv_safely(path: str) -> pd.DataFrame:
    """Read CSV filling empty strings/spaces with NA as requested."""
    df = pd.read_csv(
        path,
        na_values=['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NaN'],
        keep_default_na=True
    )
    # Replace cells that are only whitespace with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df

def safe_subset(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return only columns that exist in df, preserving order."""
    return [c for c in cols if c in df.columns]

def reduce_block(
    df: pd.DataFrame,
    cols: List[str],
    method: str = 'pca',
    var_threshold: float = 0.65,
    max_components: int = 3,
    min_components: int = 1,
    prefix: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Reduce a block of variables via PCA or FactorAnalysis.
    - Keeps only numeric columns.
    - Imputes missing via median.
    - Standardizes before decomposition.
    - Chooses the smallest number of components (1..max_components) whose cumulative
      explained variance >= var_threshold (for PCA). For FA, uses max_components.
    Returns:
    - DataFrame with new index columns appended.
    - Dictionary of loadings per component (for interpretability).
    """
    existing = safe_subset(df, cols)
    if len(existing) == 0:
        return pd.DataFrame(index=df.index), {}

    # Select numeric subset
    X = df[existing].apply(pd.to_numeric, errors='coerce')

    # Impute + scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    loadings: Dict[str, np.ndarray] = {}
    out = pd.DataFrame(index=df.index)

    if method.lower() == 'pca':
        # Fit with max_components first
        n_components_try = min(max_components, X.shape[1])
        pca_full = PCA(n_components=n_components_try, random_state=42)
        pca_full.fit(X_scaled)

        # Determine number of components by cumulative variance
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_opt = np.searchsorted(cumvar, var_threshold) + 1
        n_opt = int(np.clip(n_opt, min_components, n_components_try))

        pca = PCA(n_components=n_opt, random_state=42)
        scores = pca.fit_transform(X_scaled)

        # Add components to output
        for i in range(n_opt):
            name = f"{prefix or 'IDX'}_PC{i+1}"
            out[name] = scores[:, i]
            loadings[name] = pca.components_[i]

    elif method.lower() in ['fa', 'factor', 'factor_analysis']:
        n_components_try = min(max_components, X.shape[1])
        fa = FactorAnalysis(n_components=n_components_try, random_state=42)
        scores = fa.fit_transform(X_scaled)

        for i in range(n_components_try):
            name = f"{prefix or 'IDX'}_FA{i+1}"
            out[name] = scores[:, i]
            # FactorAnalysis components_ is not standardized like PCA; use components_ (loadings) if available
            if hasattr(fa, 'components_') and fa.components_ is not None:
                loadings[name] = fa.components_[i]
    else:
        raise ValueError("method must be 'pca' or 'fa'")

    # Attach metadata to columns for later interpretability
    out.attrs['block_columns'] = existing
    out.attrs['loadings'] = loadings
    return out, loadings

def build_indices(
    df: pd.DataFrame,
    blocks: Dict[str, List[str]],
    method: str = 'pca',
    var_threshold: float = 0.65,
    max_components: int = 3
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Run reduction per thematic block and concatenate indices."""
    all_indices = []
    all_loadings: Dict[str, Dict[str, np.ndarray]] = {}
    for name, cols in blocks.items():
        idx_df, load = reduce_block(
            df, cols, method=method, var_threshold=var_threshold,
            max_components=max_components, prefix=name.upper()
        )
        # Only append if we actually created any components
        if idx_df.shape[1] > 0:
            all_indices.append(idx_df)
            all_loadings[name] = load
    if len(all_indices) == 0:
        return pd.DataFrame(index=df.index), {}
    return pd.concat(all_indices, axis=1), all_loadings

def make_design_matrix(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    controls: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build model matrix X (indices + controls). Categorical controls are one-hot encoded."""
    # Prepare control frame
    control_cols = safe_subset(df, controls)
    C = df[control_cols].copy()

    # Identify numeric vs categorical (object) columns
    num_cols = C.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in control_cols if c not in num_cols]

    # Impute missing and encode
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    # Fit/transform controls
    C_proc = pre.fit_transform(C)
    cat_feature_names = []
    if len(cat_cols) > 0:
        cat_feature_names = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
    X_controls_cols = num_cols + cat_feature_names

    # Standardize index_df too (already standardized via PCA scores, but keep consistent)
    X_index = index_df.values
    X = np.hstack([X_index, C_proc])
    X_cols = index_df.columns.tolist() + X_controls_cols

    return X, pre, X_cols

def fit_regression(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'linear'
):
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        return model
    elif model_type == 'logistic':
        model = LogisticRegression(max_iter=200, n_jobs=None)
        model.fit(X, y)
        return model
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

# ==============================
# Example main workflow
# ==============================

def main(
    indiv_csv_path: str,
    hh_csv_path: Optional[str] = None,
    id_keys: Tuple[str, ...] = ('STATEID', 'DISTID', 'PSUID', 'HHID', 'IDPERSON'),
    target_col: Optional[str] = 'INCOMEPC',
    method: str = 'pca',
    var_threshold: float = 0.65
):
    """
    indiv_csv_path: path to individual-level CSV
    hh_csv_path: optional path to household-level CSV (will be merged if provided)
    id_keys: ID keys to merge on
    target_col: dependent variable (default: INCOMEPC if available)
    method: 'pca' or 'fa'
    var_threshold: threshold for PCA cumulative variance (ignored for FA)
    """
    # 1) Read CSVs with strict NA handling (empty strings -> NA)
    indiv = read_csv_safely(indiv_csv_path)
    if hh_csv_path:
        hh = read_csv_safely(hh_csv_path)
    else:
        hh = None

    # Ensure empty/whitespace -> NaN (redundant but safe if user provided different CSV semantics)
    indiv = indiv.replace(r'^\s*$', np.nan, regex=True)
    if hh is not None:
        hh = hh.replace(r'^\s*$', np.nan, regex=True)

    # 2) Merge (inner) if household file provided
    if hh is not None:
        # Keep only merge keys that both have
        merge_keys = [k for k in id_keys if k in indiv.columns and k in hh.columns]
        if len(merge_keys) == 0:
            raise ValueError("None of the id_keys are present in both dataframes.")
        df = indiv.merge(hh, on=merge_keys, how='inner', suffixes=('', '_HH'))
    else:
        df = indiv.copy()

    # 3) Build indices for regression blocks
    blocks = {
        'farm': Farm,
        'animal': Animal_work,
        'nonfarm': Non_farm,
        'education': Education,
        'technology': Technology,
        'teacher_school': Teacher_and_school,
        'college_school': College_school,
        'child_school': Child_and_school,
        'short_morb': Short_term_Morbidity,
        'major_morb': Major_Morbidity,
        'activity': Activity_difficulty,
        'tobacco': Tobacco_and_other,
        'anthro': Anthropometry,
        'workspace': WorkSpace,
        'wsearn': WSEARN,
        'income_hh': Income_Household
    }

    index_df, loadings = build_indices(df, blocks, method=method, var_threshold=var_threshold, max_components=3)

    # 4) Controls
    control_vars = list(dict.fromkeys(Basic_info_list + Household_details + Caste_and_Religion + Urban))
    control_vars = safe_subset(df, control_vars)

    # 5) Target
    y = None
    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors='coerce').values
    else:
        # If default not found, try INCOME or any numeric var from Household_financial
        fallback = [c for c in Household_financial if c in df.columns]
        if len(fallback) > 0:
            y = pd.to_numeric(df[fallback[0]], errors='coerce').values
            target_col = fallback[0]
        else:
            raise ValueError("No valid target variable found. Pass target_col explicitly.")

    # Drop rows with NA in y
    valid = ~pd.isna(y)
    df = df.loc[valid].reset_index(drop=True)
    index_df = index_df.loc[valid].reset_index(drop=True)
    y = y[valid]

    # 6) Build design matrix
    X, preproc, X_cols = make_design_matrix(df, index_df, controls=control_vars)

    # 7) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 8) Fit regression
    model = fit_regression(X_train, y_train, model_type='linear')

    # 9) Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 10) Cross-validated R^2 for sanity
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2')

    # 11) Export indices + target for inspection
    out = pd.DataFrame(index_df)
    out[target_col] = y
    out_path = os.path.splitext(indiv_csv_path)[0] + "_indices_and_target.csv"
    out.to_csv(out_path, index=False)

    # 12) Print summary
    print("=== Model Summary ===")
    print(f"Target: {target_col}")
    print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
    print(f"Test R^2: {r2:.3f} | RMSE: {rmse:.3f}")
    print(f"CV R^2 (mean±sd): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("\nTop 10 coefficients by absolute magnitude:")
    coefs = model.coef_
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    for i in top_idx:
        print(f"{X_cols[i]}: {coefs[i]:.4f}")

    # 13) Save a human-readable loadings report
    loadings_report = []
    for block_name, comp_dict in loadings.items():
        if len(comp_dict) == 0:
            continue
        # Retrieve the original columns used in the block
        block_cols = blocks[block_name]
        existing = [c for c in block_cols if c in df.columns]
        for comp_name, weights in comp_dict.items():
            # Ensure lengths match (numeric-only subset may have reduced columns)
            # Recompute numeric-only existing used in reduction:
            existing_numeric = pd.to_numeric(df[existing], errors='coerce')
            actual_cols = existing_numeric.columns.tolist()
            k = min(len(actual_cols), len(weights))
            pairs = list(zip(actual_cols[:k], weights[:k]))
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
            for var, w in pairs_sorted:
                loadings_report.append({
                    'block': block_name,
                    'component': comp_name,
                    'variable': var,
                    'weight': float(w)
                })
    if len(loadings_report) > 0:
        lr_df = pd.DataFrame(loadings_report)
        load_path = os.path.splitext(indiv_csv_path)[0] + "_index_loadings.csv"
        lr_df.to_csv(load_path, index=False)
        print(f"\nSaved: {load_path}")

    print(f"Saved: {out_path}")
    return {
        'model': model,
        'preprocessor': preproc,
        'X_cols': X_cols,
        'index_df': index_df,
        'target_name': target_col
    }

if __name__ == "__main__":
    # Example usage:
    # Update these paths to your local CSVs before running.
    INDIV_PATH = "./IHDS_Indiv.csv"
    HH_PATH = "./IHDS_HH.csv"  # or None if you only have individual-level file

    if os.path.exists(INDIV_PATH):
        main(INDIV_PATH, HH_PATH if os.path.exists(HH_PATH) else None)
    else:
        print("Please edit the script to point to your CSV file paths (INDIV_PATH / HH_PATH)."
              " The workflow will: read CSVs with empty strings as NA, build indices per block,"
              " and fit a linear model on INCOMEPC (or fallback)."
        )
