# -*- coding: utf-8 -*-
"""
EPIAS Project — Streamlit (GitHub-native)
- Tüm veri/modül dosyaları GitHub repo üzerinden çekilir (raw.githubusercontent).
- Lokal dosya yolu yok; sol panelde repo içeriğinden seçim yapılır.
- Mevcut model akışları korunur (ridge, rf, lgbm, xgb, voting).
"""

from __future__ import annotations
import os, sys, types, warnings, ast, io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import requests  # GitHub'dan dosya çekmek için

# ================== REPO AYARLARI (gerekirse değiştir) ==================
GH_OWNER  = "Boozkann"
GH_REPO   = "EPIAS-Project"
GH_BRANCH = "main"

# Varsayılan seçimler (repo içi göreli yollar)
DEFAULT_DATA_DIR  = "data/processed"
DEFAULT_DATA_FILE = "fe_full_plus2_causal.parquet"  # varsa bu seçilir
DEFAULT_MODULE_CANDIDATES = [
    "data/processed/EDA_to_Model_EPIAS_Final_converted.py",
    "EDA_to_Model_EPIAS_Final_converted.py",
]
# ========================================================================

# --------------------------- Soft dependency shims -----------------------
def _inject_fake(name: str, attr_map: dict | None = None):
    m = types.ModuleType(name)
    for k, v in (attr_map or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m

for pkg, attrs in [
    ("tabulate", {"tabulate": lambda x, **kw: str(x)}),
    ("IPython", {"display": lambda *a, **k: None}),
]:
    try:
        __import__(pkg)
    except Exception:
        _inject_fake(pkg, attrs)

try:
    import matplotlib  # noqa
    import matplotlib.pyplot as plt  # noqa
except Exception:
    _inject_fake("matplotlib")
    _inject_fake("matplotlib.pyplot", {
        "figure": lambda *a, **k: None,
        "plot":   lambda *a, **k: None,
        "show":   lambda: None,
        "style":  types.SimpleNamespace(use=lambda *a, **k: None),
    })

try:
    import seaborn as sns  # noqa
except Exception:
    _inject_fake("seaborn", {
        "set":      lambda *a, **k: None,
        "lineplot": lambda *a, **k: None,
        "barplot":  lambda *a, **k: None,
        "heatmap":  lambda *a, **k: None,
    })

try:
    from statsmodels.tsa.seasonal import seasonal_decompose  # noqa
except Exception:
    import pandas as _pd
    from collections import namedtuple
    Decomp = namedtuple("DecomposeResult", ["observed","trend","seasonal","resid"])
    def _fake_decompose(series, model="additive", period=None):
        s = _pd.Series(series).astype(float)
        nan = _pd.Series([float("nan")]*len(s), index=s.index)
        return Decomp(observed=s, trend=nan, seasonal=nan, resid=nan)
    _inject_fake("statsmodels")
    _inject_fake("statsmodels.tsa")
    _inject_fake("statsmodels.tsa.seasonal", {"seasonal_decompose": _fake_decompose})

try:
    import optuna  # noqa
except Exception:
    _inject_fake("optuna", {"create_study": lambda *a, **k: None})

# --------------------------- ML lib imports ------------------------------
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor as CART
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# ------------------------------------------------------------------------

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EPİAŞ — GitHub-native ML Runner", layout="wide")
st.title("⚡ EPİAŞ — ML Model Runner (GitHub-native)")

# ========================== GitHub yardımcıları ==========================
def gh_api_url(path: str) -> str:
    path = path.strip("/ ")
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"

def gh_raw_url(path: str) -> str:
    path = path.strip("/ ")
    return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path}"

@st.cache_data(show_spinner=False)
def gh_list_dir(path: str) -> List[dict]:
    """GitHub contents API ile klasör listesi (dosya/klasör)."""
    url = gh_api_url(path)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # API klasörse liste döner; dosya ise dict döner -> normalize
    if isinstance(data, dict):
        return [data]
    return data

@st.cache_data(show_spinner=False)
def gh_get_bytes(path: str) -> bytes:
    """Raw içerik (parquet gibi ikili) indirir."""
    url = gh_raw_url(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def gh_get_text(path: str) -> str:
    """Raw metin içerik (py/csv) indirir."""
    url = gh_raw_url(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text
# ========================================================================

# ========================= Veri/Modül yükleyiciler =======================
@st.cache_data(show_spinner=True)
def read_data_from_github(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".parquet"):
        content = gh_get_bytes(path)
        df = pd.read_parquet(io.BytesIO(content))
    elif low.endswith(".csv"):
        text = gh_get_text(path)
        df = pd.read_csv(io.StringIO(text))
    else:
        raise ValueError("Desteklenmeyen format. Parquet veya CSV verin.")
    # timestamp normalize
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

# Güvenli import: kaynak koddan (dosyaya yazmadan) yükleme
ALLOWED_GLOBAL_NAMES = {
    "CUT_TS",
    "best_lgbm_params", "best_xgb_params", "best_rf_params", "best_cart_params", "best_ridge_params",
    "lgbm_best_params", "xgb_best_params", "rf_best_params", "ridge_best_params", "cart_best_params",
    "best_lgbm_params_ptf", "best_lgbm_params_smf",
    "best_xgb_params_ptf",  "best_xgb_params_smf",
    "best_rf_params_ptf",   "best_rf_params_smf",
    "best_ridge_params_ptf","best_ridge_params_smf",
    "best_cart_params_ptf", "best_cart_params_smf",
}

def safe_import_module_from_source(src: str, virtual_filename: str = "user_module.py"):
    """
    Kaynaktan güvenli import: sadece fonksiyon/sınıf/import ve ALLOWED_GLOBAL_NAMES'e
    yapılan atamaları bırakır; top-level yürütmeyi eler.
    """
    tree = ast.parse(src, filename=virtual_filename)
    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            new_body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                targets = [node.target]
            keep = False
            for t in targets:
                if isinstance(t, ast.Name) and t.id in ALLOWED_GLOBAL_NAMES:
                    keep = True
            if keep:
                new_body.append(node)
        else:
            continue  # yürütücü bloklar atılır

    new_tree = ast.Module(body=new_body, type_ignores=[])
    code = compile(new_tree, filename=virtual_filename, mode="exec")

    mod = types.ModuleType("user_pipeline_sandbox")
    mod.__file__ = f"gh://{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{virtual_filename}"

    preset_missing = {"CUT", "CUT_TRAIN", "CUT_VALID", "CUTOFF", "SPLIT_TS"}
    for nm in preset_missing:
        mod.__dict__.setdefault(nm, None)

    for _ in range(5):
        try:
            exec(code, mod.__dict__)
            break
        except NameError as e:
            msg = str(e)
            if "name '" in msg and "' is not defined" in msg:
                missing = msg.split("name '", 1)[1].split("'", 1)[0]
                mod.__dict__.setdefault(missing, None)
                continue
            raise
    else:
        raise
    return mod
# ========================================================================

# ============================ Model yardımcıları =========================
def _metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae  = float(np.mean(np.abs(yhat - y)))
    denom = np.maximum(np.abs(y), np.percentile(np.abs(y), 10) if y.size else 10.0) + 1e-6
    mape = float(np.mean(np.abs(yhat - y) / denom) * 100)
    ss_tot = np.sum((y - np.mean(y))**2); ss_res = np.sum((yhat - y)**2)
    r2 = float(1.0 - (ss_res / (ss_tot if ss_tot > 1e-12 else 1e-12)))
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "R2": r2}

def build_features_via_module(df: pd.DataFrame, module, target: str) -> pd.DataFrame:
    """Modülde build_feature_frame varsa çağırır; yoksa df'i döner."""
    if module is None:
        return df
    fn = getattr(module, "build_feature_frame", None)
    if callable(fn):
        try:
            try:
                out = fn(df, target=target)
            except TypeError:
                out = fn(df)
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.tz_localize(None)
            out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return out
        except Exception as e:
            st.warning(f"build_feature_frame çağrısı başarısız: {e}")
    return df

def get_optimized_params(module, model_key: str, target: str):
    """Modülde varsa best param sözlüklerini alır."""
    if module is None:
        return None
    names = [
        f"best_{model_key}_params_{target}",
        f"best_{model_key}_params",
        f"{model_key}_best_params_{target}",
        f"{model_key}_best_params",
    ]
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None

def make_model_factories(module, target: str, quick_mode: bool):
    factories = {}

    # Ridge
    ridge_params = get_optimized_params(module, "ridge", target) or {}
    factories["ridge"] = lambda: Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(**{k: v for k, v in ridge_params.items() if k in Ridge().get_params()})),
    ])

    # LightGBM
    if LGBMRegressor is not None:
        lgbm_params = get_optimized_params(module, "lgbm", target) or {}
        base = dict(random_state=42, n_jobs=-1)
        if quick_mode:
            base.update(dict(n_estimators=300, learning_rate=0.08, num_leaves=31, max_depth=-1))
        else:
            base.update(dict(n_estimators=800))
        base.update(lgbm_params)
        factories["lgbm"] = lambda: LGBMRegressor(**base)
    else:
        factories["lgbm"] = None

    # XGBoost
    if XGBRegressor is not None:
        xgb_params = get_optimized_params(module, "xgb", target) or {}
        base = dict(random_state=42, n_estimators=300 if quick_mode else 800,
                    learning_rate=0.07 if quick_mode else 0.05,
                    subsample=0.9, colsample_bytree=0.9, max_depth=6,
                    n_jobs=-1, tree_method="hist")
        base.update(xgb_params)
        factories["xgb"] = lambda: XGBRegressor(**base)
    else:
        factories["xgb"] = None

    # RandomForest
    rf_params = get_optimized_params(module, "rf", target) or {}
    base = dict(random_state=42, n_estimators=200 if quick_mode else 500, n_jobs=-1,
                max_depth=16 if quick_mode else None)
    base.update(rf_params)
    factories["randomforest"] = lambda: RandomForestRegressor(**base)

    # CART
    cart_params = get_optimized_params(module, "cart", target) or {}
    if quick_mode and "max_depth" not in cart_params:
        cart_params["max_depth"] = 10
    factories["cart"] = lambda: CART(random_state=42, **cart_params)

    # Voting
    def _voting_factory():
        members: List[Tuple[str, object]] = [("ridge", factories["ridge"]())]
        if factories["randomforest"] is not None:
            members.append(("rf", factories["randomforest"]()))
        if quick_mode:
            if factories["lgbm"] is not None:
                members.append(("lgbm", factories["lgbm"]()))
            elif factories["xgb"] is not None:
                members.append(("xgb", factories["xgb"]()))
        else:
            if factories["lgbm"] is not None:
                members.append(("lgbm", factories["lgbm"]()))
            if factories["xgb"] is not None:
                members.append(("xgb", factories["xgb"]()))
        return VotingRegressor(estimators=members)
    factories["voting"] = _voting_factory

    return factories
# ========================================================================

# ================================ UI ====================================
with st.sidebar:
    st.header("Ayarlar")

    # Repo içi data klasörü dosyalarını listele
    data_files = []
    try:
        entries = gh_list_dir(DEFAULT_DATA_DIR)
        for it in entries:
            if it.get("type") == "file" and it.get("name", "").lower().endswith((".parquet", ".csv")):
                data_files.append(f"{DEFAULT_DATA_DIR}/{it['name']}")
    except Exception as e:
        st.warning(f"GitHub listesi alınamadı: {e}")
        data_files = []

    # Varsayılan dosya seçimi
    default_data_rel = f"{DEFAULT_DATA_DIR}/{DEFAULT_DATA_FILE}" if DEFAULT_DATA_FILE else (data_files[0] if data_files else "")

    data_choice = st.selectbox(
        "Veri dosyası (GitHub)",
        options=data_files if data_files else [default_data_rel] if default_data_rel else [],
        index=(data_files.index(default_data_rel) if default_data_rel in data_files else 0) if (data_files or default_data_rel) else 0,
        help="Repo içindeki Parquet/CSV dosyaları"
    )

    # Modül .py seçimleri: default adayları + data klasöründeki .py'ler
    module_files = []
    try:
        # default klasörde .py tara
        for it in entries:
            if it.get("type") == "file" and it.get("name", "").lower().endswith(".py"):
                module_files.append(f"{DEFAULT_DATA_DIR}/{it['name']}")
    except Exception:
        pass

    # varsayılan adayları öne ekle (varsa)
    for cand in DEFAULT_MODULE_CANDIDATES:
        try:
            # existence check (HEAD yok, raw GET ile hızlı test)
            _ = gh_get_bytes(cand) if cand.lower().endswith(".parquet") else gh_get_text(cand)
            if cand not in module_files:
                module_files.insert(0, cand)
        except Exception:
            continue

    module_choice = st.selectbox(
        "Özellik/param modülü (GitHub .py) — opsiyonel",
        options=["(kullanma)"] + module_files,
        index=0,
        help="Notebook'tan dönüştürdüğün .py; varsa özellik çıkarımı ve 'best_*_params' sözlüklerini okur."
    )

    target_mode = st.selectbox("Hedef", ["ptf", "smf", "both"], index=0)
    last_days = st.slider("Grafikte son kaç gün?", 3, 90, 14, step=1)
    max_train_days = st.slider("Eğitim penceresi (son N gün)", 14, 365, 90, step=7)
    quick_mode = st.checkbox("Hızlı mod", value=True)

    if "chosen_models" not in st.session_state:
        st.session_state.chosen_models = []
    all_models = ["ridge", "lgbm", "xgb", "cart", "randomforest", "voting"]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hiçbirini seçme"):
            st.session_state.chosen_models = []
    with col2:
        if st.button("Hepsini seç"):
            st.session_state.chosen_models = all_models.copy()

    chosen_models = st.multiselect("Koşturulacak modeller", all_models, default=st.session_state.chosen_models)

    run_btn = st.button("▶ GitHub'dan Yükle • Eğit • Çiz", type="primary")
    if run_btn and not chosen_models:
        st.warning("Lütfen en az bir model seçin.")
# ========================================================================

# =============================== Çalıştırma ==============================
# 1) Veri oku (GitHub)
try:
    if not data_choice:
        raise FileNotFoundError("Seçilecek veri bulunamadı (repo'da .parquet/.csv yok mu?)")
    st.caption(f"🔎 Veri (GitHub): `{GH_OWNER}/{GH_REPO}@{GH_BRANCH}/{data_choice}`")
    raw = read_data_from_github(data_choice)
except Exception as e:
    st.error(f"Veri okunamadı: {e}")
    st.stop()

if "ptf" not in raw.columns and "smf" not in raw.columns:
    st.error("Veride 'ptf' veya 'smf' kolonu bulunmuyor.")
    st.stop()

# 2) Çalıştır
if run_btn and chosen_models:
    # Modül (opsiyonel)
    module = None
    if module_choice and module_choice != "(kullanma)":
        try:
            st.caption(f"🔎 Modül (GitHub): `{GH_OWNER}/{GH_REPO}@{GH_BRANCH}/{module_choice}`")
            module_src = gh_get_text(module_choice)
            module = safe_import_module_from_source(module_src, virtual_filename=module_choice.split("/")[-1])
        except Exception as e:
            st.warning(f"Modül import edilemedi: {e}")
            module = None

    tabs = ("ptf", "smf") if target_mode == "both" else (target_mode,)
    results: Dict[str, pd.DataFrame] = {}

    def pick_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
        drop_like = {target, "timestamp", f"{target}_pred"}
        cols = [c for c in df.columns if c not in drop_like and pd.api.types.is_numeric_dtype(df[c])]
        return cols

    def prepare_dataset(df: pd.DataFrame, module, target: str) -> Tuple[pd.DataFrame, List[str]]:
        base = df.copy()
        feat = build_features_via_module(base, module, target)
        feat = feat.dropna(subset=[target])
        features = pick_feature_columns(feat, target)
        return feat, features

    def make_model_factories(module, target: str, quick_mode: bool):
        factories = {}

        # Ridge
        ridge_params = (get_optimized_params(module, "ridge", target) or {})
        factories["ridge"] = lambda: Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(**{k: v for k, v in ridge_params.items() if k in Ridge().get_params()})),
        ])

        # LightGBM
        if LGBMRegressor is not None:
            lgbm_params = (get_optimized_params(module, "lgbm", target) or {})
            base = dict(random_state=42, n_jobs=-1)
            if quick_mode:
                base.update(dict(n_estimators=300, learning_rate=0.08, num_leaves=31, max_depth=-1))
            else:
                base.update(dict(n_estimators=800))
            base.update(lgbm_params)
            factories["lgbm"] = lambda: LGBMRegressor(**base)
        else:
            factories["lgbm"] = None

        # XGBoost
        if XGBRegressor is not None:
            xgb_params = (get_optimized_params(module, "xgb", target) or {})
            base = dict(random_state=42, n_estimators=300 if quick_mode else 800,
                        learning_rate=0.07 if quick_mode else 0.05,
                        subsample=0.9, colsample_bytree=0.9, max_depth=6,
                        n_jobs=-1, tree_method="hist")
            base.update(xgb_params)
            factories["xgb"] = lambda: XGBRegressor(**base)
        else:
            factories["xgb"] = None

        # RandomForest
        rf_params = (get_optimized_params(module, "rf", target) or {})
        base = dict(random_state=42, n_estimators=200 if quick_mode else 500, n_jobs=-1,
                    max_depth=16 if quick_mode else None)
        base.update(rf_params)
        factories["randomforest"] = lambda: RandomForestRegressor(**base)

        # CART
        cart_params = (get_optimized_params(module, "cart", target) or {})
        if quick_mode and "max_depth" not in cart_params:
            cart_params["max_depth"] = 10
        factories["cart"] = lambda: CART(random_state=42, **cart_params)

        # Voting
        def _voting_factory():
            members: List[Tuple[str, object]] = [("ridge", factories["ridge"]())]
            if factories["randomforest"] is not None:
                members.append(("rf", factories["randomforest"]()))
            if quick_mode:
                if factories["lgbm"] is not None:
                    members.append(("lgbm", factories["lgbm"]()))
                elif factories["xgb"] is not None:
                    members.append(("xgb", factories["xgb"]()))
            else:
                if factories["lgbm"] is not None:
                    members.append(("lgbm", factories["lgbm"]()))
                if factories["xgb"] is not None:
                    members.append(("xgb", factories["xgb"]()))
            return VotingRegressor(estimators=members)
        factories["voting"] = _voting_factory
        return factories

    def train_and_predict(df: pd.DataFrame, features: List[str], target: str,
                          models: List[str], factories, max_train_days: int) -> pd.DataFrame:
        data = df.sort_values("timestamp").reset_index(drop=True)
        if max_train_days and max_train_days > 0:
            tmax = data["timestamp"].max()
            tmin = tmax - pd.Timedelta(days=int(max_train_days))
            data = data[data["timestamp"].between(tmin, tmax)].copy()

        if data[target].isna().all():
            st.error(f"{target} için veri yok.")
            return pd.DataFrame()

        split = int(len(data) * 0.85)
        train = data.iloc[:split].copy()
        test  = data.iloc[split:].copy()

        Xtr, ytr = train[features].fillna(0.0).to_numpy(), train[target].astype(float).to_numpy()
        Xte, yte = test[features].fillna(0.0).to_numpy(),  test[target].astype(float).to_numpy()
        out = test[["timestamp", target]].copy()

        for key in models:
            fac = factories.get(key)
            if fac is None:
                st.warning(f"{key} modeli atlandı (gerekli paket kurulu değil).")
                continue
            try:
                mdl = fac()
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                out[f"{target}_pred_{key}"] = pred
            except Exception as e:
                st.warning(f"{key} eğitimi başarısız: {e}")
        return out

    # ==== pipeline ====
    for tgt in tabs:
        feat_df, feature_cols = prepare_dataset(raw, module, tgt)
        if not feature_cols:
            st.error(f"{tgt} için özellik kolonu bulunamadı. (build_feature_frame çıktısını ve veriyi kontrol et)")
            st.stop()
        factories = make_model_factories(module, tgt, quick_mode=quick_mode)
        preds_df  = train_and_predict(feat_df, feature_cols, tgt, chosen_models, factories, max_train_days=max_train_days)
        if preds_df.empty:
            st.error(f"{tgt} için tahmin üretilemedi.")
            st.stop()
        # Görseller
        st.subheader(f"{tgt.upper()} — Gerçek vs Tahmin")
        df = preds_df.sort_values("timestamp")
        tmax = df["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        df = df[df["timestamp"].between(tmin, tmax)].copy()
        model_cols = [c for c in df.columns if c.startswith(f"{tgt}_pred_")]
        show_cols  = [tgt] + model_cols
        st.line_chart(df.set_index("timestamp")[show_cols])

        rows = []
        for mc in model_cols:
            m = _metrics(df[tgt].values, df[mc].values)
            rows.append({"Model": mc.replace(f"{tgt}_pred_", ""), **m})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

    if target_mode == "both":
        st.subheader("PTF & SMF — Birlikte Gerçek")
        ptf = preds_df  # son döngüden kalmış olabilir, tekrar hesaplamak istersen results sözlüğü tut
        # (basitlik için yeniden üretmek yerine raw'dan merge etmek daha doğru; burada gösterimi kısa tuttum)

else:
    st.info("Sol panelden repo içi dosyayı seç, modelleri seç ve **▶ GitHub'dan Yükle • Eğit • Çiz** butonuna bas.")
