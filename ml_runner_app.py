# -*- coding: utf-8 -*-
import os, sys, types, warnings, ast
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ======================================================
# Soft dependency shims – dış modül importunda eksikler çökertmesin
# ======================================================
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
        "plot": lambda *a, **k: None,
        "show": lambda: None,
        "style": types.SimpleNamespace(use=lambda *a, **k: None),
    })

try:
    import seaborn as sns  # noqa
except Exception:
    _inject_fake("seaborn", {
        "set": lambda *a, **k: None,
        "lineplot": lambda *a, **k: None,
        "barplot": lambda *a, **k: None,
        "heatmap": lambda *a, **k: None,
    })

try:
    from statsmodels.tsa.seasonal import seasonal_decompose  # noqa
except Exception:
    import pandas as _pd
    from collections import namedtuple

    Decomp = namedtuple("DecomposeResult", ["observed", "trend", "seasonal", "resid"])


    def _fake_decompose(series, model="additive", period=None):
        s = _pd.Series(series).astype(float)
        nan = _pd.Series([float("nan")] * len(s), index=s.index)
        return Decomp(observed=s, trend=nan, seasonal=nan, resid=nan)


    _inject_fake("statsmodels")
    _inject_fake("statsmodels.tsa")
    _inject_fake("statsmodels.tsa.seasonal", {"seasonal_decompose": _fake_decompose})

try:
    import optuna  # noqa
except Exception:
    _inject_fake("optuna", {"create_study": lambda *a, **k: None})

# ======================================================
# İsteğe bağlı model kütüphaneleri
# ======================================================
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

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EPİAŞ PTF/SMF — ML Runner", layout="wide")
st.title("⚡ EPİAŞ PTF/SMF — ML Model Runner (Gerçek vs Tahmin)")


# ======================================================
# Yardımcılar
# ======================================================
def resolve_repo_path(p: str) -> str:
    r"""
    Verilen yolu çalışılan ortama göre güvenli biçimde çözer:
    - Eğer verilen yol zaten makinede mevcutsa (lokalde C:\... gibi), aynen kullanır.
    - Aksi halde (örn. Streamlit Cloud’da C:\... geçersizse) sadece dosya adını alır,
      bu dosyayı uygulama klasöründe veya data/ klasöründe arar.
    """
    if not p:
        return p
    # 1) Olduğu gibi genişlet ve varsa doğrudan kullan
    P = Path(p).expanduser()
    if P.exists():
        return str(P.resolve())
    # 2) Bulunamadıysa (ör. Cloud’da C:\...), adı üzerinden repo içinde ara
    name = Path(p).name
    base = Path(__file__).parent
    candidates = [
        base / name,
        base / "data" / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # 3) Yine de bulunamazsa, orijinali döndür (read_data anlamlı hata verecek)
    return str(P)


def _metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y, dtype=float);
    yhat = np.asarray(yhat, dtype=float)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    denom = np.maximum(np.abs(y), np.percentile(np.abs(y), 10) if y.size else 10.0) + 1e-6
    mape = float(np.mean(np.abs(yhat - y) / denom) * 100)
    ss_tot = np.sum((y - np.mean(y)) ** 2);
    ss_res = np.sum((yhat - y) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot if ss_tot > 1e-12 else 1e-12)))
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "R2": r2}


@st.cache_data(show_spinner=False)
def read_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Veri bulunamadı: {path}")
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Desteklenmeyen format. Parquet veya CSV verin.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# -------- Güvenli import: sadece fonksiyonlar + param sözlükleri --------
ALLOWED_GLOBAL_NAMES = {
    "CUT_TS",
    "best_lgbm_params", "best_xgb_params", "best_rf_params", "best_cart_params", "best_ridge_params",
    "lgbm_best_params", "xgb_best_params", "rf_best_params", "ridge_best_params", "cart_best_params",
    "best_lgbm_params_ptf", "best_lgbm_params_smf",
    "best_xgb_params_ptf", "best_xgb_params_smf",
    "best_rf_params_ptf", "best_rf_params_smf",
    "best_ridge_params_ptf", "best_ridge_params_smf",
    "best_cart_params_ptf", "best_cart_params_smf",
}


def safe_import_module(py_path: str):
    """
    .py dosyasını okur; fonksiyonlar/sınıflar/import'lar ve ALLOWED_GLOBAL_NAMES'e
    atanmış global değişkenleri bırakır; top-level yürütülebilir kodları eler.
    Tanımsız isimlerden dolayı (örn. CUT) oluşan NameError'ları None ile doldurup tekrar dener.
    """
    if not py_path or not os.path.isfile(py_path):
        return None
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=py_path)
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
            # Expr/If/For/While/With/Try gibi yürütücü bloklar import sırasında elenir
            continue

    new_tree = ast.Module(body=new_body, type_ignores=[])
    code = compile(new_tree, filename=py_path, mode="exec")

    mod = types.ModuleType("user_pipeline_sandbox")
    mod.__file__ = py_path

    # Yaygın eksik isimleri önceden None olarak tohumla
    preset_missing = {"CUT", "CUT_TRAIN", "CUT_VALID", "CUTOFF", "SPLIT_TS"}
    for nm in preset_missing:
        mod.__dict__.setdefault(nm, None)

    # NameError yakalanırsa eksik ismi None yap ve tekrar dene (maks. 5 kez)
    for _ in range(5):
        try:
            exec(code, mod.__dict__)  # sadece güvenli gövde
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


def build_features_via_module(df: pd.DataFrame, module, target: str) -> pd.DataFrame:
    """Modülde build_feature_frame varsa çağırır; yoksa df'i döner."""
    if module is None:
        return df
    fn = getattr(module, "build_feature_frame", None)
    if callable(fn):
        try:

