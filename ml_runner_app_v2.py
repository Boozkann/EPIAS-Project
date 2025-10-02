# -*- coding: utf-8 -*-
"""
EPIAS Project — Streamlit (GitHub-native, token'sız)
- Public GitHub reposundan (token gerektirmez) dosya listesi ve içerik.
- Repo default branch otomatik bulunur.
- Tüm repo ağacı tek istekle çekilir (git/trees?recursive=1).
- Veri (.parquet/.csv) ve opsiyonel modül (.py) dosyaları listeden seçilir.
- Lokal dosya yolu KULLANILMAZ.
"""

from __future__ import annotations
import os, sys, types, warnings, ast, io, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== REPO AYARLARI (senin repoun) ==================
GH_OWNER  = "Boozkann"
GH_REPO   = "EPIAS-Project"
DEFAULT_DATA_FILE = "final_data_with_features.parquet"  # Default dosya
DEFAULT_MODULE_FILE = "pipeline.py"  # Default modül
# ================================================================

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
st.set_page_config(page_title="EPİAŞ ML Tahmin Sistemi", layout="wide", initial_sidebar_state="collapsed")

# ========================== Custom CSS ==========================
st.markdown("""
<style>
    /* Ana başlık stili */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4f8 0%, #ffffff 50%, #e8f4f8 100%);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Alt başlık stili */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-left: 5px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    /* Metrik kartları */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    /* Sidebar stili */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Buton stilleri */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info box */
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================== GitHub yardımcıları ==========================
def _gh(url: str, timeout=30):
    """Basit GET, hatayı raise eder. Token yok (public repo)."""
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    return r

@st.cache_data(show_spinner=False)
def gh_default_branch(owner: str, repo: str) -> str:
    info_url = f"https://api.github.com/repos/{owner}/{repo}"
    r = _gh(info_url, timeout=30)
    return r.json().get("default_branch") or "main"

@st.cache_data(show_spinner=False)
def gh_tree(owner: str, repo: str, branch: str) -> List[dict]:
    """Tüm repo ağacı; her dosya/klasör path döner."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    r = _gh(url, timeout=60)
    data = r.json()
    return data.get("tree", [])

@st.cache_data(show_spinner=False)
def gh_raw_bytes(owner: str, repo: str, branch: str, path: str) -> bytes:
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path.lstrip('/')}"
    r = requests.get(raw, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def gh_raw_text(owner: str, repo: str, branch: str, path: str) -> str:
    return gh_raw_bytes(owner, repo, branch, path).decode("utf-8", errors="replace")
# ========================================================================

# ========================= Veri/Modül yükleyiciler =======================
@st.cache_data(show_spinner=True)
def read_data_from_github(owner: str, repo: str, branch: str, path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".parquet"):
        content = gh_raw_bytes(owner, repo, branch, path)
        df = pd.read_parquet(io.BytesIO(content))
    elif low.endswith(".csv"):
        text = gh_raw_text(owner, repo, branch, path)
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
            continue
    new_tree = ast.Module(body=new_body, type_ignores=[])
    code = compile(new_tree, filename=virtual_filename, mode="exec")
    mod = types.ModuleType("user_pipeline_sandbox")
    mod.__file__ = f"gh://{GH_OWNER}/{GH_REPO}/{virtual_filename}"
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
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "R²": r2}

def build_features_via_module(df: pd.DataFrame, module, target: str) -> pd.DataFrame:
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

    ridge_params = get_optimized_params(module, "ridge", target) or {}
    factories["ridge"] = lambda: Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(**{k: v for k, v in ridge_params.items() if k in Ridge().get_params()})),
    ])

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

    rf_params = get_optimized_params(module, "rf", target) or {}
    base = dict(random_state=42, n_estimators=200 if quick_mode else 500, n_jobs=-1,
                max_depth=16 if quick_mode else None)
    base.update(rf_params)
    factories["randomforest"] = lambda: RandomForestRegressor(**base)

    cart_params = get_optimized_params(module, "cart", target) or {}
    if quick_mode and "max_depth" not in cart_params:
        cart_params["max_depth"] = 10
    factories["cart"] = lambda: CART(random_state=42, **cart_params)

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

def create_interactive_chart(df: pd.DataFrame, target: str, model_cols: List[str], title: str):
    """Plotly ile interaktif grafik oluştur"""
    fig = go.Figure()
    
    # Gerçek değerler
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[target],
        name="Gerçek Değer",
        mode='lines',
        line=dict(color='#2c3e50', width=3),
        hovertemplate='<b>Gerçek</b><br>Tarih: %{x}<br>Değer: %{y:.2f}<extra></extra>'
    ))
    
    # Model tahminleri
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for i, mc in enumerate(model_cols):
        model_name = mc.replace(f"{target}_pred_", "").upper()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[mc],
            name=model_name,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
            hovertemplate=f'<b>{model_name}</b><br>Tarih: %{{x}}<br>Değer: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title="Tarih",
        yaxis_title="Değer (MWh)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#2c3e50',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            tickformat='%d %b\n%Y',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1'
        )
    )
    
    return fig
# ========================================================================

# ================================ UI ====================================
# Ana başlık
st.markdown('<div class="main-header">⚡ EPİAŞ Enerji Tahmin Sistemi</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Konfigürasyon")

    # 1) Default branch öğren
    try:
        GH_BRANCH = gh_default_branch(GH_OWNER, GH_REPO)
    except Exception as e:
        st.error(f"❌ Repo bilgisi alınamadı: {e}")
        st.stop()
    
    # 2) Tüm repo ağacını çek ve filtrele
    parquet_csv_files: List[str] = []
    py_files: List[str] = []
    error_tree = None
    try:
        tree = gh_tree(GH_OWNER, GH_REPO, GH_BRANCH)
        for node in tree:
            if node.get("type") != "blob":
                continue
            p = node.get("path", "")
            low = p.lower()
            if low.endswith((".parquet", ".csv")):
                parquet_csv_files.append(p)
            elif low.endswith(".py"):
                py_files.append(p)
    except Exception as e:
        error_tree = str(e)

    if error_tree:
        st.error(f"❌ Repo ağacı alınamadı: {error_tree}")
        st.stop()

    if not parquet_csv_files:
        st.error("❌ Repoda .parquet veya .csv dosyası bulunamadı.")
        st.stop()

    # Default dosyayı bul
    default_data_idx = 0
    for i, f in enumerate(sorted(parquet_csv_files)):
        if DEFAULT_DATA_FILE in f:
            default_data_idx = i
            break

    default_module_idx = 0
    for i, f in enumerate(sorted(py_files)):
        if DEFAULT_MODULE_FILE in f:
            default_module_idx = i
            break

    # Gelişmiş ayarlar expander'ı
    with st.expander("🔧 Gelişmiş Ayarlar", expanded=False):
        data_choice = st.selectbox(
            "Veri dosyası",
            options=sorted(parquet_csv_files),
            index=default_data_idx
        )
        
        module_choice = st.selectbox(
            "Özellik/Param Modülü",
            options=["(kullanma)"] + sorted(py_files),
            index=default_module_idx + 1 if py_files else 0
        )

    st.markdown("---")
    
    # Model seçimleri
    st.subheader("📊 Model Seçimi")
    target_mode = st.radio("Hedef Değişken", ["PTF", "SMF", "Her İkisi"], index=2)
    target_mode = target_mode.lower().replace("ptf", "ptf").replace("smf", "smf").replace("Her İkisi", "both")

       
    if "chosen_models" not in st.session_state:
        st.session_state.chosen_models = ["lgbm", "xgb", "randomforest", "voting"]
    
    all_models = ["ridge", "lgbm", "xgb", "cart", "randomforest", "voting"]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Temizle", use_container_width=True):
            st.session_state.chosen_models = []
            st.rerun()
    with col2:
        if st.button("Tümü", use_container_width=True):
            st.session_state.chosen_models = all_models.copy()
            st.rerun()

    chosen_models = st.multiselect(
        "Modeller", 
        all_models, 
        default=st.session_state.chosen_models,
        format_func=lambda x: x.upper()
    )
    st.session_state.chosen_models = chosen_models

    st.markdown("---")

        
    # Parametreler
    st.subheader("Parametreler")
    last_days = st.slider("Görselleştirme Dönemi (gün)", 7, 90, 30, step=7)
    max_train_days = st.slider("Eğitim Penceresi (gün)", 30, 365, 90, step=30)
    quick_mode = st.checkbox("Hızlı Mod (daha az iterasyon)", value=True)

    st.markdown("---")
    run_btn = st.button("Analizi Başlat", type="primary", use_container_width=True)
    
    if run_btn and not chosen_models:
        st.warning("⚠️ Lütfen en az bir model seçin.")
# ========================================================================

# Ana içerik alanı - default dosyayı otomatik kullan
if 'data_choice' not in locals():
    data_choice = sorted(parquet_csv_files)[default_data_idx]
if 'module_choice' not in locals():
    module_choice = sorted(py_files)[default_module_idx] if py_files else "(kullanma)"

# =============================== Veri yükleme ==============================
try:
    with st.spinner(f"📥 Veri yükleniyor..."):
        raw = read_data_from_github(GH_OWNER, GH_REPO, GH_BRANCH, data_choice)
    
    # Veri özeti göster
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Toplam Kayıt", f"{len(raw):,}")
    with col2:
        st.metric("📅 Başlangıç", raw["timestamp"].min().strftime("%d.%m.%Y"))
    with col3:
        st.metric("📅 Bitiş", raw["timestamp"].max().strftime("%d.%m.%Y"))
    with col4:
        days = (raw["timestamp"].max() - raw["timestamp"].min()).days
        st.metric("⏱️ Dönem", f"{days} gün")
    
except Exception as e:
    st.error(f"❌ Veri okunamadı: {e}")
    st.stop()

if "ptf" not in raw.columns and "smf" not in raw.columns:
    st.error("❌ Veride 'ptf' veya 'smf' kolonu bulunmuyor.")
    st.stop()

# =============================== Çalıştırma ==============================
if run_btn and chosen_models:
    # Modül yükle
    module = None
    if module_choice and module_choice != "(kullanma)":
        try:
            with st.spinner(f"📦 Modül yükleniyor..."):
                module_src = gh_raw_text(GH_OWNER, GH_REPO, GH_BRANCH, module_choice)
                module = safe_import_module_from_source(module_src, virtual_filename=module_choice.split("/")[-1])
            st.success("Modül başarıyla yüklendi")
        except Exception as e:
            st.warning(f"⚠️ Modül import edilemedi: {e}")
            module = None

    tabs = ("ptf", "smf") if target_mode == "both" else (target_mode,)

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

    def train_and_predict(df: pd.DataFrame, features: List[str], target: str,
                          models: List[str], factories, max_train_days: int) -> pd.DataFrame:
        data = df.sort_values("timestamp").reset_index(drop=True)
        if max_train_days and max_train_days > 0:
            tmax = data["timestamp"].max()
            tmin = tmax - pd.Timedelta(days=int(max_train_days))
            data = data[data["timestamp"].between(tmin, tmax)].copy()

        if data[target].isna().all():
            st.error(f"❌ {target} için veri yok.")
            return pd.DataFrame()

        split = int(len(data) * 0.85)
        train = data.iloc[:split].copy()
        test  = data.iloc[split:].copy()

        Xtr, ytr = train[features].fillna(0.0).to_numpy(), train[target].astype(float).to_numpy()
        Xte, yte = test[features].fillna(0.0).to_numpy(),  test[target].astype(float).to_numpy()
        out = test[["timestamp", target]].copy()

        progress_bar = st.progress(0)
        for idx, key in enumerate(models):
            fac = factories.get(key)
            if fac is None:
                st.warning(f"⚠️ {key} modeli atlandı (kütüphane yüklenmemiş)")
                continue
            col = f"{target}_pred_{key}"
            with st.spinner(f"🔄 {key.upper()} eğitiliyor..."):
                m = fac()
                m.fit(Xtr, ytr)
                out[col] = m.predict(Xte)
            progress_bar.progress((idx + 1) / len(models))
        
        progress_bar.empty()
        return out

    # Her hedef için işlem
    for tgt in tabs:
        st.markdown(f'<div class="sub-header">📈 {tgt.upper()} Analizi</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner(f"⚙️ {tgt.upper()} için veriler hazırlanıyor..."):
                feat_df, feat_cols = prepare_dataset(raw, module, tgt)
            
            if feat_df.empty or not feat_cols:
                st.warning(f"⚠️ {tgt} için özellik bulunamadı.")
                continue

            st.info(f"ℹ️ **Kullanılan özellik sayısı:** {len(feat_cols)}")
            
            factories = make_model_factories(module, tgt, quick_mode)
            available = [k for k in chosen_models if factories.get(k) is not None]
            
            if not available:
                st.warning(f"⚠️ {tgt} için model yok.")
                continue

            st.info(f"🎯 **Eğitilecek modeller:** {', '.join([m.upper() for m in available])}")
            
            result_df = train_and_predict(feat_df, feat_cols, tgt, available, factories, max_train_days)
            
            if result_df.empty:
                st.warning(f"⚠️ {tgt} için sonuç oluşturulamadı.")
                continue

            # Son N gün için filtrele
            end_ts = result_df["timestamp"].max()
            start_ts = end_ts - pd.Timedelta(days=last_days)
            plot_df = result_df[result_df["timestamp"] >= start_ts].copy()

            # Grafik
            model_cols = [c for c in result_df.columns if c.startswith(f"{tgt}_pred_")]
            if plot_df.empty or not model_cols:
                st.warning(f"⚠️ {tgt} için grafik verisi yok.")
                continue

            st.markdown("#### 📊 Tahmin Grafikleri")
            fig = create_interactive_chart(
                plot_df, 
                tgt, 
                model_cols,
                f"{tgt.upper()} - Gerçek vs Tahmin (Son {last_days} Gün)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrikler
            st.markdown("#### 📏 Model Performans Metrikleri")
            
            metrics_data = []
            for mc in model_cols:
                model_name = mc.replace(f"{tgt}_pred_", "").upper()
                sub = plot_df[[tgt, mc]].dropna()
                if len(sub) < 2:
                    continue
                m = _metrics(sub[tgt].values, sub[mc].values)
                metrics_data.append({
                    "Model": model_name,
                    "RMSE": f"{m['RMSE']:.2f}",
                    "MAE": f"{m['MAE']:.2f}",
                    "MAPE": f"{m['MAPE%']:.2f}%",
                    "R²": f"{m['R²']:.4f}"
                })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Metrik kartları
                cols = st.columns(len(metrics_data))
                for idx, row in enumerate(metrics_data):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">{row['Model']}</h4>
                            <p style="margin: 0.25rem 0;"><strong>RMSE:</strong> {row['RMSE']}</p>
                            <p style="margin: 0.25rem 0;"><strong>MAE:</strong> {row['MAE']}</p>
                            <p style="margin: 0.25rem 0;"><strong>MAPE:</strong> {row['MAPE']}</p>
                            <p style="margin: 0.25rem 0;"><strong>R²:</strong> {row['R²']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detaylı tablo
                with st.expander("📋 Detaylı Metrik Tablosu", expanded=False):
                    st.dataframe(
                        metrics_df,
                        use_container_width=True,
                        hide_index=True
                    )

            # Günlük hata analizi
            st.markdown("#### 📉 Günlük Hata Analizi")
            
            daily_errors = plot_df.copy()
            daily_errors['date'] = daily_errors['timestamp'].dt.date
            
            error_data = []
            for mc in model_cols:
                model_name = mc.replace(f"{tgt}_pred_", "").upper()
                daily_errors[f'error_{model_name}'] = abs(daily_errors[tgt] - daily_errors[mc])
            
            error_cols = [c for c in daily_errors.columns if c.startswith('error_')]
            
            if error_cols:
                fig_error = go.Figure()
                
                for ec in error_cols:
                    model_name = ec.replace('error_', '')
                    fig_error.add_trace(go.Scatter(
                        x=daily_errors['timestamp'],
                        y=daily_errors[ec],
                        name=model_name,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
                
                fig_error.update_layout(
                    title={
                        'text': f'{tgt.upper()} - Mutlak Hata Trendi',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': '#2c3e50'}
                    },
                    xaxis_title="Tarih",
                    yaxis_title="Mutlak Hata (MWh)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1', tickformat='%d %b\n%Y'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
                )
                
                st.plotly_chart(fig_error, use_container_width=True)

            st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ {tgt} işlenirken hata: {e}")
            import traceback
            with st.expander("🔍 Hata Detayları"):
                st.code(traceback.format_exc())

    st.success("Analiz tamamlandı!")
    
    # İndirme seçenekleri
    st.markdown("---")
    st.markdown('<div class="sub-header">💾 Veri İndirme</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 PTF Tahminlerini İndir", use_container_width=True):
            st.info("İndirme özelliği yakında eklenecek")
    with col2:
        if st.button("📥 SMF Tahminlerini İndir", use_container_width=True):
            st.info("İndirme özelliği yakında eklenecek")

else:
    # Başlangıç ekranı
    st.info("👋 **Hoş Geldiniz!** Bu uygulama EPİAŞ enerji piyasası verilerini analiz eder ve makine öğrenmesi modelleri ile tahminler oluşturur.")
    
    st.markdown("### 📝 Kullanım Adımları:")
    st.markdown("""
1. **Sol panelden** analiz parametrelerini ayarlayın
2. **Model Seçimi** yapın (LGBM, XGBoost, Random Forest, vb.)
3. **Görselleştirme Dönemi** ve eğitim parametrelerini belirleyin
4. **"Analizi Başlat"** butonuna tıklayın
    """)
    
    st.markdown("### 🎯 Özellikler:")
    st.markdown("""
- PTF ve SMF Tahminleri
- Çoklu Model Karşılaştırması
- İnteraktif Grafikler
- Detaylı Performans Metrikleri
- Günlük Hata Analizi
    """)
    
    # Örnek veri önizlemesi
    st.markdown("### 📊 Veri Önizleme")
    
    preview_df = raw.head(10).copy()
    if 'timestamp' in preview_df.columns:
        preview_df['timestamp'] = preview_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    
    st.info("💡 **İpucu:** Başlamak için sol panelden '**Analizi Başlat**'")
