# -*- coding: utf-8 -*-
import os, sys, importlib.util, types, warnings, ast
from typing import Dict, List, Tuple
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
def _metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae  = float(np.mean(np.abs(yhat - y)))
    denom = np.maximum(np.abs(y), np.percentile(np.abs(y), 10) if y.size else 10.0) + 1e-6
    mape = float(np.mean(np.abs(yhat - y) / denom) * 100)
    ss_tot = np.sum((y - np.mean(y))**2); ss_res = np.sum((yhat - y)**2)
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
    "best_xgb_params_ptf",  "best_xgb_params_smf",
    "best_rf_params_ptf",   "best_rf_params_smf",
    "best_ridge_params_ptf","best_ridge_params_smf",
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
    # Eğitim penceresi ile hızlandır
    if max_train_days and max_train_days > 0:
        tmax = data["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=int(max_train_days))
        data = data[data["timestamp"].between(tmin, tmax)].copy()

    if data[target].isna().all():
        st.error(f"{target} için veri yok.")
        return pd.DataFrame()

    # Basit zaman bazlı split: son %15 test
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

# ======================================================
# UI — Kontroller
# ======================================================
with st.sidebar:
    st.header("Ayarlar")

    target_mode = st.selectbox("Hedef", ["ptf", "smf", "both"], index=0)

    data_path = st.text_input(
        "Veri yolu (Parquet/CSV)",
        value=r"C:\Users\ozkan\OneDrive\Desktop\Project Main\data\processed\fe_full_plus2_causal.parquet",
    )
    module_path = st.text_input(
        ".py modülü (özellik müh. + paramlar)",
        value=r"C:\Users\ozkan\OneDrive\Desktop\Project Main\data\processed\EDA_to_Model_EPIAS_Final_converted.py",
        help="Defterden dönüştürdüğün .py; içindeki build_feature_frame ve en iyi paramlar kullanılır. (Import sadece butona basınca ve güvenli şekilde)"
    )

    last_days = st.slider("Grafikte son kaç gün gösterilsin?", 3, 90, 14, step=1)
    max_train_days = st.slider("Eğitim penceresi (son N gün)", 14, 365, 90, step=7,
                               help="Eğitimi tüm tarih yerine son N gün ile sınırla (çok hızlandırır).")
    quick_mode = st.checkbox("Hızlı mod (daha az estimator / daha sığ ağaç)", value=True)

    # Model çoklu seçim + kısayollar (hiçbiri seçili değil)
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

    chosen_models = st.multiselect(
        "Koşturulacak modeller",
        all_models,
        default=st.session_state.chosen_models,
    )

    run_btn = st.button("▶ Modelleri Eğit ve Çiz", type="primary")
    if run_btn and not chosen_models:
        st.warning("Lütfen en az bir model seçin.")

# ======================================================
# Çalıştırma
# ======================================================
# Veri oku (ilk açılışta sadece veri okunur; eğitim yok)
try:
    raw = read_data(data_path)
except Exception as e:
    st.error(f"Veri okunamadı: {e}")
    st.stop()

if "ptf" not in raw.columns and "smf" not in raw.columns:
    st.error("Veride 'ptf' veya 'smf' kolonu bulunmuyor.")
    st.stop()

if run_btn and chosen_models:
    # Modülü sadece butona basınca ve güvenli import ile al
    module = safe_import_module(module_path) if module_path.strip() else None

    tabs = ("ptf", "smf") if target_mode == "both" else (target_mode,)
    results: Dict[str, pd.DataFrame] = {}

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
        results[tgt] = preds_df

    # Görseller
    for tgt in tabs:
        st.subheader(f"{tgt.upper()} — Gerçek vs Tahmin")
        df = results[tgt].sort_values("timestamp")
        # grafikte son X gün
        tmax = df["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        df = df[df["timestamp"].between(tmin, tmax)].copy()

        model_cols = [c for c in df.columns if c.startswith(f"{tgt}_pred_")]
        show_cols  = [tgt] + model_cols
        st.line_chart(df.set_index("timestamp")[show_cols])

        # metrikler
        rows = []
        for mc in model_cols:
            m = _metrics(df[tgt].values, df[mc].values)
            rows.append({"Model": mc.replace(f"{tgt}_pred_", ""), **m})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

    if target_mode == "both":
        st.subheader("PTF & SMF — Birlikte Gerçek")
        common = pd.merge_asof(
            results["ptf"].sort_values("timestamp")[["timestamp","ptf"]],
            results["smf"].sort_values("timestamp")[["timestamp","smf"]],
            on="timestamp"
        )
        tmax = common["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        common = common[common["timestamp"].between(tmin, tmax)]
        st.line_chart(common.set_index("timestamp")[["ptf", "smf"]])
else:
    st.info("Solda veri ve modül yolunu ayarlayın, modelleri seçin ve **▶ Modelleri Eğit ve Çiz** butonuna basın.")
