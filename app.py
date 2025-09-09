# app.py — Bakery Forecast (Upload & Run)

import io
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from pytz import timezone
import lightgbm as lgb

import gspread
from google.oauth2 import service_account

# ==============================
# Streamlit pagina-instellingen
# ==============================
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🥖", layout="wide")
st.title("🥖 Bakery Forecast – Upload & Run")
st.caption(
    "Deze pagina is voor **upload + forecast**. De resultaten verschijnen in je Sheet-tab "
    f"**{st.secrets.get('TAB_FORECASTS', 'forecasts')}** en zijn te bekijken in de Viewer."
)

# ==============================
# Helpers: secrets + Google auth
# ==============================

REQUIRED_SECRETS = ["SHEET_ID", "TAB_HISTORY", "TAB_FORECASTS", "LAT", "LON", "GCP_SERVICE_JSON"]

missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(
        "❌ Secrets ontbreken. Voeg minimaal toe: "
        + ", ".join(REQUIRED_SECRETS)
        + " (Settings → Secrets)."
    )
    st.stop()

SHEET_ID = st.secrets["SHEET_ID"]
TAB_HISTORY = st.secrets["TAB_HISTORY"]
TAB_FORECASTS = st.secrets["TAB_FORECASTS"]
LAT = float(st.secrets["LAT"])
LON = float(st.secrets["LON"])

# Service Account JSON (als platte tekst in secrets)
try:
    service_account_info = json.loads(st.secrets["GCP_SERVICE_JSON"])
except Exception as e:
    st.error(f"❌ Kon GCP_SERVICE_JSON niet lezen: {e}")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_gspread_client():
    creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return gspread.authorize(creds)

def open_ws(sheet_id: str, tab_name: str):
    gc = get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    try:
        return sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        sh.add_worksheet(title=tab_name, rows=1000, cols=20)
        return sh.worksheet(tab_name)

def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    ws = open_ws(sheet_id, tab_name)
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def write_sheet(sheet_id: str, tab_name: str, df: pd.DataFrame):
    ws = open_ws(sheet_id, tab_name)
    if df is None or df.empty:
        ws.clear()
        return
    # Converteer naar strings (Google Sheets update verwacht list of lists)
    values = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    ws.clear()
    ws.update(values)

# ==============================
# Datum normalisatie (TZ fix)
# ==============================

AMS = timezone("Europe/Amsterdam")

def normalize_date_col(df: pd.DataFrame, col="date") -> pd.DataFrame:
    """Maak datumkolom tz-naive en 00:00 (middernacht)."""
    if col not in df.columns:
        return df
    s = pd.to_datetime(df[col], errors="coerce")
    # Als tz-aware → naar AMS en daarna tz eraf
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(AMS).dt.tz_localize(None)
    df[col] = s.dt.normalize()
    return df

# ==============================
# Data inlezen & upload
# ==============================

@st.cache_data(ttl=120)
def load_history() -> pd.DataFrame:
    df = read_sheet(SHEET_ID, TAB_HISTORY)
    if df.empty:
        return pd.DataFrame(columns=["date", "sku_id", "product_name", "sales"])
    # Schoonmaak
    df = df.rename(columns=lambda c: c.strip())
    # dtypes
    df["sku_id"] = df.get("sku_id", "").astype(str)
    df["product_name"] = df.get("product_name", "").astype(str)
    df["sales"] = pd.to_numeric(df.get("sales", np.nan), errors="coerce")
    df = normalize_date_col(df, "date")
    df = df.dropna(subset=["date", "sku_id"])
    # sort & dedupe
    df = df.sort_values(["sku_id", "date"]).drop_duplicates(subset=["date", "sku_id"], keep="last").reset_index(drop=True)
    return df

def append_new_sales_csv(file) -> int:
    """Voeg alleen nieuwe regels toe op (date, sku_id)."""
    if file is None:
        return 0
    content = file.read()
    new_df = pd.read_csv(io.BytesIO(content))
    # vereiste kolommen
    req = {"date", "sku_id", "product_name", "sales"}
    if not req.issubset(set(new_df.columns)):
        raise ValueError(f"CSV mist kolommen: {sorted(list(req - set(new_df.columns)))}")

    # schoonmaak
    new_df = new_df.rename(columns=lambda c: c.strip())
    new_df["sku_id"] = new_df["sku_id"].astype(str)
    new_df["product_name"] = new_df["product_name"].astype(str)
    new_df["sales"] = pd.to_numeric(new_df["sales"], errors="coerce").fillna(0)
    new_df = normalize_date_col(new_df, "date")
    new_df = new_df.dropna(subset=["date", "sku_id"])

    hist = load_history()
    combo = pd.concat([hist, new_df], ignore_index=True)
    combo = combo.sort_values(["sku_id", "date"])
    combo = combo.drop_duplicates(subset=["date", "sku_id"], keep="last").reset_index(drop=True)

    # hoeveel nieuw?
    before = len(hist)
    after = len(combo)
    added = max(after - before, 0)

    write_sheet(SHEET_ID, TAB_HISTORY, combo)
    # cache invalideren
    load_history.clear()
    return added

# ==============================
# Weer (Open-Meteo)
# ==============================

def get_weather_future(lat: float, lon: float, start_date: pd.Timestamp, days: int) -> pd.DataFrame:
    """Dagelijkse temp/rain voor horizon, TZ AMS → tz-naive normalized."""
    end_date = start_date + pd.Timedelta(days=days - 1)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Europe%2FAmsterdam"
        f"&start_date={start_date.date()}&end_date={end_date.date()}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    w = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
        "temp_c": daily.get("temperature_2m_mean", []),
        "rain_mm": daily.get("precipitation_sum", []),
    })
    w = normalize_date_col(w, "date")
    return w

# ==============================
# Features & model
# ==============================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.weekday
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    # temp/rain kunnen ontbreken in history → laat als NaN
    return out

def train_per_sku(history_feat: pd.DataFrame):
    """Train per sku_id een LGBMRegressor op simpele features."""
    models = {}
    feature_cols = ["dow", "is_weekend", "temp_c", "rain_mm"]
    for sku, g in history_feat.groupby("sku_id"):
        g = g.dropna(subset=["sales"])
        if len(g) < 7:
            continue
        X = g[feature_cols]
        y = g["sales"]
        # Light params voor snelheid/robustheid
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X, y)
        models[sku] = model
    return models

def forecast_week(history: pd.DataFrame, lat: float, lon: float, tz="Europe/Amsterdam") -> pd.DataFrame:
    """Voorspel komende 7 dagen per SKU; schrijf geen NaN/negatieven."""
    if history.empty:
        raise ValueError("History is leeg.")

    # Horizon: vanaf morgen 7 dagen
    now_ams = datetime.now(AMS)
    tomorrow = (pd.Timestamp(now_ams.date()) + pd.Timedelta(days=1))
    horizon = pd.date_range(start=tomorrow, periods=7, freq="D")

    # Weer ophalen en normaliseren
    weather = get_weather_future(lat, lon, start_date=tomorrow, days=7)
    weather = normalize_date_col(weather, "date")  # **TZ FIX**

    # History features
    hist = history.copy()
    hist = normalize_date_col(hist, "date")        # **TZ FIX**
    hist = add_basic_features(hist)

    # Train
    models = train_per_sku(hist)

    # Toekomst frame per sku
    skus = hist[["sku_id", "product_name"]].drop_duplicates()
    future = pd.MultiIndex.from_product([skus["sku_id"], horizon], names=["sku_id", "date"]).to_frame(index=False)
    future = future.merge(skus, on="sku_id", how="left")
    future = normalize_date_col(future, "date")    # **TZ FIX**

    # Merge met weer
    future = future.merge(weather, on="date", how="left")
    future = add_basic_features(future)

    # Voorspellen
    feature_cols = ["dow", "is_weekend", "temp_c", "rain_mm"]
    preds = []
    for sku, g in future.groupby("sku_id"):
        Xf = g[feature_cols].copy()
        # Als temp/rain ontbreken → 0
        Xf["temp_c"] = pd.to_numeric(Xf["temp_c"], errors="coerce").fillna(0)
        Xf["rain_mm"] = pd.to_numeric(Xf["rain_mm"], errors="coerce").fillna(0)

        if sku in models:
            yhat = models[sku].predict(Xf)
        else:
            # fallback: gemiddelde van dit sku uit history
            base = hist.loc[hist["sku_id"] == sku, "sales"].mean()
            yhat = np.full(len(g), base if not np.isnan(base) else 0.0)

        yhat = np.clip(yhat, 0, None)  # geen negatieve aantallen
        preds.append(pd.DataFrame({
            "sku_id": g["sku_id"].values,
            "product_name": g["product_name"].values,
            "date": g["date"].values,
            "forecast_qty": np.rint(yhat).astype(int)
        }))

    out = pd.concat(preds, ignore_index=True)
    out = normalize_date_col(out, "date")          # consistent
    out = out.sort_values(["sku_id", "date"]).reset_index(drop=True)
    out["run_ts"] = datetime.now(AMS).strftime("%Y-%m-%d %H:%M:%S%z")
    return out

# ==============================
# UI — Upload + Forecast
# ==============================

st.subheader("1) Nieuwe verkoopregels uploaden (CSV, alleen nieuwe rijen)")

with st.container(border=True):
    up = st.file_uploader("CSV met kolommen: date, sku_id, product_name, sales", type=["csv"])
    if up is not None:
        try:
            added = append_new_sales_csv(up)
            st.success(f"✅ {added} rijen toegevoegd (met dedupe).")
        except Exception as e:
            st.error(f"❌ Upload mislukt: {e}")

st.subheader("2) Forecast draaien")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Run forecast voor komende 7 dagen", use_container_width=True):
        try:
            history = load_history()
            fc = forecast_week(history, LAT, LON)
            write_sheet(SHEET_ID, TAB_FORECASTS, fc)
            st.success("✅ Forecast geschreven naar Sheet-tab "
                       f"**{TAB_FORECASTS}**. (Run-timestamp staat erbij.)")
            with col2:
                st.dataframe(fc.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Forecast mislukt: {e}")

with col2:
    st.caption("Laatste 10 regels uit **history** (na upload/dedupe):")
    try:
        st.dataframe(load_history().sort_values("date").tail(10), use_container_width=True)
    except Exception:
        st.info("Nog geen history gevonden.")
