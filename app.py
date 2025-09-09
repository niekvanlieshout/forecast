# app.py — Upload & Forecast (history bijschrijven, forecasts 7 dagen overschrijven)
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials
from lightgbm import LGBMRegressor

# =========================
# Basis UI
# =========================
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🥖", layout="centered")
st.title("🥖 Bakery Forecast – Upload & Run")

st.caption(
    "• **Upload** nieuwe verkoopregels (alleen nieuwe rijen; app voegt toe & dedupliceert).  \n"
    f"• **Forecast** schrijft altijd **alleen de komende 7 dagen vanaf morgen** naar tab "
    f"**{st.secrets.get('TAB_FORECASTS', 'forecasts')}** en **overschrijft** die tab volledig per run."
)

# =========================
# Secrets & verbinding
# =========================
REQUIRED = ["GCP_SERVICE_JSON", "SHEET_ID", "TAB_HISTORY", "TAB_FORECASTS", "LAT", "LON"]
missing = [k for k in REQUIRED if k not in st.secrets]
if missing:
    st.error("Secrets ontbreken: " + ", ".join(missing) + " (vul aan bij Settings → Secrets).")
    st.stop()

SHEET_ID      = st.secrets["SHEET_ID"]
HISTORY_TAB   = st.secrets["TAB_HISTORY"]
FORECASTS_TAB = st.secrets["TAB_FORECASTS"]
LAT           = float(st.secrets["LAT"])
LON           = float(st.secrets["LON"])
SERVICE_INFO  = json.loads(st.secrets["GCP_SERVICE_JSON"])

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(SERVICE_INFO, scopes=SCOPES)
gc = gspread.authorize(creds)

# =========================
# Helpers Google Sheets
# =========================
def open_ws(sheet_id: str, title: str):
    sh = gc.open_by_key(sheet_id)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=2000, cols=20)

def read_sheet(ws, expected_cols: list[str]) -> pd.DataFrame:
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame(columns=expected_cols)
    cols = [c.strip() for c in rows[0]]
    df = pd.DataFrame(rows[1:], columns=cols)
    # Zorg dat alle verwachte kolommen bestaan
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Houd alleen verwachte volgorde
    df = df[expected_cols]
    return df

def write_sheet(ws, df: pd.DataFrame):
    # Alles naar strings; expliciet 'A1' update zodat range klopt
    ws.clear()
    values = [list(df.columns)] + df.astype(str).values.tolist()
    ws.update("A1", values)

# =========================
# Data schoonmaak
# =========================
HISTORY_COLS = ["date", "sku_id", "product_name", "sales", "temp_c", "rain_mm", "derving_qty"]

def to_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False, format=None)
    # tz-naive (strings naar datum)
    dt = dt.dt.tz_localize(None)
    return dt

def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Date verplicht
    df["date"] = to_date_series(df["date"])
    # Minimale kolommen
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()

    # Numeriek maken waar mogelijk (optioneel weer/derving)
    for c in ["sales", "temp_c", "rain_mm", "derving_qty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sales: geen negatieve, afronden
    df["sales"] = df["sales"].fillna(0).clip(lower=0).round().astype(int)

    # Overige: laat NaN staan (ok)
    # Verwijder rijen zonder geldige datum of sku
    df = df[~df["date"].isna() & df["sku_id"].ne("")]
    # Sort
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df

# =========================
# Upload sectie
# =========================
st.subheader("1) Nieuwe verkoopregels uploaden (CSV, alleen nieuwe rijen)")
st.caption("CSV-kolommen: **date, sku_id, product_name, sales**. "
           "Optioneel ook **temp_c, rain_mm, derving_qty**.")

uploaded = st.file_uploader("Drag & drop of kies een CSV", type=["csv"], label_visibility="collapsed")

hist_ws = open_ws(SHEET_ID, HISTORY_TAB)
hist_df = clean_history(read_sheet(hist_ws, HISTORY_COLS))

if uploaded is not None:
    try:
        new_df_raw = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception:
        # soms helpt encoding='utf-8-sig'
        uploaded.seek(0)
        new_df_raw = pd.read_csv(io.BytesIO(uploaded.read()), encoding="utf-8-sig")

    # Zorg dat alle kolommen bestaan
    for c in HISTORY_COLS:
        if c not in new_df_raw.columns:
            new_df_raw[c] = np.nan

    new_df = clean_history(new_df_raw[HISTORY_COLS])

    # Bijschrijven + dedupe: laatst geüploade wint
    combined = pd.concat([hist_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "sku_id"], keep="last")
    combined = combined.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # Schrijf volledig terug naar Sheet (history wordt groter, niets overschreven buiten dedupe)
    write_sheet(hist_ws, combined)

    # UI feedback
    added = len(combined) - len(hist_df)
    st.success(f"✅ {max(0, added)} rijen toegevoegd (na dedupe).")
    with st.expander("Voorbeeld van laatst toegevoegde / bijgewerkte regels"):
        st.dataframe(new_df.tail(10), use_container_width=True)

    # ververs lokale hist_df voor de forecasting hieronder
    hist_df = combined

# Toon altijd laatste 10 regels uit history
st.caption("Laatst bekende history (voorbeeld):")
if len(hist_df):
    st.dataframe(hist_df.tail(10), use_container_width=True)
else:
    st.info("Nog geen history aanwezig. Upload eerst een CSV.")

# =========================
# Weer (Open-Meteo) + horizon
# =========================
def get_horizon_dates(days: int = 7, tzname: str = "Europe/Amsterdam") -> pd.DatetimeIndex:
    now_local = pd.Timestamp.now(tz=tzname).normalize()
    tomorrow = now_local + pd.Timedelta(days=1)
    # tz-naive datums om te vergelijken/wegschrijven
    rng = pd.date_range(start=tomorrow.tz_localize(None), periods=days, freq="D")
    return rng

def get_weather_future(lat: float, lon: float, dates: pd.DatetimeIndex) -> pd.DataFrame:
    start = dates.min().date().isoformat()
    end   = dates.max().date().isoformat()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Europe%2FAmsterdam"
        f"&start_date={start}&end_date={end}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j["daily"]
    w = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).tz_localize(None),
        "temp_c": daily["temperature_2m_mean"],
        "rain_mm": daily["precipitation_sum"],
    })
    return w

# =========================
# Feature bouw
# =========================
def build_train_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["dow"] = x["date"].dt.dayofweek
    x["is_weekend"] = x["dow"].isin([5, 6]).astype(int)

    # rollings/means per sku (alleen uit verleden)
    x = x.sort_values(["sku_id", "date"])
    x["lag7"] = x.groupby("sku_id")["sales"].shift(7)
    x["mean7"] = x.groupby("sku_id")["sales"].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    # targets
    x["y"] = x["sales"]
    # vul exogene features als ontbreken
    for c in ["temp_c", "rain_mm"]:
        if c not in x.columns:
            x[c] = np.nan
    return x

def build_future_frame(skus: pd.DataFrame, weather_future: pd.DataFrame) -> pd.DataFrame:
    """Maak elke toekomstige dag × sku_id met basisfeatures + weer."""
    dates = weather_future["date"].copy()
    future = pd.MultiIndex.from_product([skus["sku_id"].unique(), dates], names=["sku_id", "date"]).to_frame(index=False)
    # Voeg product_name op sku-niveau toe (laatste bekende naam)
    pname = skus.sort_values("date").groupby("sku_id")["product_name"].last().reset_index()
    future = future.merge(pname, on="sku_id", how="left")
    # Weer
    future = future.merge(weather_future, on="date", how="left")
    # Basis kalendar
    future["dow"] = future["date"].dt.dayofweek
    future["is_weekend"] = future["dow"].isin([5, 6]).astype(int)
    return future

# =========================
# Train & Predict (per SKU)
# =========================
def train_and_predict_7days(hist_df: pd.DataFrame) -> pd.DataFrame:
    # Vereiste kolommen
    if hist_df.empty:
        raise ValueError("History is leeg.")
    # features voor training
    tr = build_train_features(hist_df)

    # Weer toekomst + horizon
    horizon = get_horizon_dates(days=7, tzname="Europe/Amsterdam")
    weather_future = get_weather_future(LAT, LON, horizon)

    future = build_future_frame(hist_df[["date", "sku_id", "product_name"]], weather_future)

    preds = []
    for sku, g in tr.groupby("sku_id"):
        # voldoende data?
        if g["y"].notna().sum() < 14:
            # te weinig: simpele fallback = laatste 7d gemiddelde of overall mean
            last_mean = g["y"].tail(7).mean() if g["y"].tail(7).notna().any() else g["y"].mean()
            last_mean = 0.0 if pd.isna(last_mean) else float(last_mean)
            tmp = future[future["sku_id"] == sku].copy()
            tmp["pred"] = last_mean
            preds.append(tmp)
            continue

        # Train set (alleen rijen met target)
        feat_cols = ["dow", "is_weekend", "temp_c", "rain_mm", "mean7"]
        g_model = g.copy()
        # mean7 finaliseren (NaN → per sku mean)
        sku_mean = g_model["y"].mean()
        g_model["mean7"] = g_model["mean7"].fillna(sku_mean)

        X = g_model[feat_cols].fillna(g_model[feat_cols].median(numeric_only=True))
        y = g_model["y"]

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X, y)

        # Future features (mean7 → laatst bekende mean7 per sku)
        fsku = future[future["sku_id"] == sku].copy()
        last_mean7 = g_model["mean7"].tail(1).values[0] if not g_model["mean7"].tail(1).isna().all() else sku_mean
        fsku["mean7"] = last_mean7
        Xf = fsku[feat_cols].fillna(g_model[feat_cols].median(numeric_only=True))
        fsku["pred"] = model.predict(Xf)
        preds.append(fsku)

    out = pd.concat(preds, ignore_index=True)
    # Postprocess: geen negatieve, afronden naar hele stuks
    out["pred"] = np.maximum(0, out["pred"]).round().astype(int)

    # Outputformat: sku_id, product_name, date, forecast_qty, run_ts
    run_ts = pd.Timestamp.now(tz="Europe/Amsterdam")
    res = out[["sku_id", "product_name", "date", "pred"]].rename(columns={"pred": "forecast_qty"}).copy()
    res["run_ts"] = run_ts
    # Sorteer mooi
    res = res.sort_values(["date", "sku_id"]).reset_index(drop=True)
    return res

# =========================
# Forecast sectie
# =========================
st.subheader("2) Forecast draaien")

if st.button("Run forecast voor komende 7 dagen"):
    if hist_df.empty:
        st.error("Forecast mislukt: history is leeg. Upload eerst verkoopdata.")
        st.stop()
    try:
        # Train + predict
        forecast_df = train_and_predict_7days(hist_df)

        # Schrijf **volledig** de forecasts-tab opnieuw (alleen komende 7 dagen)
        f_ws = open_ws(SHEET_ID, FORECASTS_TAB)
        # Zorg voor string/seriële velden
        out = forecast_df.copy()
        # dates als yyyy-mm-dd 00:00:00 (zoals jouw sheet)
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d 00:00:00")
        # run_ts leesbaar
        out["run_ts"] = pd.to_datetime(out["run_ts"]).dt.tz_convert("Europe/Amsterdam").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out = out[["sku_id", "product_name", "date", "forecast_qty", "run_ts"]]

        write_sheet(f_ws, out)

        st.success(f"✅ Forecast geschreven: {len(out)} regels (alleen komende 7 dagen, oude forecasts overschreven).")
        st.dataframe(out.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Forecast mislukt: {e}")

# Einde
