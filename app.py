# app.py – Upload & Forecast (voor baas)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import gspread
from google.oauth2 import service_account
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
import pytz

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🍞", layout="wide")
st.title("🍞 Bakery Forecast – Upload & Run")
st.markdown(
    "Deze pagina is voor **upload + forecast**. "
    "De resultaten verschijnen in je Sheet-tab **`forecasts`** en zijn te bekijken in de Viewer."
)

# -----------------------------
# Secrets & credentials
# -----------------------------
def _load_service_account_from_secrets():
    """
    Leest het service-account JSON uit Streamlit secrets.
    Ondersteunt ofwel string met JSON (tussen triple quotes) of direct een dict.
    """
    if "GCP_SERVICE_JSON" in st.secrets:
        raw = st.secrets["GCP_SERVICE_JSON"]
    elif "gcp_service_account" in st.secrets:
        raw = st.secrets["gcp_service_account"]
    else:
        raise KeyError("Service-account secret ontbreekt. Voeg 'GCP_SERVICE_JSON' toe bij Settings → Secrets.")

    if isinstance(raw, dict):
        return raw
    # anders string
    txt = str(raw)
    # Eventuele BOM/whitespace strippen
    txt = txt.strip().lstrip("\ufeff")
    return json.loads(txt)

try:
    SHEET_ID = st.secrets["SHEET_ID"]
    HISTORY_TAB = st.secrets.get("TAB_HISTORY", "history")
    FORECASTS_TAB = st.secrets.get("TAB_FORECASTS", "forecasts")
    LAT = float(str(st.secrets.get("LAT", "51.44")))
    LON = float(str(st.secrets.get("LON", "5.48")))
    SA_INFO = _load_service_account_from_secrets()
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
except Exception as e:
    st.error(
        "❌ Secrets ontbreken. Voeg minimaal toe: `GCP_SERVICE_JSON`, `SHEET_ID`, "
        "`TAB_HISTORY`, `TAB_FORECASTS`, `LAT`, `LON` (Settings → Secrets)."
    )
    st.stop()

credentials = service_account.Credentials.from_service_account_info(SA_INFO, scopes=SCOPES)
gc = gspread.authorize(credentials)

# -----------------------------
# Helpers Google Sheets
# -----------------------------
@st.cache_data(ttl=60)
def _get_worksheet(sheet_id: str, tab_name: str):
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=2000, cols=20)
    return ws

def read_history() -> pd.DataFrame:
    ws = _get_worksheet(SHEET_ID, HISTORY_TAB)
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(columns=["date", "sku_id", "product_name", "sales", "temp_c", "rain_mm"])
    df = pd.DataFrame(rows)
    # kolomnamen netjes
    df.columns = [c.strip() for c in df.columns]
    # verplichte kolommen
    for col in ["date", "sku_id", "product_name", "sales"]:
        if col not in df.columns:
            df[col] = np.nan
    # types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sku_id"] = df["sku_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    # optioneel weer (als aanwezig)
    if "temp_c" in df.columns:
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    else:
        df["temp_c"] = np.nan
    if "rain_mm" in df.columns:
        df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    else:
        df["rain_mm"] = np.nan

    df = df.dropna(subset=["date", "sku_id", "product_name", "sales"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df

def write_forecasts(forecast_df: pd.DataFrame):
    ws = _get_worksheet(SHEET_ID, FORECASTS_TAB)
    ws.clear()
    out = forecast_df.copy()
    # strings die Google Sheets prettig vindt
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["run_ts"] = out["run_ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    rows = [out.columns.tolist()] + out.astype(str).values.tolist()
    # batch update
    ws.update("A1", rows, value_input_option="USER_ENTERED")

# -----------------------------
# Feature engineering
# -----------------------------
AMS = pytz.timezone("Europe/Amsterdam")

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    # Fill weervelden als ze ontbreken
    if "temp_c" not in d.columns:
        d["temp_c"] = np.nan
    if "rain_mm" not in d.columns:
        d["rain_mm"] = np.nan
    # Vullen met eenvoudige imputaties
    d["temp_c"] = d.groupby("sku_id")["temp_c"].transform(lambda s: s.fillna(s.mean()))
    d["rain_mm"] = d["rain_mm"].fillna(0.0)
    return d

def build_future_frame(skus: pd.DataFrame, horizon_dates: pd.DatetimeIndex,
                       weather_future: pd.DataFrame) -> pd.DataFrame:
    # alle (sku, date) combinaties
    future = skus.assign(key=1).merge(pd.DataFrame({"date": horizon_dates, "key": 1}), on="key").drop("key", axis=1)
    # features
    future["dow"] = future["date"].dt.dayofweek
    future["is_weekend"] = (future["dow"] >= 5).astype(int)
    # weer toevoegen
    future = future.merge(weather_future, on="date", how="left")
    # als weer ontbreekt, 0/mean
    future["temp_c"] = future["temp_c"].fillna(weather_future["temp_c"].mean())
    future["rain_mm"] = future["rain_mm"].fillna(0.0)
    return future

# -----------------------------
# Weer (Open-Meteo) – t/m komende zondag
# -----------------------------
def horizon_dates_to_sunday(now_tz=AMS) -> pd.DatetimeIndex:
    today_local = datetime.now(tz=now_tz).date()
    tomorrow = today_local + timedelta(days=1)
    # tot komende zondag
    days_ahead = (6 - datetime.fromordinal(tomorrow.toordinal()).weekday()) % 7
    end = tomorrow + timedelta(days=days_ahead)
    rng = pd.date_range(start=pd.Timestamp(tomorrow), end=pd.Timestamp(end), freq="D", tz=now_tz).normalize()
    return rng

@st.cache_data(ttl=1800)  # 30 minuten
def get_weather_future(lat: float, lon: float, dates_index: pd.DatetimeIndex, tz=AMS) -> pd.DataFrame:
    # Open-Meteo gebruikt start_date/end_date in ISO-formaat (zonder tijd)
    start = dates_index.min().date().isoformat()
    end = dates_index.max().date().isoformat()
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
        "date": pd.to_datetime(daily["time"]).dt.tz_localize("Europe/Amsterdam"),
        "temp_c": daily["temperature_2m_mean"],
        "rain_mm": daily["precipitation_sum"],
    })
    return w

# -----------------------------
# Model & voorspellen
# -----------------------------
FEATURES = ["dow", "is_weekend", "temp_c", "rain_mm"]

def train_and_predict(history: pd.DataFrame, future_frame: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    run_ts = datetime.now(tz=AMS)

    for sku, df_sku in history.groupby("sku_id", sort=False):
        df_sku = df_sku.copy()
        df_sku = add_basic_features(df_sku)

        # trainset (alle records met geldige sales)
        train = df_sku.dropna(subset=["sales"]).copy()
        if train.empty:
            continue

        # eenvoudige validatie-indicatie (laatste 7 dagen als val)
        # (optioneel; we trainen op alle data voor productievoorspelling)
        X = train[FEATURES].fillna(0.0)
        y = train["sales"].astype(float).clip(lower=0)

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X, y)

        # toekomst voor deze sku
        fsku = future_frame[future_frame["sku_id"] == sku].copy()
        if fsku.empty:
            continue
        Xf = fsku[FEATURES].fillna(0.0)
        yhat = model.predict(Xf)
        yhat = np.maximum(yhat, 0)               # geen negatieve aantallen
        yhat = np.rint(yhat).astype(int)         # afronden naar hele stuks

        fsku["forecast_qty"] = yhat
        fsku["run_ts"] = run_ts
        out_rows.append(fsku[["sku_id", "product_name", "date", "forecast_qty", "run_ts"]])

    if not out_rows:
        return pd.DataFrame(columns=["sku_id", "product_name", "date", "forecast_qty", "run_ts"])

    res = pd.concat(out_rows, axis=0).sort_values(["date", "sku_id"]).reset_index(drop=True)
    return res

# -----------------------------
# UI – Upload + Run
# -----------------------------
with st.sidebar:
    st.subheader("📥 Upload nieuwe verkoopregels (CSV)")
    st.caption("Minimaal kolommen: date, sku_id, product_name, sales. (opt: temp_c, rain_mm)")
    uploaded = st.file_uploader("Kies CSV", type=["csv"])

# Toon bestaande history (head)
with st.expander("📚 Huidige history (eerste 20 rijen)"):
    hist_now = read_history()
    st.dataframe(hist_now.head(20), use_container_width=True)

if uploaded is not None:
    st.info("Nieuwe CSV ontvangen. Voeg toe aan history (de-dup op `(date, sku_id)`, laatste wint).")
    new_df = pd.read_csv(uploaded)
    # schoonmaken types
    new_df.columns = [c.strip() for c in new_df.columns]
    required = ["date", "sku_id", "product_name", "sales"]
    missing = [c for c in required if c not in new_df.columns]
    if missing:
        st.error(f"CSV mist kolommen: {missing}")
        st.stop()

    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
    new_df["sku_id"] = new_df["sku_id"].astype(str)
    new_df["product_name"] = new_df["product_name"].astype(str)
    new_df["sales"] = pd.to_numeric(new_df["sales"], errors="coerce")
    # optioneel
    if "temp_c" in new_df.columns:
        new_df["temp_c"] = pd.to_numeric(new_df["temp_c"], errors="coerce")
    if "rain_mm" in new_df.columns:
        new_df["rain_mm"] = pd.to_numeric(new_df["rain_mm"], errors="coerce")

    new_df = new_df.dropna(subset=["date", "sku_id", "product_name", "sales"])
    new_df = new_df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # merge met bestaande history
    hist = read_history()
    merged = pd.concat([hist, new_df], axis=0, ignore_index=True)
    # de-dup: laatste upload wint
    merged = merged.sort_values(["date"]).drop_duplicates(subset=["date", "sku_id"], keep="last")
    merged = merged.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # schrijf terug naar Sheet (history)
    ws_hist = _get_worksheet(SHEET_ID, HISTORY_TAB)
    ws_hist.clear()
    tmp = merged.copy()
    tmp["date"] = tmp["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = [tmp.columns.tolist()] + tmp.astype(str).values.tolist()
    ws_hist.update("A1", rows, value_input_option="USER_ENTERED")

    st.success(f"✅ History bijgewerkt: {len(new_df)} nieuwe rijen toegevoegd (na de-dup).")

st.markdown("---")
col1, col2 = st.columns([1, 2])
with col1:
    run_forecast = st.button("🚀 Forecast t/m komende zondag", type="primary")
with col2:
    st.caption("Gebruikt de volledige `history` + weersverwachting (Open-Meteo). Output gaat naar `forecasts`.")

if run_forecast:
    with st.spinner("Weer ophalen en modellen trainen…"):
        hist = read_history()
        if hist.empty:
            st.error("Geen history in Sheet. Upload eerst verkopen.")
            st.stop()
        # set timezone netjes
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(AMS, nonexistent="shift_forward", ambiguous="NaT", errors="coerce")
        hist = hist.dropna(subset=["date"])

        # Skulijst voor toekomstframe
        skus = hist[["sku_id", "product_name"]].drop_duplicates().sort_values("sku_id")

        # horizon + weer
        H = horizon_dates_to_sunday(AMS)
        weather_f = get_weather_future(LAT, LON, H, tz=AMS)

        future = build_future_frame(skus, H, weather_f)

        # Train + predict
        forecast_df = train_and_predict(hist, future)

        if forecast_df.empty:
            st.warning("Geen voorspellingen gegenereerd.")
        else:
            # schrijven naar Sheet
            write_forecasts(forecast_df)
            st.success(f"✅ Naar Sheet geschreven: {len(forecast_df)} regels → tab `{FORECASTS_TAB}`.")
            st.dataframe(forecast_df.head(30), use_container_width=True)

    # simpele kwaliteitschecks
    if not forecast_df.empty:
        neg = int((forecast_df["forecast_qty"] < 0).sum())
        st.caption(f"Negatieve aantallen: {neg} (geclipped naar ≥ 0). Alles afgerond naar hele stuks.")
