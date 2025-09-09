# app.py — Upload & Forecast (voor baas)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import gspread
from google.oauth2.service_account import Credentials
from lightgbm import LGBMRegressor
from datetime import timedelta
import pytz

# ---------- UI ----------
st.set_page_config(page_title="Bakery Forecast - Upload & Run", page_icon="🧁", layout="wide")
st.title("🧁 Bakery Forecast – Upload & Run")

st.markdown(
    "Deze pagina is voor **upload + forecast**. "
    "De resultaten verschijnen in de Sheet-tab **`forecasts`** en zijn te bekijken in de Viewer."
)

# ---------- Secrets & constants ----------
try:
    sa_dict = dict(st.secrets["gcp_service_account"])  # hele service-account JSON
    SHEET_ID = st.secrets["gcp_service_account"]["SHEET_ID"]  # je Google Sheet ID
    # Optioneel: locatie in secrets, anders defaults
    LAT = float(st.secrets["gcp_service_account"].get("LAT", 52.37))   # Amsterdam
    LON = float(st.secrets["gcp_service_account"].get("LON", 4.90))
except Exception:
    st.error("Secrets ontbreken. Voeg gcp_service_account + SHEET_ID toe bij Streamlit → Settings → Secrets.")
    st.stop()

HISTORY_TAB = "history"
FORECASTS_TAB = "forecasts"
NOW_TZ = "Europe/Amsterdam"
HORIZON_DAYS = 7  # tot en met komende zondag (in praktijk: komende 7 dagen)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(sa_dict, scopes=SCOPES)
gc = gspread.authorize(creds)

# ---------- Helpers ----------
def _get_sheet():
    return gc.open_by_key(SHEET_ID)

def _ensure_worksheet(sh, title):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=1000, cols=20)

def _read_history() -> pd.DataFrame:
    sh = _get_sheet()
    ws = _ensure_worksheet(sh, HISTORY_TAB)
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "sku_id", "product_name", "sales", "temp_c", "rain_mm", "derving_qty"])
    # schoonmaak
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # minimal required:
    for col in ["sku_id", "product_name"]:
        if col not in df.columns:
            df[col] = ""
    # numeriek
    for col in ["sales", "temp_c", "rain_mm", "derving_qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df

def _write_sheet_dataframe(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        ws.update([[c for c in df.columns]])  # alleen header
        return
    values = [list(df.columns)] + df.astype(object).astype(str).values.tolist()
    ws.update(values)

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Zorg dat temp/rain bestaan (LightGBM kan met NaN overweg)
    for col in ["temp_c", "rain_mm"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def _get_weather_future(lat: float, lon: float, date_index: pd.DatetimeIndex, tz: str) -> pd.DataFrame:
    start = date_index.min().date().isoformat()
    end   = date_index.max().date().isoformat()
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=auto"
        f"&start_date={start}&end_date={end}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temp_c": daily.get("temperature_2m_mean", []),
        "rain_mm": daily.get("precipitation_sum", []),
    })
    # normaliseer naar lokale midnights
    df["date"] = df["date"].dt.tz_localize(None)
    return df

def _train_and_forecast(history: pd.DataFrame) -> pd.DataFrame:
    """Train per SKU en voorspel 7 dagen vooruit."""
    if history.empty:
        st.error("History is leeg; upload eerst verkoopregels.")
        return pd.DataFrame()

    # kalenderfeatures in history
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"])
    hist = _add_calendar_features(hist)

    # horizon (morgen t/m 7 dagen)
    now_local = pd.Timestamp.now(tz=NOW_TZ).normalize()
    tomorrow = now_local + timedelta(days=1)
    horizon_dates = pd.date_range(start=tomorrow, periods=HORIZON_DAYS, freq="D").tz_localize(None)

    # weer ophalen en op horizon mergen
    weather = _get_weather_future(LAT, LON, horizon_dates, NOW_TZ)
    future = (
        hist[["sku_id", "product_name"]].drop_duplicates()
        .assign(key=1)
        .merge(pd.DataFrame({"date": horizon_dates, "key": 1}), on="key", how="outer")
        .drop(columns="key")
    )
    future = future.merge(weather, on="date", how="left")
    future = _add_calendar_features(future)

    # feature set (zelfde kolommen in train & future)
    FEATS = ["dow", "is_weekend", "temp_c", "rain_mm"]
    # training-waarden: NaN OK voor LGBM
    # (zorg wel dat 'sales' bestaat en numeriek)
    if "sales" not in hist.columns:
        st.error("Kolom 'sales' ontbreekt in history.")
        return pd.DataFrame()
    hist["sales"] = pd.to_numeric(hist["sales"], errors="coerce").fillna(0)

    # train & predict per sku
    out = []
    for sku, g in hist.groupby("sku_id"):
        g = g.sort_values("date")
        # minimale hoeveelheid om te trainen
        if g["sales"].notna().sum() < 5:
            # te weinig data → naive: gemiddelde
            base = future[future["sku_id"] == sku].copy()
            avg = float(g["sales"].mean()) if g["sales"].notna().any() else 0.0
            base["forecast_qty"] = avg
            out.append(base)
            continue

        X = g[FEATS]
        y = g["sales"].astype(float)
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X, y)

        fut = future[future["sku_id"] == sku].copy()
        fut["forecast_qty"] = model.predict(fut[FEATS])
        out.append(fut)

    forecast_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if forecast_df.empty:
        return forecast_df

    # post-process: geen negatieven, afronden
    forecast_df["forecast_qty"] = (
        pd.to_numeric(forecast_df["forecast_qty"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    # run_ts
    forecast_df["run_ts"] = pd.Timestamp.now(tz=NOW_TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    # netjes ordenen
    forecast_df = forecast_df[["sku_id", "product_name", "date", "forecast_qty", "run_ts"]].sort_values(["date", "sku_id"])
    return forecast_df

# ---------- Upload sectie ----------
st.subheader("1) Upload nieuwe verkoopregels (CSV)")

st.markdown(
    "CSV met minimaal kolommen: **date, sku_id, product_name, sales** "
    "(optioneel: temp_c, rain_mm, derving_qty)."
)

uploaded = st.file_uploader("Kies CSV", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    try:
        up_df = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        up_df = pd.read_csv(uploaded, encoding="latin-1")
    up_df.columns = [c.strip() for c in up_df.columns]
    # basistypes
    if "date" not in up_df.columns:
        st.error("CSV mist kolom 'date'.")
        st.stop()
    up_df["date"] = pd.to_datetime(up_df["date"], errors="coerce")
    for col in ["sales", "temp_c", "rain_mm", "derving_qty"]:
        if col in up_df.columns:
            up_df[col] = pd.to_numeric(up_df[col], errors="coerce")
    up_df = up_df.dropna(subset=["date"])

    st.write("Voorbeeld van upload:")
    st.dataframe(up_df.head(10), use_container_width=True)

    if st.button("➡️ Append naar history (dedupe op (date, sku_id))"):
        history = _read_history()
        merged = pd.concat([history, up_df], ignore_index=True)
        # dedupe: laatste wint
        merged = merged.sort_values("date")
        merged = merged.drop_duplicates(subset=["date", "sku_id"], keep="last")
        merged = merged.sort_values(["sku_id", "date"]).reset_index(drop=True)

        # schrijf terug
        sh = _get_sheet()
        ws = _ensure_worksheet(sh, HISTORY_TAB)
        _write_sheet_dataframe(ws, merged)
        st.success(f"History bijgewerkt: {len(merged)} rijen totaal.")

# ---------- Forecast sectie ----------
st.subheader("2) Run forecast voor komende 7 dagen")

if st.button("🔮 Run forecast"):
    history = _read_history()
    if history.empty:
        st.error("History is leeg. Upload eerst data.")
        st.stop()

    with st.spinner("Modellen trainen en voorspellen…"):
        forecast_df = _train_and_forecast(history)

    if forecast_df.empty:
        st.error("Geen forecast gegenereerd.")
        st.stop()

    st.success(f"Forecast klaar: {forecast_df['date'].nunique()} dagen, {forecast_df.shape[0]} rijen.")

    # Schrijf naar forecasts-tab (overschrijf)
    sh = _get_sheet()
    ws_f = _ensure_worksheet(sh, FORECASTS_TAB)
    _write_sheet_dataframe(ws_f, forecast_df)

    # Laat voorbeeld zien
    st.dataframe(forecast_df.head(20), use_container_width=True)

    # Kleine sanity checks
    negatives = (forecast_df["forecast_qty"] < 0).sum()
    st.info(
        f"Negatieve aantallen: {negatives} • "
        f"Datumrange: {forecast_df['date'].min().date()} → {forecast_df['date'].max().date()} • "
        f"Laatst run_ts: {forecast_df['run_ts'].max()}"
    )

st.caption("Tip: De **Viewer**-pagina toont dezelfde forecasts voor medewerkers (read-only).")
