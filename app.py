# app.py — Upload & Forecast (voor baas)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io
import time
from datetime import datetime, timedelta
import pytz

import gspread
from google.oauth2.service_account import Credentials

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


# ---------- Page config ----------
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🥖", layout="wide")
st.title("🥖 Bakery Forecast – Upload & Run")
st.write("Deze pagina is voor **upload + forecast**. De resultaten verschijnen in je Sheet-tab "
         "**forecasts** en zijn te bekijken in de Viewer.")


# ---------- Helpers: safe secrets (met fallback formulier) ----------
def load_settings_from_secrets_or_form():
    """
    Probeert eerst Streamlit Secrets te lezen. Als dat niet lukt of onvolledig is,
    laat een formulier zien waarin je alles tijdelijk kunt invullen.
    Retourneert dict: SHEET_ID, TAB_HISTORY, TAB_FORECASTS, LAT, LON, SA (service-account dict)
    """
    def _has_all(s):
        need = {"SHEET_ID", "TAB_HISTORY", "TAB_FORECASTS", "LAT", "LON"}
        has_needed = all(k in s for k in need)
        has_json = ("GCP_SERVICE_JSON" in s) or ("gcp_service_account" in s)
        return has_needed and has_json

    # 1) probeer secrets
    try:
        if _has_all(st.secrets):
            sheet_id = st.secrets["SHEET_ID"]
            tab_hist = st.secrets["TAB_HISTORY"]
            tab_fc = st.secrets["TAB_FORECASTS"]
            lat = float(str(st.secrets["LAT"]))
            lon = float(str(st.secrets["LON"]))
            raw = st.secrets.get("GCP_SERVICE_JSON", st.secrets.get("gcp_service_account"))
            sa = raw if isinstance(raw, dict) else json.loads(str(raw).strip().lstrip("\ufeff"))
            st.caption("🔐 Secrets gevonden.")
            return dict(SHEET_ID=sheet_id, TAB_HISTORY=tab_hist, TAB_FORECASTS=tab_fc,
                        LAT=lat, LON=lon, SA=sa)
        else:
            st.info("Secrets onvolledig of ontbreken. Vul tijdelijk hieronder in.")
    except Exception as e:
        st.info(f"Secrets konden niet gebruikt worden ({e}). Vul tijdelijk hieronder in.")

    # 2) fallback formulier (wordt niet opgeslagen)
    with st.form("cfg"):
        sheet_id = st.text_input("SHEET_ID (Google Sheet ID)")
        tab_hist = st.text_input("TAB_HISTORY", "history")
        tab_fc = st.text_input("TAB_FORECASTS", "forecasts")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.text_input("LAT", "51.44")
        with c2:
            lon = st.text_input("LON", "5.48")
        sa_txt = st.text_area("Service Account JSON (volledige JSON, exact)", height=220,
                              placeholder='{"type":"service_account", ...}')
        go = st.form_submit_button("Doorgaan")
    if not go:
        st.stop()
    try:
        sa = json.loads(sa_txt.strip().lstrip("\ufeff"))
    except Exception as e:
        st.error(f"Service account JSON ongeldig: {e}")
        st.stop()
    return dict(SHEET_ID=sheet_id.strip(), TAB_HISTORY=tab_hist.strip(), TAB_FORECASTS=tab_fc.strip(),
                LAT=float(lat), LON=float(lon), SA=sa)


CFG = load_settings_from_secrets_or_form()

# ---------- Google Sheets client ----------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(CFG["SA"], scopes=SCOPES)
gc = gspread.authorize(credentials)

def ws(sheet_id, tab_name):
    return gc.open_by_key(sheet_id).worksheet(tab_name)

# ---------- Data I/O ----------
def read_history():
    sh = ws(CFG["SHEET_ID"], CFG["TAB_HISTORY"])
    df = pd.DataFrame(sh.get_all_records())
    if df.empty:
        return pd.DataFrame(columns=["date", "sku_id", "product_name", "sales"])
    # normaliseer
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df

def write_forecasts(forecast_df):
    sh = ws(CFG["SHEET_ID"], CFG["TAB_FORECASTS"])
    # leeg en schrijf opnieuw
    sh.clear()
    sh.update([forecast_df.columns.tolist()] + forecast_df.astype(str).values.tolist())

def append_new_history(rows_df):
    """Voegt nieuwe rijen toe aan history met dedupe (date, sku_id). Laatste win."""
    if rows_df.empty:
        return
    hist = read_history()
    combined = pd.concat([hist, rows_df], ignore_index=True)
    combined = (combined
        .sort_values(["sku_id", "date"])
        .drop_duplicates(subset=["date", "sku_id"], keep="last"))
    # overschrijf heel tab
    sh = ws(CFG["SHEET_ID"], CFG["TAB_HISTORY"])
    sh.clear()
    # zorg voor kolomvolgorde
    cols = ["date", "sku_id", "product_name", "sales"]
    combined = combined[cols]
    sh.update([combined.columns.tolist()] + combined.astype(str).values.tolist())


# ---------- Feature engineering ----------
def make_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # rolling per sku
    df = df.sort_values(["sku_id", "date"])
    df["sales_lag1"] = df.groupby("sku_id")["sales"].shift(1)
    df["sales_ma7"] = df.groupby("sku_id")["sales"].shift(1).rolling(7).mean()
    # vul missings
    for c in ["sales_lag1", "sales_ma7"]:
        df[c] = df[c].fillna(df[c].median())
    # weer kolommen kunnen bestaan in history (optioneel)
    for c in ["temp_c", "rain_mm"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ---------- Weer (Open-Meteo) ----------
def get_weather(lat, lon, dates, tz="Europe/Amsterdam"):
    base = "https://api.open-meteo.com/v1/forecast"
    params = dict(
        latitude=lat, longitude=lon,
        daily=["temperature_2m_mean","precipitation_sum"],
        timezone=tz,
        start_date=dates.min().date().isoformat(),
        end_date=dates.max().date().isoformat(),
    )
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j["daily"]
    w = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_c": daily["temperature_2m_mean"],
        "rain_mm": daily["precipitation_sum"],
    })
    return w


# ---------- Train & forecast ----------
def train_one_sku(hist_feat):
    X = hist_feat[["dow","is_weekend","sales_lag1","sales_ma7","temp_c","rain_mm"]]
    y = hist_feat["sales"]
    # simpele, stabiele instellingen
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X, y)
    return model

def forecast_week(history_df, lat, lon, tz="Europe/Amsterdam"):
    ams = pytz.timezone(tz)
    today_local = pd.Timestamp.now(tz=ams).normalize()
    tomorrow = today_local + pd.Timedelta(days=1)
    horizon = pd.date_range(start=tomorrow, periods=7, freq="D")

    # basisfeatures op historie
    hist = history_df.copy()
    hist["sales"] = hist["sales"].astype(float).clip(lower=0)
    hist_feat = make_features(hist)

    # weer voor horizon
    weather = get_weather(lat, lon, horizon, tz=tz)

    # per sku trainen & voorspellen
    rows = []
    for sku, g in hist_feat.groupby("sku_id"):
        # laatste bekende staat voor deze sku
        last_row = g.sort_values("date").tail(1).iloc[0]
        model = train_one_sku(g)

        # maak future frame
        f = pd.DataFrame({"date": horizon})
        f["sku_id"] = sku
        f["product_name"] = g["product_name"].iloc[-1]
        f["dow"] = f["date"].dt.dayofweek
        f["is_weekend"] = (f["dow"] >= 5).astype(int)

        # lags/rolling approx: gebruik laatste bekende waarden als start
        f["sales_lag1"] = last_row["sales"]
        f["sales_ma7"]  = g["sales"].tail(7).mean()

        # join weer
        f = f.merge(weather, on="date", how="left")
        # safety in case weather missing
        f["temp_c"] = f["temp_c"].fillna(f["temp_c"].median())
        f["rain_mm"] = f["rain_mm"].fillna(0)

        Xf = f[["dow","is_weekend","sales_lag1","sales_ma7","temp_c","rain_mm"]]
        yhat = model.predict(Xf)
        f["forecast_qty"] = np.round(np.clip(yhat, 0, None)).astype(int)
        rows.append(f[["sku_id","product_name","date","forecast_qty"]])

    out = pd.concat(rows, ignore_index=True)
    out["run_ts"] = pd.Timestamp.now(tz=pytz.timezone(tz))
    return out


# ---------- UI: upload + run ----------
st.subheader("1) Nieuwe verkoopregels uploaden (CSV, alleen nieuwe rijen)")
up = st.file_uploader("CSV met kolommen: date, sku_id, product_name, sales", type=["csv"])
if up is not None:
    try:
        new_df = pd.read_csv(up)
        # schoonmaken
        new_df.columns = [c.strip() for c in new_df.columns]
        need = {"date","sku_id","product_name","sales"}
        if not need.issubset(set(new_df.columns)):
            st.error(f"CSV mist kolommen: {sorted(list(need - set(new_df.columns)))}")
            st.stop()
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df["sales"] = pd.to_numeric(new_df["sales"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        append_new_history(new_df)
        st.success(f"{len(new_df)} rijen toegevoegd (met dedupe).")
    except Exception as e:
        st.error(f"Upload mislukt: {e}")
        st.stop()

st.subheader("2) Forecast draaien")
if st.button("Run forecast voor komende 7 dagen"):
    with st.spinner("Bezig met trainen en voorspellen…"):
        hist = read_history()
        if hist.empty:
            st.error("History is leeg. Upload eerst data.")
            st.stop()
        # simpele validatie
        if hist["sales"].dtype != int and hist["sales"].dtype != float:
            hist["sales"] = pd.to_numeric(hist["sales"], errors="coerce").fillna(0)

        out = forecast_week(hist, CFG["LAT"], CFG["LON"], tz="Europe/Amsterdam")
        # optioneel: mini-validatie
        st.caption("✅ Geen negatieve aantallen, afronding op hele stuks.")
        write_forecasts(out)
        st.success("Voorspellingen geschreven naar tab **forecasts** in je Sheet.")
        st.dataframe(out.head(20), use_container_width=True)

st.caption("Tip: gebruik de **Viewer**-app om de resultaten (alleen-lezen) te bekijken of te downloaden.")
