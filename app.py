# app.py — Bakery Forecast (Upload & Run)
# Streamlit app die:
# 1) nieuwe verkoopregels (CSV) toevoegt aan Google Sheet tab 'history' (met dedupe),
# 2) per SKU een lichte LightGBM-forecast draait voor de komende 7 dagen,
# 3) resultaten naar tab 'forecasts' schrijft.

import io
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

import gspread
from google.oauth2 import service_account
from lightgbm import LGBMRegressor
from datetime import timedelta

# ────────────────────────────────────────────────────────────────────────────────
# UI header
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🥖", layout="wide")
st.title("🥖 Bakery Forecast – Upload & Run")
st.caption("Deze pagina is voor **upload + forecast**. De resultaten verschijnen in je Sheet-tab "
           f"**{st.secrets.get('TAB_FORECASTS', 'forecasts')}** en zijn te bekijken in de Viewer.")

# ────────────────────────────────────────────────────────────────────────────────
# Secrets & Google auth

REQUIRED_SECRETS = ["GCP_SERVICE_JSON", "SHEET_ID", "TAB_HISTORY", "TAB_FORECASTS", "LAT", "LON"]
missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(f"Secrets ontbreken. Voeg minimaal toe: {', '.join(REQUIRED_SECRETS)} "
             f"(Settings → Secrets).")
    st.stop()

try:
    SHEET_ID = st.secrets["SHEET_ID"]
    HISTORY_TAB = st.secrets["TAB_HISTORY"]
    FORECASTS_TAB = st.secrets["TAB_FORECASTS"]
    LAT = float(str(st.secrets["LAT"]))
    LON = float(str(st.secrets["LON"]))

    service_info = json.loads(st.secrets["GCP_SERVICE_JSON"])
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    credentials = service_account.Credentials.from_service_account_info(service_info, scopes=SCOPES)
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key(SHEET_ID)
    st.success("✅ Secrets gevonden.")
except Exception as e:
    st.error(f"Authenticatie/Secrets mislukten: {e}")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────────
# Helpers: Sheets

def get_or_create_ws(title: str, rows: int = 1000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)

def ws_to_df(ws, expected_cols: List[str]) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=expected_cols)
    header = values[0] if values else expected_cols
    data = values[1:] if len(values) > 1 else []
    df = pd.DataFrame(data, columns=header)
    # forceer alle expected kolommen
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[expected_cols]
    return df

def df_to_ws(ws, df: pd.DataFrame):
    # zet alles serialiseerbaar (geen NaN/NaT/np types)
    safe = df.copy()
    safe = safe.replace({np.nan: ""})
    # Zorg dat alles Python-native is
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].astype(str)
        elif pd.api.types.is_float_dtype(safe[c]) or pd.api.types.is_integer_dtype(safe[c]):
            # ints als int, floats als float; geen numpy types
            safe[c] = safe[c].apply(lambda x: int(x) if (isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())) else x)
    values = [list(safe.columns)] + safe.astype(object).values.tolist()
    ws.clear()
    ws.update(values, value_input_option="RAW")


# ────────────────────────────────────────────────────────────────────────────────
# Helpers: data normaliseren en features

NOW_TZ = "Europe/Amsterdam"

def normalize_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Zorgt dat df[col] een *kolom* is met string YYYY-MM-DD (geen index, geen tz)."""
    out = df.copy()

    # Als 'date' in de index zit → naar kolom
    if isinstance(out.index, pd.DatetimeIndex) and col not in out.columns:
        out[col] = out.index
        out.reset_index(drop=True, inplace=True)

    # Als kolom bestaat, casten; zo niet en index is datetime → alsnog kolom maken
    if col not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out[col] = out.index

    # Eenduidig casten
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out[col] = out[col].dt.tz_localize(None)
    out[col] = out[col].dt.date.astype(str)

    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = normalize_date_col(out, "date")
    out["dow"] = pd.to_datetime(out["date"]).dt.weekday
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    # per SKU lag/rolling (optioneel; val terug als te weinig historie)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0).astype(float)
    out = out.sort_values(["sku_id", "date"])
    out["lag7"] = out.groupby("sku_id")["sales"].shift(7)
    out["roll7"] = out.groupby("sku_id")["sales"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    out["lag7"] = out["lag7"].fillna(out["roll7"])
    out["roll7"] = out["roll7"].fillna(out.groupby("sku_id")["sales"].transform("mean"))
    return out

def get_weather_range(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum"
        f"&timezone={NOW_TZ}"
        f"&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily:
        return pd.DataFrame(columns=["date", "temp_c", "rain_mm"])
    w = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).dt.tz_localize(None).dt.date.astype(str),
        "temp_c": daily.get("temperature_2m_mean", [np.nan] * len(daily["time"])),
        "rain_mm": daily.get("precipitation_sum", [np.nan] * len(daily["time"])),
    })
    return w

def forecast_week(history: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Train per SKU en voorspel 7 dagen vanaf morgen. Return df[sku_id, product_name, date, forecast_qty, run_ts]."""
    if history.empty:
        raise ValueError("History is leeg.")

    hist = build_features(history)
    # mapping product_name per sku (laatste bekende naam)
    name_map = (history.sort_values("date")
                      .groupby("sku_id")["product_name"]
                      .last()
                      .to_dict())

    # horizon (morgen tm 7 dagen)
    today_local = pd.Timestamp.now(tz=NOW_TZ).normalize()
    tomorrow = today_local + timedelta(days=1)
    horizon_dates = pd.date_range(start=tomorrow, periods=7, freq="D", tz=NOW_TZ)
    horizon_str = horizon_dates.tz_localize(None).date.astype(str).tolist()

    # Weer ophalen
    weather_future = get_weather_range(lat, lon, horizon_str[0], horizon_str[-1])
    weather_future = normalize_date_col(weather_future, "date")

    # features voor toekomst per sku_id
    skus = sorted(hist["sku_id"].unique().tolist())
    rows = []
    for sku in skus:
        past = hist[hist["sku_id"] == sku].copy()
        if past["sales"].count() < 10:
            # te weinig historie → baseline = recente gemiddelde
            baseline = max(0, int(round(past["sales"].tail(7).mean() if len(past) else 0)))
            for d in horizon_str:
                rows.append({"sku_id": sku,
                             "product_name": name_map.get(sku, ""),
                             "date": d,
                             "forecast_qty": baseline})
            continue

        # train set
        X_cols = ["dow", "is_weekend", "temp_c", "rain_mm", "lag7", "roll7"]
        # verrijk past met recente weer? (optioneel – vaak niet nodig); voor eenvoud laten we dat achterwege.

        # model + train
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        # Train op alle rijen waar features niet NaN zijn
        past_train = past.dropna(subset=["lag7", "roll7"]).copy()
        if past_train.empty:
            base = max(0, int(round(past["sales"].tail(7).mean())))
            for d in horizon_str:
                rows.append({"sku_id": sku,
                             "product_name": name_map.get(sku, ""),
                             "date": d,
                             "forecast_qty": base})
            continue

        # simpele join met weer voor “dow/is_weekend/temp/rain” van de target-datum
        # (we trainen op historische target y, features komen uit datum zelf)
        # Voor het verleden hebben we geen weer-tab → val terug op roll7
        past_train = past_train.merge(
            pd.DataFrame({
                "date": past_train["date"].tolist(),
                "temp_c": np.nan,
                "rain_mm": np.nan
            }), on="date", how="left"
        )

        X = past_train[X_cols].fillna(past_train[["lag7", "roll7"]].mean(axis=1))
        y = past_train["sales"].values
        model.fit(X, y)

        # toekomst features
        fut = pd.DataFrame({
            "date": horizon_str,
        })
        fut = normalize_date_col(fut, "date")
        fut["dow"] = pd.to_datetime(fut["date"]).dt.weekday
        fut["is_weekend"] = (fut["dow"] >= 5).astype(int)
        fut = fut.merge(weather_future, on="date", how="left")
        # lags/rolls: gebruik laatste bekende uit verleden
        last_roll = past["sales"].rolling(7, min_periods=3).mean().iloc[-1]
        last_lag = past["sales"].iloc[-7] if len(past) >= 7 else past["sales"].iloc[-1]
        fut["roll7"] = last_roll
        fut["lag7"] = last_lag

        Xf = fut[X_cols].fillna(fut[["lag7", "roll7"]].mean(axis=1))
        pred = model.predict(Xf)
        pred = np.maximum(0, np.rint(pred).astype(int))  # geen negatieven, afronden

        for d, q in zip(fut["date"].tolist(), pred.tolist()):
            rows.append({
                "sku_id": sku,
                "product_name": name_map.get(sku, ""),
                "date": d,
                "forecast_qty": int(q)
            })

    out = pd.DataFrame(rows, columns=["sku_id", "product_name", "date", "forecast_qty"])
    out = normalize_date_col(out, "date")
    out = out.sort_values(["sku_id", "date"]).reset_index(drop=True)
    out["run_ts"] = pd.Timestamp.now(tz=NOW_TZ).isoformat()
    return out


# ────────────────────────────────────────────────────────────────────────────────
# UI – Sectie 1: Uploaden (CSV met alleen nieuwe regels)

st.subheader("1) Nieuwe verkoopregels uploaden (CSV, alleen nieuwe rijen)")
st.caption("CSV met minimaal kolommen: **date, sku_id, product_name, sales** (meer mag; deze 4 verplicht).")

# Zorg dat beide tabs bestaan
ws_hist = get_or_create_ws(HISTORY_TAB)
ws_fore = get_or_create_ws(FORECASTS_TAB)

# Lees history
history_cols = ["date", "sku_id", "product_name", "sales"]
history = ws_to_df(ws_hist, expected_cols=history_cols)
# normaliseer en cast
history = normalize_date_col(history, "date")
history["sku_id"] = history["sku_id"].astype(str)
history["product_name"] = history["product_name"].astype(str)
history["sales"] = pd.to_numeric(history["sales"], errors="coerce").fillna(0).astype(float)

up = st.file_uploader("Drag & drop of kies CSV", type=["csv"])
if up is not None:
    try:
        upload_df = pd.read_csv(up)
        # minimale kolommen
        for c in history_cols:
            if c not in upload_df.columns:
                st.error(f"Upload mist verplichte kolom: **{c}**")
                st.stop()

        upload_df = upload_df[history_cols].copy()
        upload_df = normalize_date_col(upload_df, "date")
        upload_df["sku_id"] = upload_df["sku_id"].astype(str).str.strip()
        upload_df["product_name"] = upload_df["product_name"].astype(str).str.strip()
        upload_df["sales"] = pd.to_numeric(upload_df["sales"], errors="coerce").fillna(0).astype(float)

        # concat + dedupe op (date, sku_id) -> laatste wint
        before = len(history)
        combined = pd.concat([history, upload_df], ignore_index=True)
        combined = combined.sort_values(["date"]).drop_duplicates(subset=["date", "sku_id"], keep="last")
        added = len(combined) - before

        # terugschrijven naar Sheet
        df_to_ws(ws_hist, combined[history_cols])

        st.success(f"✅ {added} rijen toegevoegd (met dedupe).")
        st.dataframe(upload_df.tail(10), use_container_width=True)
        history = combined  # update in-memory
    except Exception as e:
        st.error(f"Upload mislukt: {e}")

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# UI – Sectie 2: Forecast draaien

st.subheader("2) Forecast draaien")
run_btn = st.button("Run forecast voor komende 7 dagen")

if run_btn:
    try:
        if history.empty:
            st.error("Forecast mislukt: **history is leeg**.")
            st.stop()
        with st.spinner("Voorspellen…"):
            fc = forecast_week(history, LAT, LON)
            # netjes wegschrijven
            out_cols = ["sku_id", "product_name", "date", "forecast_qty", "run_ts"]
            df_to_ws(ws_fore, fc[out_cols])
        st.success("✅ Forecast geschreven naar Sheet-tab **forecasts**.")
        st.dataframe(fc.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Forecast mislukt: {e}")

st.caption("Tip: open je Google Sheet om resultaten te bekijken/downloaden. Viewer-app kan tab **forecasts** alleen-lezen tonen.")
