# app.py — Bakery Forecast (Upload & Run)
# (c) jouw team — eenvoudig, robuust, gratis stack

import io
import json
import time
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from typing import Dict, Tuple

# ============ UI basis ============
st.set_page_config(page_title="Bakery Forecast – Upload & Run", page_icon="🥖", layout="wide")
st.title("🥖 Bakery Forecast – Upload & Run")
st.caption("Deze pagina is voor **upload + forecast**. De resultaten verschijnen in je Sheet-tab "
           "**forecasts** en zijn te bekijken in de Viewer.")

# ============ Helpers: config & authenticatie ============

def read_config_from_secrets() -> Dict:
    """Lees config uit st.secrets (als aanwezig)."""
    try:
        secrets = st.secrets  # type: ignore[attr-defined]
        cfg = {
            "SHEET_ID": secrets["SHEET_ID"],
            "TAB_HISTORY": secrets.get("TAB_HISTORY", "history"),
            "TAB_FORECASTS": secrets.get("TAB_FORECASTS", "forecasts"),
            "LAT": float(secrets.get("LAT", 52.37)),
            "LON": float(secrets.get("LON", 4.90)),
            "GCP_SERVICE_JSON": secrets["GCP_SERVICE_JSON"],
        }
        # indien dict → naar json string
        if isinstance(cfg["GCP_SERVICE_JSON"], dict):
            cfg["GCP_SERVICE_JSON"] = json.dumps(cfg["GCP_SERVICE_JSON"])
        return cfg
    except Exception:
        return {}

def config_form(defaults: Dict) -> Dict:
    """Toon formulier als secrets ontbreken. Retourneert config-dict."""
    with st.expander("⚙️ Instellingen (alleen nodig als er geen Streamlit Secrets zijn)", expanded=not bool(defaults)):
        c1, c2, c3, c4 = st.columns(4)
        sheet_id = c1.text_input("SHEET_ID (Google Sheet ID)", value=defaults.get("SHEET_ID", ""))
        tab_history = c2.text_input("TAB_HISTORY", value=defaults.get("TAB_HISTORY", "history"))
        tab_forecasts = c3.text_input("TAB_FORECASTS", value=defaults.get("TAB_FORECASTS", "forecasts"))
        lat = c4.text_input("LAT", value=str(defaults.get("LAT", "52.37")))
        lon = c4.text_input("LON", value=str(defaults.get("LON", "4.90")))
        gcp_json = st.text_area("Service Account JSON (volledige JSON, exact)", height=180,
                                value=defaults.get("GCP_SERVICE_JSON", ""))

        ok = st.button("Doorgaan")
        if ok:
            st.session_state["_cfg"] = {
                "SHEET_ID": sheet_id.strip(),
                "TAB_HISTORY": tab_history.strip() or "history",
                "TAB_FORECASTS": tab_forecasts.strip() or "forecasts",
                "LAT": float(str(lat).replace(",", ".")),
                "LON": float(str(lon).replace(",", ".")),
                "GCP_SERVICE_JSON": gcp_json.strip(),
            }

    return st.session_state.get("_cfg", defaults)

def get_config() -> Dict:
    secrets_cfg = read_config_from_secrets()
    if secrets_cfg:
        st.success("🔐 Secrets gevonden.")
        return secrets_cfg
    cfg = config_form({})
    # minimale checks
    missing = [k for k in ["SHEET_ID", "GCP_SERVICE_JSON"] if not cfg.get(k)]
    if missing:
        st.error("❌ Secrets/instellingen ontbreken. Vul minimaal **GCP_SERVICE_JSON** en **SHEET_ID** in "
                 "(via Settings → Secrets of hierboven).")
        st.stop()
    return cfg

CFG = get_config()

# ============ Google Sheets client ============

def make_gspread_client(cfg: Dict) -> gspread.Client:
    try:
        info = cfg["GCP_SERVICE_JSON"]
        if isinstance(info, str):
            service_info = json.loads(info)
        else:
            service_info = info
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_info(service_info, scopes=scopes)
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        st.error(f"❌ Fout in authenticatie naar Google: {e}")
        st.stop()

GC = make_gspread_client(CFG)

def open_ws(sheet_id: str, tab_name: str) -> gspread.Worksheet:
    sh = GC.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
    return ws

def read_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    ws = open_ws(sheet_id, tab_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values[1:], columns=values[0])
    # basic clean
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df

def write_sheet(sheet_id: str, tab_name: str, df: pd.DataFrame):
    """Schrijf DataFrame veilig naar Sheets (datum → string, NaN → '')."""
    ws = open_ws(sheet_id, tab_name)
    ws.clear()
    if df is None or df.empty:
        return
    out = df.copy()

    # Standaardiseer datumkolommen vóór schrijven
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "run_ts" in out.columns:
        out["run_ts"] = out["run_ts"].astype(str)
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    values = [list(out.columns)] + out.astype(object).where(pd.notnull(out), "").values.tolist()
    ws.update(values)

# ============ Data validatie & upload ============

REQUIRED_COLS = ["date", "sku_id", "product_name", "sales"]

def validate_history(df: pd.DataFrame) -> Tuple[bool, str]:
    cols_missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if cols_missing:
        return False, f"Ontbrekende kolommen: {cols_missing}"
    # types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if df["date"].isna().any():
        return False, "Sommige datums zijn ongeldig."
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    # nette regels
    df["sales"] = df["sales"].clip(lower=0).round().astype(int)
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    return True, "OK"

def append_new_sales(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([existing, new_rows], ignore_index=True)
    # dedupe op (date, sku_id) → laatste wint
    all_df = (all_df
              .sort_values(["date"])  # laatste onderaan
              .drop_duplicates(subset=["date", "sku_id"], keep="last")
              .sort_values(["date", "sku_id"])
              .reset_index(drop=True))
    return all_df

# ============ Weer & features & model ============

def get_weather_df(lat: float, lon: float, days: int = 7, tz: str = "Europe/Amsterdam") -> pd.DataFrame:
    # horizon: morgen t/m 7 dagen
    today = pd.Timestamp.now(tz=tz).normalize()
    start = (today + pd.Timedelta(days=1)).date()
    end = (today + pd.Timedelta(days=days)).date()

    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_mean,precipitation_sum"
           f"&timezone={tz}&start_date={start}&end_date={end}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j["daily"]
    w = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).dt.date,
        "temp_c": daily["temperature_2m_mean"],
        "rain_mm": daily["precipitation_sum"],
    })
    return w

def build_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["dow"] = pd.to_datetime(df["date"]).dt.weekday  # 0=ma
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Voor eenvoudige per-SKU features maken we per sku_id lags met groupby
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    df["sales_lag7"] = df.groupby("sku_id")["sales"].shift(7)
    df["sales_ma7"] = (df.groupby("sku_id")["sales"]
                         .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
    df["sales_lag7"] = df["sales_lag7"].fillna(df["sales"].median())
    df["sales_ma7"] = df["sales_ma7"].fillna(df["sales"].median())
    return df

def train_per_sku(df_feat: pd.DataFrame) -> Dict[str, lgb.Booster]:
    models = {}
    feature_cols = ["dow", "is_weekend", "temp_c", "rain_mm", "sales_lag7", "sales_ma7"]
    for sku, sdf in df_feat.groupby("sku_id"):
        sdf = sdf.dropna(subset=["sales"])
        if len(sdf) < 14:  # te weinig historie? overslaan
            continue
        X = sdf[feature_cols]
        y = sdf["sales"].astype(int)
        params = dict(
            objective="regression",
            metric="mae",
            learning_rate=0.1,
            num_leaves=31,
            min_data_in_leaf=10,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            verbose=-1,
        )
        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(params, dtrain, num_boost_round=150)
        models[sku] = model
    return models

def forecast_week(hist: pd.DataFrame, lat: float, lon: float, tz: str = "Europe/Amsterdam") -> pd.DataFrame:
    # Features uit historie
    hist_ok = build_features(hist)

    # Weer voor komende 7 dagen
    weather = get_weather_df(lat, lon, days=7, tz=tz)  # date (date), temp_c, rain_mm

    # Bouw toekomst-frame: cartesiaans product sku x forecast-dates
    skus = hist[["sku_id", "product_name"]].drop_duplicates()
    future = skus.assign(key=1).merge(
        pd.DataFrame({"date": weather["date"], "key": 1}), on="key", how="left"
    ).drop(columns="key")

    # Voeg kalenderfeatures toe
    future["dow"] = pd.to_datetime(future["date"]).dt.weekday
    future["is_weekend"] = (future["dow"] >= 5).astype(int)

    # Voeg weer toe (merge op pure date)
    future = future.merge(weather, on="date", how="left")

    # Voeg sales_lag7 en sales_ma7 uit historie toe (laatste bekende waarden per sku)
    last_feat = (hist_ok
                 .sort_values("date")
                 .groupby("sku_id")[["sales_lag7", "sales_ma7"]]
                 .tail(1)
                 .reset_index())
    future = future.merge(last_feat, on="sku_id", how="left")
    future["sales_lag7"] = future["sales_lag7"].fillna(hist["sales"].median())
    future["sales_ma7"] = future["sales_ma7"].fillna(hist["sales"].median())

    # Train per-SKU
    models = train_per_sku(hist_ok)

    # Voorspel
    feature_cols = ["dow", "is_weekend", "temp_c", "rain_mm", "sales_lag7", "sales_ma7"]
    preds = []
    for sku, g in future.groupby("sku_id"):
        Xf = g[feature_cols]
        if sku in models:
            yhat = models[sku].predict(Xf)
        else:
            # fallback: recente gemiddelde sales van deze sku
            base = (hist.loc[hist["sku_id"] == sku, "sales"].tail(14).mean()
                    if (hist["sku_id"] == sku).any() else hist["sales"].tail(14).mean())
            yhat = np.full(len(g), base)
        out = g[["sku_id", "product_name", "date"]].copy()
        out["forecast_qty"] = np.maximum(0, np.round(yhat)).astype(int)
        preds.append(out)

    forecast_df = pd.concat(preds, ignore_index=True)
    forecast_df["run_ts"] = pd.Timestamp.now(tz=tz)
    return forecast_df[["sku_id", "product_name", "date", "forecast_qty", "run_ts"]]

# ============ UI — Stap 1: upload nieuwe regels ============

st.subheader("1) Nieuwe verkoopregels uploaden (CSV, alleen nieuwe rijen)")
st.caption("CSV met kolommen: **date, sku_id, product_name, sales** (meer mag, maar deze zijn verplicht).")

uploaded = st.file_uploader("Drag and drop file here", type=["csv"], label_visibility="collapsed")

if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        ok, msg = validate_history(new_df)
        if not ok:
            st.error(f"❌ Upload mislukt: {msg}")
        else:
            # huidige history lezen
            hist = read_sheet(CFG["SHEET_ID"], CFG["TAB_HISTORY"])
            # als leeg, maak juiste kolommen
            if hist.empty:
                hist = pd.DataFrame(columns=REQUIRED_COLS)
            merged = append_new_sales(hist, new_df)
            write_sheet(CFG["SHEET_ID"], CFG["TAB_HISTORY"], merged)
            delta = len(merged) - len(hist)
            st.success(f"✅ {max(delta,0)} rijen toegevoegd (met dedupe).")
    except Exception as e:
        st.error(f"❌ Upload mislukt: {e}")

# Laat laatste 10 regels uit history zien
hist_preview = read_sheet(CFG["SHEET_ID"], CFG["TAB_HISTORY"])
st.dataframe(hist_preview.tail(10), use_container_width=True)

# ============ UI — Stap 2: forecast draaien ============

st.subheader("2) Forecast draaien")
if st.button("Run forecast voor komende 7 dagen"):
    try:
        hist = read_sheet(CFG["SHEET_ID"], CFG["TAB_HISTORY"])
        if hist.empty:
            st.error("❌ Forecast mislukt: **History is leeg**.")
            st.stop()

        fc = forecast_week(hist, CFG["LAT"], CFG["LON"], tz="Europe/Amsterdam")
        write_sheet(CFG["SHEET_ID"], CFG["TAB_FORECASTS"], fc)
        st.success(f"✅ Forecast klaar: {len(fc)} rijen naar tab **{CFG['TAB_FORECASTS']}** geschreven.")
        st.dataframe(fc.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Forecast mislukt: {e}")

st.caption("Tip: open je Google Sheet om resultaten te bekijken/downloaderen. "
           "Viewer-app kan deze tab **forecasts** alleen-lezen tonen.")
