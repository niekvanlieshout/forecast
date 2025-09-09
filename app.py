# app.py  — Upload & beheer (skelet)
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import pytz
from datetime import datetime

st.set_page_config(page_title="Bakery Forecast - Upload", page_icon="🍞", layout="centered")
st.title("🍞 Bakery Forecast – Upload (skelet)")

# --- Secrets lezen ---
# Zet straks in Streamlit Cloud bij Secrets:
#   [gcp_service_account]
#   type = "service_account"
#   project_id = "..."
#   private_key_id = "..."
#   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
#   client_email = "robot@project.iam.gserviceaccount.com"
#   client_id = "..."
#   ...
#   SHEET_ID = "je_google_sheet_id"
try:
    sa_dict = dict(st.secrets["gcp_service_account"])
    SHEET_ID = st.secrets["gcp_service_account"]["SHEET_ID"]
except Exception as e:
    st.error("Secrets ontbreken. Voeg je service-account JSON + SHEET_ID toe bij Streamlit → Settings → Secrets.")
    st.stop()

# --- GSpread client ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_info(sa_dict, scopes=SCOPES)
gc = gspread.authorize(creds)

# --- Sheet & tabs ophalen (aanmaken indien niet bestaan) ---
def open_sheet(sheet_id: str):
    sh = gc.open_by_key(sheet_id)
    wsnames = [w.title for w in sh.worksheets()]
    if "history" not in wsnames:
        sh.add_worksheet(title="history", rows=1000, cols=10)
    if "forecasts" not in wsnames:
        sh.add_worksheet(title="forecasts", rows=1000, cols=10)
    return sh

try:
    sh = open_sheet(SHEET_ID)
    st.success("Verbonden met Google Sheet ✅")
except Exception as e:
    st.exception(e)
    st.stop()

ws_history = sh.worksheet("history")
ws_forecasts = sh.worksheet("forecasts")

# --- Helper: sheet -> DataFrame ---
def ws_to_df(ws):
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def df_to_ws(ws, df: pd.DataFrame):
    ws.clear()
    if df.empty:
        return
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())

# --- Laat huidige status zien ---
with st.expander("Huidige status"):
    hist_df = ws_to_df(ws_history)
    st.write("history shape:", hist_df.shape)
    st.dataframe(hist_df.head(10), use_container_width=True)
    fc_df = ws_to_df(ws_forecasts)
    st.write("forecasts shape:", fc_df.shape)
    st.dataframe(fc_df.head(10), use_container_width=True)

st.divider()
st.subheader("Nieuwe verkoopdata uploaden")

uploaded = st.file_uploader(
    "Upload CSV of XLSX met nieuwe rijen (kolommen: date, sku_id, product_name, sales; optioneel temp_c, rain_mm).",
    type=["csv", "xlsx"]
)

def read_uploaded(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    # kolommen netjes maken
    df.columns = [c.strip() for c in df.columns]
    # verplichte kolommen check
    needed = {"date","sku_id","product_name","sales"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Ontbrekende kolommen: {missing}")
        return pd.DataFrame()
    # types
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for c in ["sales","temp_c","rain_mm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sales"] = df["sales"].fillna(0).clip(lower=0).round().astype(int)
    return df

new_rows = read_uploaded(uploaded)
if not new_rows.empty:
    st.info(f"Voorbeeld nieuwe rijen ({new_rows.shape[0]}):")
    st.dataframe(new_rows.head(10), use_container_width=True)

if st.button("Alleen uploaden naar history (dedupe op date+sku_id)"):
    if new_rows.empty:
        st.warning("Upload eerst een bestand.")
        st.stop()
    # huidige history + samenvoegen
    hist_df = ws_to_df(ws_history)
    combined = pd.concat([hist_df, new_rows], ignore_index=True)
    # normaliseer sleutel
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    # dedupe: laatste upload wint
    combined.sort_values(["date","sku_id"], inplace=True)
    combined = combined.drop_duplicates(subset=["date","sku_id"], keep="last")
    # kolomvolgorde vriendelijk
    base_cols = ["date","sku_id","product_name","sales"]
    other = [c for c in combined.columns if c not in base_cols]
    combined = combined[base_cols + other]
    # wegschrijven
    try:
        df_to_ws(ws_history, combined)
        st.success(f"History bijgewerkt: {combined.shape[0]} rijen. ✅")
    except Exception as e:
        st.exception(e)

st.divider()
st.subheader("Voorspellen (komt in de volgende stap)")

st.write("In de **volgende stap** voegen we de forecast-knop toe die:")
st.markdown("""
- Open-Meteo weer voor komende 7 dagen ophaalt,  
- per `sku_id` snel een LightGBM model traint,  
- en de uitkomst naar de tab **`forecasts`** schrijft.
""")
