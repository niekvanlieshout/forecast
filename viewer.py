# viewer.py — Read-only viewer voor medewerkers (laatste forecast)
import json
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Bakery Forecast – Viewer", page_icon="📈", layout="wide")
st.title("📈 Bakery Forecast – Viewer (laatste 7 dagen vooruit)")

# ==== Secrets check ====
REQ = ["GCP_SERVICE_JSON", "SHEET_ID", "TAB_FORECASTS"]
miss = [k for k in REQ if k not in st.secrets]
if miss:
    st.error("Secrets ontbreken: " + ", ".join(miss) + " (beheerder: vul in bij Settings → Secrets).")
    st.stop()

SERVICE_INFO  = json.loads(st.secrets["GCP_SERVICE_JSON"])
SHEET_ID      = st.secrets["SHEET_ID"]
FORECASTS_TAB = st.secrets["TAB_FORECASTS"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
creds = Credentials.from_service_account_info(SERVICE_INFO, scopes=SCOPES)
gc = gspread.authorize(creds)

# ==== Data inlezen ====
def read_forecasts():
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(FORECASTS_TAB)
    except gspread.WorksheetNotFound:
        return pd.DataFrame(columns=["sku_id","product_name","date","forecast_qty","run_ts"])
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame(columns=["sku_id","product_name","date","forecast_qty","run_ts"])
    cols = [c.strip() for c in rows[0]]
    df = pd.DataFrame(rows[1:], columns=cols)
    # types
    if "forecast_qty" in df.columns:
        df["forecast_qty"] = pd.to_numeric(df["forecast_qty"], errors="coerce").fillna(0).astype(int)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    if "run_ts" in df.columns:
        # run_ts is string met offset, gewoon tonen
        pass
    return df

df = read_forecasts()

if df.empty:
    st.info("Nog geen forecasts beschikbaar. Vraag je beheerder de upload/run-app te draaien.")
    st.stop()

# ==== Filters ====
left, right = st.columns([2,1])
with left:
    unique_sku = ["(alle)"] + sorted(df["sku_id"].dropna().unique().tolist())
    sku_sel = st.selectbox("Filter op artikel (SKU)", unique_sku, index=0)
with right:
    dl = st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="forecasts_latest.csv",
        mime="text/csv"
    )

if sku_sel != "(alle)":
    df = df[df["sku_id"] == sku_sel]

# ==== Info over run ====
run_info = df["run_ts"].iloc[0] if "run_ts" in df.columns and len(df) else "onbekend"
st.caption(f"Laatste run: **{run_info}** (alleen komende 7 dagen).")

# ==== Tabel + simpele pivot ====
st.dataframe(
    df.sort_values(["date","sku_id"]),
    use_container_width=True,
    hide_index=True
)

# Kleine dag-samenvatting (optioneel)
if {"date","forecast_qty"}.issubset(df.columns):
    day_sum = df.groupby("date", as_index=False)["forecast_qty"].sum()
    st.bar_chart(day_sum.set_index("date")["forecast_qty"])
