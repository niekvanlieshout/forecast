# viewer.py — Alleen-lezen viewer
import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Bakery Forecast – Viewer", page_icon="📈", layout="wide")
st.title("📈 Bakery Forecast – Viewer")
st.write("Read-only weergave van de laatste voorspellingen uit je Google Sheet.")

# ---- Secrets of fallback mini-form ----
def load_cfg():
    def ok(s): 
        need = {"SHEET_ID","TAB_FORECASTS"}
        has_json = ("GCP_SERVICE_JSON" in s) or ("gcp_service_account" in s)
        return all(k in s for k in need) and has_json

    if ok(st.secrets):
        sheet_id = st.secrets["SHEET_ID"]
        tab_fc = st.secrets["TAB_FORECASTS"]
        raw = st.secrets.get("GCP_SERVICE_JSON", st.secrets.get("gcp_service_account"))
        sa = raw if isinstance(raw, dict) else json.loads(str(raw).strip().lstrip("\ufeff"))
        return sheet_id, tab_fc, sa

    st.info("Secrets ontbreken. Vul tijdelijk hieronder in.")
    sheet_id = st.text_input("SHEET_ID")
    tab_fc = st.text_input("TAB_FORECASTS", "forecasts")
    sa_txt = st.text_area("Service Account JSON", height=220)
    if not st.button("Laden"):
        st.stop()
    sa = json.loads(sa_txt.strip().lstrip("\ufeff"))
    return sheet_id, tab_fc, sa

SHEET_ID, TAB_FC, SA = load_cfg()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(SA, scopes=SCOPES)
gc = gspread.authorize(credentials)

def read_forecasts():
    sh = gc.open_by_key(SHEET_ID).worksheet(TAB_FC)
    df = pd.DataFrame(sh.get_all_records())
    if df.empty:
        return pd.DataFrame(columns=["sku_id","product_name","date","forecast_qty","run_ts"])
    df["date"] = pd.to_datetime(df["date"])
    df["run_ts"] = pd.to_datetime(df["run_ts"], errors="coerce")
    df["forecast_qty"] = pd.to_numeric(df["forecast_qty"], errors="coerce").fillna(0).astype(int)
    return df

df = read_forecasts()
if df.empty:
    st.warning("Er zijn nog geen voorspellingen.")
    st.stop()

# Filters
c1, c2 = st.columns(2)
with c1:
    prod = st.multiselect("Producten", sorted(df["product_name"].unique().tolist()))
with c2:
    dates = st.date_input("Datumrange", [df["date"].min().date(), df["date"].max().date()])

f = df.copy()
if prod:
    f = f[f["product_name"].isin(prod)]
if isinstance(dates, list) and len(dates) == 2:
    start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1]) + pd.Timedelta(days=1)
    f = f[(f["date"] >= start) & (f["date"] < end)]

st.dataframe(f.sort_values(["date","product_name"]), use_container_width=True)
st.download_button("Download CSV", data=f.to_csv(index=False).encode("utf-8"),
                   file_name="forecasts.csv", mime="text/csv")
