# viewer.py — Alleen-lezen forecast viewer
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import pytz

st.set_page_config(page_title="Bakery Forecast - Viewer", page_icon="📈", layout="wide")
st.title("📈 Bakery Forecast – Viewer")

# --- Secrets / service account ---
try:
    sa_dict = dict(st.secrets["gcp_service_account"])
    SHEET_ID = st.secrets["gcp_service_account"]["SHEET_ID"]
except Exception:
    st.error("Secrets ontbreken. Voeg gcp_service_account + SHEET_ID toe bij Streamlit → Settings → Secrets.")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
creds = Credentials.from_service_account_info(sa_dict, scopes=SCOPES)
gc = gspread.authorize(creds)

# --- Data inladen uit 'forecasts' ---
@st.cache_data(ttl=60)  # 1 minuut cache
def load_forecasts(sheet_id: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet("forecasts")
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # types & nette kolommen
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "forecast_qty" in df.columns:
        df["forecast_qty"] = pd.to_numeric(df["forecast_qty"], errors="coerce").fillna(0).round().astype(int)
    if "run_ts" in df.columns:
        # run_ts is tekst; toon als-is
        pass
    return df

df = load_forecasts(SHEET_ID)

if df.empty:
    st.warning("Nog geen forecasts gevonden. Vraag je baas om de upload/forecast-app te draaien.")
    st.stop()

# --- Filters sidebar ---
with st.sidebar:
    st.header("Filters")
    # datumfilter
    min_d, max_d = df["date"].min(), df["date"].max()
    d_range = st.date_input("Datumrange", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
    # productfilter
    q = st.text_input("Zoek productnaam / SKU", "")
    # sorteren
    sort_col = st.selectbox("Sorteren op", ["date", "sku_id", "product_name", "forecast_qty"])
    sort_asc = st.toggle("Oplopend", value=True)

# toepassen filters
mask = (df["date"].dt.date >= d_range[0]) & (df["date"].dt.date <= d_range[1])
if q.strip():
    ql = q.lower()
    mask &= df["product_name"].str.lower().str.contains(ql) | df["sku_id"].astype(str).str.lower().str.contains(ql)

view = df.loc[mask].copy().sort_values(sort_col, ascending=sort_asc)

# --- Info bovenaan ---
last_run = df["run_ts"].max() if "run_ts" in df.columns else "—"
left, right = st.columns([1,1])
with left:
    st.markdown(f"**Aantal rijen:** {view.shape[0]}")
with right:
    st.markdown(f"**Laatste run_ts:** `{last_run}`")

# --- Tabel ---
st.dataframe(
    view[["date", "sku_id", "product_name", "forecast_qty", "run_ts"]]
      .sort_values(["date", "sku_id"]),
    use_container_width=True,
    height=520,
)

# --- Download knop ---
csv = view.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="forecasts.csv", mime="text/csv")
