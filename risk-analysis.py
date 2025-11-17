# risk_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# Optional click-event support
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    PLOTLY_EVENTS = True
except Exception:
    PLOTLY_EVENTS = False

st.set_page_config(page_title="Broadway Risk Dashboard", layout="wide")
st.title("Broadway — Interactive Risk Dashboard")
st.markdown(
    "Upload your Excel risk register and explore risks with a high-contrast heatmap (or Tree/Bubble map). "
    "Click a heatmap cell to list Risk IDs and view RISK + CONSEQUENCE/IMPACT details."
)

# ---------------------------
# Constants: exact label orders
# ---------------------------
PROBABILITY_ORDER = ["Almost Certain", "Likely", "Moderate", "Unlikely", "Rare"]
IMPACT_ORDER = ["Negligible", "Marginal", "Serious", "Critical", "Catastrophic"]

# color options (high contrast)
COLOR_OPTIONS = ["Turbo", "Viridis", "Cividis", "Plasma", "Reds"]

# ---------------------------
# Helpers
# ---------------------------
def read_excel(uploaded) -> pd.DataFrame:
    """Read excel and dedupe duplicate column headers safely."""
    df = pd.read_excel(uploaded, engine="openpyxl")
    # dedupe duplicate headers (Risk ID, Risk ID_1, etc.)
    if len(df.columns) != len(set(df.columns)):
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df

def map_exact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename exact known columns to canonical names used in the app."""
    mapping = {
        "Risk ID": "risk_id",
        "RISKS": "risk",
        "CONSEQUENCE/ IMPACT": "consequence_impact",
        "Impact / Consequence Rating": "impact_rating",
        "Probability / Likelihood Rating": "probability_rating"
    }
    df = df.rename(columns=mapping)
    return df

def normalize_rating_text(val: any) -> str | None:
    """Return a cleaned rating string if it matches one of the expected textual categories."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Accept common typos/case-insensitive mapping (e.g., "Likely", "Likeliy")
    # We'll match by first characters / case-insensitive exact words
    s_lower = s.lower()
    # map known alternatives
    alt_map = {
        "almost certain": "Almost Certain",
        "almost_certain": "Almost Certain",
        "almostcertain": "Almost Certain",
        "likely": "Likely",
        "likeliy": "Likely",   # handle typo user earlier had
        "moderate": "Moderate",
        "moderately": "Moderate",
        "unlikely": "Unlikely",
        "rare": "Rare",
        "negligible": "Negligible",
        "marginal": "Marginal",
        "serious": "Serious",
        "critical": "Critical",
        "catastrophic": "Catastrophic"
    }
    # direct normalization if exact match ignoring case
    for key, label in alt_map.items():
        if key == s_lower or key in s_lower:
            return label
    return None

def prepare_matrix(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Build a matrix (DataFrame) with index=PROBABILITY_ORDER, columns=IMPACT_ORDER
    and values = count of risks. Also return cell mapping with Risk IDs and details.
    """
    # Ensure canonical columns exist
    if not all(c in df.columns for c in ["probability_rating", "impact_rating", "risk_id", "risk", "consequence_impact"]):
        # Data missing - return empty
        empty_mat = pd.DataFrame(0, index=PROBABILITY_ORDER, columns=IMPACT_ORDER)
        return empty_mat, {}

    # Normalize rating strings
    df = df.copy()
    df["probability_rating_norm"] = df["probability_rating"].apply(normalize_rating_text)
    df["impact_rating_norm"] = df["impact_rating"].apply(normalize_rating_text)
    # Filter to rows that successfully normalized to expected categories
    df = df[df["probability_rating_norm"].notna() & df["impact_rating_norm"].notna()]

    # Create mapping and matrix
    mat = pd.DataFrame(0, index=PROBABILITY_ORDER, columns=IMPACT_ORDER)
    cell_map = {}  # (prob, impact) -> {"count": int, "risk_ids": [...], "risks": [...], "consequences": [...]}

    for _, row in df.iterrows():
        p = row["probability_rating_norm"]
        i = row["impact_rating_norm"]
        if p not in PROBABILITY_ORDER or i not in IMPACT_ORDER:
            continue
        mat.at[p, i] += 1
        key = (p, i)
        if key not in cell_map:
            cell_map[key] = {"count": 0, "risk_ids": [], "risks": [], "consequences": []}
        cell_map[key]["count"] += 1
        cell_map[key]["risk_ids"].append(str(row.get("risk_id", "")))
        cell_map[key]["risks"].append(str(row.get("risk", "")))
        cell_map[key]["consequences"].append(str(row.get("consequence_impact", "")))
    return mat, cell_map

def build_heatmap_figure(mat: pd.DataFrame, color_scale: str, value_mode: str, cell_map: dict, filters_text: str) -> go.Figure:
    """
    value_mode: "Count" or "AverageSeverity"
    For AverageSeverity we compute severity = ordinal(prob) * ordinal(impact) and average per cell.
    """
    z = mat.values.astype(float)  # counts by default

    # If AverageSeverity requested, compute severity average per cell
    if value_mode == "AverageSeverity":
        # Placeholder: we'll convert category to numeric scale 1..5 by their index
        prob_idx = {label: idx+1 for idx, label in enumerate(PROBABILITY_ORDER)}
        impact_idx = {label: idx+1 for idx, label in enumerate(IMPACT_ORDER)}
        # compute sum and counts
        sum_mat = np.zeros_like(z)
        cnt_mat = np.zeros_like(z)
        # iterate cell_map to fill sums
        for r, c in cell_map.keys():
            pr = prob_idx[r]
            ic = impact_idx[c]
            key = (r, c)
            sev_each = pr * ic
            idx_r = PROBABILITY_ORDER.index(r)
            idx_c = IMPACT_ORDER.index(c)
            count = cell_map[key]["count"]
            sum_mat[idx_r, idx_c] = sev_each * count
            cnt_mat[idx_r, idx_c] = count
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_mat = np.divide(sum_mat, np.where(cnt_mat == 0, np.nan, cnt_mat))
            avg_mat = np.nan_to_num(avg_mat, nan=0.0)
        z = avg_mat

    # Build hover text from cell_map
    hover = []
    for r in PROBABILITY_ORDER:
        row_hover = []
        for c in IMPACT_ORDER:
            k = (r, c)
            info = cell_map.get(k, {"count": 0, "risk_ids": []})
            if value_mode == "AverageSeverity":
                val = float(z[PROBABILITY_ORDER.index(r), IMPACT_ORDER.index(c)])
                h = f"Probability: {r}<br>Impact: {c}<br>Avg Severity: {val:.1f}<br>Count: {info['count']}"
            else:
                cnt = int(z[PROBABILITY_ORDER.index(r), IMPACT_ORDER.index(c)])
                h = f"Probability: {r}<br>Impact: {c}<br>Count: {cnt}"
            if info["count"] > 0:
                sample = ", ".join(info["risk_ids"][:8])
                h += f"<br>Risk IDs: {sample}"
            row_hover.append(h)
        hover.append(row_hover)

    # Use high contrast color scales; plotly supports 'Turbo' as 'Turbo' when using colorscale param
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=IMPACT_ORDER,
        y=PROBABILITY_ORDER,
        hoverinfo="text",
        hovertext=hover,
        colorscale=color_scale,
        colorbar=dict(title=("Avg Severity" if value_mode == "AverageSeverity" else "Count"))
    ))
    fig.update_layout(
        xaxis_title="Impact / Consequence Rating",
        yaxis_title="Probability / Likelihood Rating",
        title_text=filters_text,
        height=650,
        margin=dict(l=60, r=20, t=80, b=60)
    )
    # Add annotations: counts in each cell for readability
    annotations = []
    for i, r in enumerate(PROBABILITY_ORDER):
        for j, c in enumerate(IMPACT_ORDER):
            v = mat.at[r, c]
            if v > 0:
                annotations.append(dict(
                    x=c, y=r, text=str(int(v)), showarrow=False,
                    font=dict(color="black", size=12)
                ))
    fig.update_layout(annotations=annotations)
    return fig

# ---------------------------
# Sidebar: Upload and options
# ---------------------------
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "xls"])
if not uploaded:
    st.info("Please upload your Excel risk register (.xlsx).")
    st.stop()

st.sidebar.subheader("Visualization options")
color_choice = st.sidebar.selectbox("Color (high contrast)", options=COLOR_OPTIONS, index=0)
value_mode_choice = st.sidebar.radio("Heatmap value", options=["Count", "AverageSeverity"], index=0,
                                     help="Count = number of risks per cell; AverageSeverity = avg (probabilityOrdinal x impactOrdinal)")
alt_choice = st.sidebar.radio("Main visualization", options=["Heatmap", "Tree Map", "Bubble Map"], index=0)

# ---------------------------
# Load and prepare data
# ---------------------------
try:
    df_raw = read_excel(uploaded)
except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

# Map columns to canonical internal names
df = map_exact_columns(df_raw)

# Ensure required canonical columns exist; create helpful errors if not
required = ["risk_id", "risk", "consequence_impact", "impact_rating", "probability_rating"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in Excel. Required columns (exact names expected in Excel header):\n"
             "- Risk ID\n- RISKS\n- CONSEQUENCE/ IMPACT\n- Impact / Consequence Rating\n- Probability / Likelihood Rating\n\n"
             f"Missing mapped columns: {missing}\nPlease correct the Excel headers and re-upload.")
    st.stop()

# ---------------------------
# Only two filters (as requested)
# ---------------------------
st.sidebar.subheader("Filters (only these two)")
prob_filter = st.sidebar.multiselect("Probability / Likelihood Rating",
                                     options=PROBABILITY_ORDER, default=PROBABILITY_ORDER)
impact_filter = st.sidebar.multiselect("Impact / Consequence Rating",
                                       options=IMPACT_ORDER, default=IMPACT_ORDER)

# Apply filters by textual normalization
df = df.copy()
df["probability_rating_norm"] = df["probability_rating"].apply(normalize_rating_text)
df["impact_rating_norm"] = df["impact_rating"].apply(normalize_rating_text)

df = df[df["probability_rating_norm"].notna() & df["impact_rating_norm"].notna()]

if prob_filter:
    df = df[df["probability_rating_norm"].isin(prob_filter)]
if impact_filter:
    df = df[df["impact_rating_norm"].isin(impact_filter)]

if df.empty:
    st.warning("No records match the chosen filters. Try widening the filters.")
    st.stop()

# Build matrix & mapping
mat, cell_map = prepare_matrix(df.rename(columns={"probability_rating":"probability_rating", "impact_rating":"impact_rating",
                                                  "risk_id":"risk_id", "risk":"risk", "consequence_impact":"consequence_impact"}))

filters_text = f"Filters — Probability: {', '.join(prob_filter) if prob_filter else 'All'} | Impact: {', '.join(impact_filter) if impact_filter else 'All'}"

# ---------------------------
# Main area: chart + interactions
# ---------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Risk Matrix")
    if alt_choice == "Heatmap":
        fig = build_heatmap_figure(mat, color_choice, ("AverageSeverity" if value_mode_choice == "AverageSeverity" else "Count"),
                                   cell_map, filters_text)
        st.markdown("**Heatmap (click a cell to see Risk IDs & details).**")
        # If plotly_events available: enable click
        if PLOTLY_EVENTS:
            clicked = plotly_events(fig, click_event=True, hover_event=False)
            st.plotly_chart(fig, use_container_width=True)
            if clicked:
                ev = clicked[0]
                # ev usually contains 'x' and 'y' as labels
                xval = ev.get("x")
                yval = ev.get("y")
                try:
                    # Clean to exact label
                    impact_label = str(xval)
                    prob_label = str(yval)
                    key = (prob_label, impact_label)
                except Exception:
                    st.error("Could not parse click coordinates.")
                    key = None
                if key:
                    info = cell_map.get(key, {"count": 0, "risk_ids": [], "risks": [], "consequences": []})
                    if info["count"] == 0:
                        st.info("No risks in this cell.")
                    else:
                        st.markdown(f"**Risks in cell ({prob_label} / {impact_label}) — {info['count']}**")
                        sel = st.selectbox("Select Risk ID to view details", options=info["risk_ids"], key="heat_sel")
                        if sel:
                            row = df[df["risk_id"].astype(str) == str(sel)]
                            if not row.empty:
                                st.markdown("**RISK**")
                                st.info(row.iloc[0].get("risk", "No description"))
                                st.markdown("**CONSEQUENCE / IMPACT**")
                                st.info(row.iloc[0].get("consequence_impact", "No details"))
        else:
            # Fallback: no click support -> show chart and use dropdowns to pick cell
            st.warning("`streamlit-plotly-events` not installed; use the cell selectors below.")
            st.plotly_chart(fig, use_container_width=True)
            sel_prob = st.selectbox("Select Probability", options=["(choose)"] + PROBABILITY_ORDER, index=0)
            sel_imp = st.selectbox("Select Impact", options=["(choose)"] + IMPACT_ORDER, index=0)
            if sel_prob != "(choose)" and sel_imp != "(choose)":
                key = (sel_prob, sel_imp)
                info = cell_map.get(key, {"count": 0, "risk_ids": []})
                if info["count"] == 0:
                    st.info("No risks in this selected cell.")
                else:
                    st.markdown(f"**Risks in cell ({sel_prob} / {sel_imp}) — {info['count']}**")
                    sel = st.selectbox("Select Risk ID to view details (fallback)", options=info["risk_ids"], key="fallback_sel")
                    if sel:
                        row = df[df["risk_id"].astype(str) == str(sel)]
                        if not row.empty:
                            st.markdown("**RISK**")
                            st.info(row.iloc[0].get("risk", "No description"))
                            st.markdown("**CONSEQUENCE / IMPACT**")
                            st.info(row.iloc[0].get("consequence_impact", "No details"))
    elif alt_choice == "Tree Map":
        st.subheader("Tree Map (groups = Probability_Impact)")
        df_tm = df.copy()
        df_tm["group"] = df_tm["probability_rating_norm"] + " / " + df_tm["impact_rating_norm"]
        agg = df_tm.groupby("group").agg(count=("risk_id","count")).reset_index()
        fig_tm = px.treemap(agg, path=["group"], values="count", color="count", color_continuous_scale=color_choice,
                            title=filters_text)
        st.plotly_chart(fig_tm, use_container_width=True)
        st.markdown("Select a group (Probability / Impact) then a Risk ID to view details.")
        groups = agg["group"].tolist()
        chosen = st.selectbox("Choose group", options=["(choose)"] + groups)
        if chosen and chosen != "(choose)":
            subset = df_tm[df_tm["group"] == chosen]
            ids = subset["risk_id"].astype(str).tolist()
            sel = st.selectbox("Choose Risk ID", options=["(choose)"] + ids)
            if sel and sel != "(choose)":
                row = df[df["risk_id"].astype(str) == str(sel)]
                if not row.empty:
                    st.markdown("**RISK**")
                    st.info(row.iloc[0].get("risk", "No description"))
                    st.markdown("**CONSEQUENCE / IMPACT**")
                    st.info(row.iloc[0].get("consequence_impact", "No details"))
    elif alt_choice == "Bubble Map":
        st.subheader("Bubble Map (each risk plotted; size = severity)")
        # Map textual categories to numeric ordinal positions so they plot nicely
        prob_to_num = {label: i+1 for i, label in enumerate(PROBABILITY_ORDER)}
        impact_to_num = {label: i+1 for i, label in enumerate(IMPACT_ORDER)}
        df_b = df.copy()
        df_b["prob_num"] = df_b["probability_rating_norm"].map(prob_to_num)
        df_b["imp_num"] = df_b["impact_rating_norm"].map(impact_to_num)
        df_b["severity"] = df_b["prob_num"] * df_b["imp_num"]
        fig_b = px.scatter(df_b, x="impact_rating_norm", y="probability_rating_norm",
                           size="severity", hover_name="risk_id",
                           hover_data={"risk": True, "consequence_impact": True, "severity": True},
                           color="severity", color_continuous_scale=color_choice,
                           category_orders={"impact_rating_norm": IMPACT_ORDER, "probability_rating_norm": PROBABILITY_ORDER},
                           title=filters_text)
        fig_b.update_layout(height=650)
        st.plotly_chart(fig_b, use_container_width=True)
        if PLOTLY_EVENTS:
            clicked = plotly_events(fig_b, click_event=True, hover_event=False)
            if clicked:
                ev = clicked[0]
                # try best-effort to get point index and risk_id
                try:
                    idx = ev.get("pointIndex")
                    sel_row = df_b.iloc[int(idx)]
                    sel_id = sel_row["risk_id"]
                    st.markdown("**Selected Risk**")
                    st.write("Risk ID:", sel_id)
                    st.info(sel_row.get("risk", "No description"))
                    st.write("CONSEQUENCE / IMPACT:")
                    st.info(sel_row.get("consequence_impact", "No details"))
                except Exception:
                    st.info("Clicked point could not be resolved. Use hover to find Risk ID.")
        else:
            st.info("Clicking points requires `streamlit-plotly-events`. Use hover to find a Risk ID and paste into the selector below.")
            hover_choice = st.text_input("Enter a Risk ID from hover to view details (e.g., R00012):")
            if hover_choice:
                row = df[df["risk_id"].astype(str) == hover_choice.strip()]
                if not row.empty:
                    st.markdown("**RISK**")
                    st.info(row.iloc[0].get("risk", "No description"))
                    st.write("CONSEQUENCE / IMPACT:")
                    st.info(row.iloc[0].get("consequence_impact", "No details"))

with col2:
    st.subheader("Summary")
    st.write("Total risks (filtered):", df["risk_id"].nunique())
    st.write("Active Probability filters:", ", ".join(prob_filter) if prob_filter else "All")
    st.write("Active Impact filters:", ", ".join(impact_filter) if impact_filter else "All")
    st.markdown("---")
    st.subheader("Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="jobnix_filtered.csv", mime="text/csv")

st.caption("Heatmap axes use the exact textual order you specified. Color and alternatives are high-contrast for visibility.")
