
import streamlit as st
import pandas as pd
import re
import plotly.express as px
from pathlib import Path
from RAGAgents_pipeline import answer_query

DATA = Path(__file__).resolve().parent / "data"/ "argo_real_sample.csv"

st.set_page_config(page_title="ArgoTalk : ARGO Data Discovery And Visualization", layout="wide")

# =============== Load Data ===============
@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    for col in ["lat", "lon", "pressure_mean", "temperature_mean", "salinity_mean"]:
        if col in df.columns:  # avoid KeyError
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


df = load_data()

# =============== Sidebar (Dataset Filter) ===============
with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/3310/3310521.png", width=80)
    st.title("🌊 ArgoTalk")

    st.markdown("### Dataset Filter")
    date_range = st.date_input("Date Range (YYYY-MM)", [])
    region = st.selectbox("Region", ["Indian Ocean", "Arabian Sea"])
    float_id_input = st.text_input("Float ID (e.g., 490...)")

    st.markdown("---")
    st.caption("Filter profiles by latitude/longitude as well:")
    lat_range = st.slider("Latitude", -60.0, 60.0, (-30.0, 30.0))
    lon_range = st.slider("Longitude", -180.0, 180.0, (40.0, 100.0))

# Compute sidebar-based subset
sub = df[(df.lat >= lat_range[0]) & (df.lat <= lat_range[1]) &
         (df.lon >= lon_range[0]) & (df.lon <= lon_range[1])]

# Decide active dataset (agent > manual)
if "matches" in st.session_state and not st.session_state["matches"].empty:
    current_df = st.session_state["matches"]
else:
    current_df = sub

# =============== Main Layout ===============
st.title("ArgoTalk: ARGO Data Discovery And Visualization")

# Top Row: Chat + Map
c1, c2 = st.columns([1.1, 1.9])

with c1:
    st.subheader("💬 Chat Interface")
    query = st.text_area("Type your message here...", height=80)

    if st.button("Ask"):
        if query.strip():
            with st.spinner("Searching..."):
                result = answer_query(query, top_k=5)

                # Always show answer text
                st.success(result["answer"])

                # Save state for other components
                st.session_state["action"] = result["type"]
                st.session_state["matches"] = result["matches"]

                # Extract parameter from query for plotting
                if result["type"] in ["parameter", "comparison"]:
                    # Example: simple keyword check
                    if "temperature" in query.lower():
                        st.session_state["agent_param"] = "temperature_mean"
                    elif "salinity" in query.lower():
                        st.session_state["agent_param"] = "salinity_mean"
                    elif "pressure" in query.lower():
                        st.session_state["agent_param"] = "pressure_mean"
                    else:
                        # fallback default
                        st.session_state["agent_param"] = "temperature_mean"


                # Show matches if available
                if not result["matches"].empty:
                    st.dataframe(result["matches"][["float_id", "date", "lat", "lon"]])
                    csv = result["matches"].to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", csv, "matches.csv", "text/csv")

                # Show visualization if exists
                if result["figure"] is not None:
                    st.plotly_chart(result["figure"], use_container_width=True)
        else:
            st.warning("Please enter a query.")

    st.markdown("---")
    st.markdown("**Select Parameter**")
    param = st.selectbox("Parameter",
                         ["temperature_mean", "salinity_mean", "pressure_mean",
                          "nitrate", "chlorophyll_a"])

with c2:
    st.subheader("🗺️ Geospatial Map")

    if "action" in st.session_state and st.session_state["action"] == "map":
        matches = st.session_state["matches"]
        if not matches.empty:
            st.map(matches.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
        else:
            st.info("No profiles found for this query.")
    else:
        # Default sidebar-driven map
        sub = df[(df.lat >= lat_range[0]) & (df.lat <= lat_range[1]) &
                 (df.lon >= lon_range[0]) & (df.lon <= lon_range[1])]
        st.write(f"Profiles in region: {len(sub)}")
        if not sub.empty:
            st.map(sub.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
        else:
            st.info("No profiles found. Adjust filters or ask the AI assistant.")

# Middle Row: Parameter Selection (Depth-Time Series) + Profile Comparison Chart
c3, c4 = st.columns(2)

with c3:
    st.subheader("📈 Parameter/Depth-Time Series Plot")

    # Determine which parameter to use
    if "action" in st.session_state and st.session_state["action"] == "parameter":
        matches = st.session_state["matches"]
        if not matches.empty:


            # Agent-driven parameter detection (default = temperature)
            agent_param = st.session_state.get("agent_param", "temperature_mean")

            # Only plot if param exists
            if agent_param in matches.columns:
                fig = px.scatter(
                    matches, x=agent_param, y="pressure_mean", color="float_id",
                    hover_data=["date", "float_id"], title=f"{agent_param} vs Depth"
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ Column '{agent_param}' not found in dataset.")
        else:
            st.info("No profiles found for this query.")
    else:
        # Manual selection via sidebar
        if not df.empty:
            filtered_df = df[(df.lat >= lat_range[0]) & (df.lat <= lat_range[1]) &
                             (df.lon >= lon_range[0]) & (df.lon <= lon_range[1])]
            fig = px.scatter(filtered_df, x=param, y="pressure_mean", color="float_id",
                             hover_data=["date", "float_id"], title=f"{param} vs Depth")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No profiles selected for comparison.")

with c4:
        st.subheader("📊 Profile Comparison")

        # Use current_df (either agent matches or manual subset)
        comp_df = current_df.copy()

        # Determine parameter: agent-driven or manual fallback
        comp_param = st.session_state.get("agent_param", param)

        if not comp_df.empty:
            if comp_param in comp_df.columns:
                fig = px.line(
                    comp_df,
                    x="pressure_mean",
                    y=comp_param,
                    color="float_id",
                    hover_data=["date", "lat", "lon"],
                    title=f"{comp_param} Profile Comparison"
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ Column '{comp_param}' not found in dataset.")
        else:
            st.info("No profiles selected for comparison.")

# Bottom Row: Table Profile Comparison
st.subheader("📑 Profile Comparison Table")
if not current_df.empty:
    st.dataframe(
        current_df[["float_id", "date", "lat", param]]
        .rename(columns={"lat": "Latitude", param: f"{param} (units)"})
        .head(10)
    )
else:
    st.info("No profiles available in this selection.")

# Optional KPIs
if not current_df.empty and param in current_df.columns:
    k1, k2, k3 = st.columns(3)
    k1.metric("Min " + param, round(current_df[param].min(), 2))
    k2.metric("Max " + param, round(current_df[param].max(), 2))
    k3.metric("Mean " + param, round(current_df[param].mean(), 2))