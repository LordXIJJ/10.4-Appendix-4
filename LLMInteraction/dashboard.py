import os
import json
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

LOG_PATH = os.path.join("logs", "actions_log.jsonl")

st.set_page_config(
    page_title="LLM Tool Call Dashboard",
    layout="wide"
)

st.title("🤖 Real-Time LLM Tool Calls")

# Auto-refresh every 800 ms
st_autorefresh(interval=800, key="refresh")

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    max_rows = st.number_input("Max rows", 50, 2000, 300, 50)
with col2:
    show_pose = st.checkbox("Show resolved pose", value=False)
with col3:
    tool_filter = st.text_input("Filter tools (comma-separated)", "")

tool_whitelist = [t.strip() for t in tool_filter.split(",") if t.strip()]

# Load JSONL
def load_events(path):
    if not os.path.exists(path):
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events

events = load_events(LOG_PATH)

if not events:
    st.warning("No tool calls logged yet. Run your main app.")
    st.stop()

df = pd.DataFrame(events)

# Optional filter
if tool_whitelist:
    df = df[df["tool"].isin(tool_whitelist)]

# Newest first
df = df.iloc[::-1].reset_index(drop=True)
df = df.head(int(max_rows))

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Total events", len(events))
m2.metric("Showing", len(df))
m3.metric("Unique tools", df["tool"].nunique())

st.divider()

# Main layout
left, right = st.columns([2, 1])

with left:
    view = df.copy()
    view["arguments"] = view["arguments"].apply(lambda x: json.dumps(x))
    if not show_pose and "resolved_pose" in view.columns:
        view = view.drop(columns=["resolved_pose"])

    st.dataframe(
        view[["timestamp", "tool", "arguments", "message"]],
        use_container_width=True,
        height=600
    )

with right:
    st.subheader("Latest event")
    st.json(df.iloc[0].to_dict())

    st.subheader("Tool usage")
    st.bar_chart(df["tool"].value_counts())
