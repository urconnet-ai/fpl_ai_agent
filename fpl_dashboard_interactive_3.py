import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import requests  # make sure it's at top of file, but ok to have here if already imported
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time 
import difflib
import os

# phase 2 engine (your local file)
from fpl_ai_agents_phase2_ml import (
    fetch_fpl_data,
    compute_fixture_difficulty,
    train_price_model,
    apply_price_predictions,
    compute_player_metrics,
    analyze_with_gemini,
    save_to_sqlite,
)

#DB_PATH = "fpl_ai.db"
DB_PATH = DB_PATH = os.path.join(os.getcwd(), "fpl_ai.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS weekly_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gw INTEGER,
    commentary TEXT,
    created_at TEXT,
    risers TEXT,
    fallers TEXT,
    watchlist TEXT,
    captaincy_picks TEXT,
    value_picks TEXT,
    bandwagons TEXT
)
""")
conn.commit()
conn.close()

# =========================================================
# PAGE CONFIG & THEME
# =========================================================
st.set_page_config(
    page_title="FPL AI Agent — Market Intelligence",
    layout="wide"
)

st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at 20% 20%, #1f1026 0%, #0b0b14 45%, #020203 100%);
    color: #f5f5f5;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container {
    backdrop-filter: blur(20px);
    background-color: rgba(6, 6, 12, 0.18);
    border-radius: 18px;
    padding: 1.5rem 2rem 2.5rem 2rem;
    border: 1px solid rgba(255,255,255,0.015);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.02);
    border-radius: 14px 14px 0 0;
    padding: 0.45rem 1.15rem;
    transition: all .2s ease-in-out;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(170,92,219,0.2);
    transform: translateY(-2px);
}
.stTabs [aria-selected="true"] {
    background: rgba(170,92,219,0.28);
    color: #fff;
    box-shadow: 0 6px 18px rgba(170,92,219,0.35);
}
.stDataFrame, .dataframe {
    background: rgba(10,10,14,0.6) !important;
    border: 1px solid rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
}
.js-plotly-plot, .plotly {
    border-radius: 14px !important;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 0.7rem 1rem;
    backdrop-filter: blur(10px);
}
hr.glow-line {
    margin-top:2.5rem;
    margin-bottom:1.8rem;
    border:none;
    height:1px;
    background:linear-gradient(90deg, rgba(170,92,219,0) 0%, rgba(170,92,219,0.4) 50%, rgba(170,92,219,0) 100%);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def load_latest_report(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM weekly_reports ORDER BY id DESC LIMIT 1", conn)
    conn.close()
    if df.empty:
        return None
    row = df.iloc[0]
    result = {}
    for col in df.columns:
        if col in ["id", "gw", "commentary", "created_at"]:
            result[col] = row[col]
        else:
            # every other column is JSON
            try:
                result[col] = json.loads(row[col])
            except Exception:
                result[col] = []
    return result


def to_df(items):
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    # normalize names
    if "player_name" in df.columns:
        df = df.rename(columns={"player_name": "Player"})
    if "name" in df.columns and "Player" not in df.columns:
        df = df.rename(columns={"name": "Player"})
    if "team" in df.columns:
        df = df.rename(columns={"team": "Team"})
    # de-dup column name clashes
    df = df.loc[:, ~df.columns.duplicated()]
    # drop ids that make no sense to show
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def safe_numeric_col(df, col, default=0):
    if col not in df.columns:
        df[col] = default
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df



def plot_vertical_price(df, title, color):
    """
    Plot using actual price (your new requirement) and show ONLY 10 players.
    """
    if df.empty or "price" not in df.columns:
        return None
    tmp = df.copy().head(10)
    if "Player" in tmp.columns:
        tmp["Player"] = tmp["Player"].astype(str).str.slice(0, 18)
    fig = px.bar(
        tmp,
        x="Player",
        y="price",
        title=title,
        text="price",
        color_discrete_sequence=[color],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-35, title=None),
        yaxis=dict(title=None),
        height=380,
        margin=dict(l=8, r=8, t=48, b=65),
        title=dict(x=0.45)
    )
    return fig


def get_next_opponent(team_name, fixtures, teams, current_gw):
    next_gw = (current_gw or 1) + 1
    for fx in fixtures:
        if fx.get("event") != next_gw:
            continue
        th, ta = teams[fx["team_h"]], teams[fx["team_a"]]
        if th == team_name:
            return ta, fx.get("team_h_difficulty", "-")
        if ta == team_name:
            return th, fx.get("team_a_difficulty", "-")
    return "TBD", "-"



# =========================================================
# LOAD DATA / PIPELINE
# =========================================================
st.title("FPL AI Agent — Market Intelligence")

if st.button("Get Latest Stats"):
    with st.spinner("Running full AI pipeline…"):
        try:
            players, gw, fixtures, teams = fetch_fpl_data()
            team_diff = compute_fixture_difficulty(fixtures, teams)
            model, scaler = train_price_model(players)
            apply_price_predictions(players, model, scaler)
            compute_player_metrics(players, team_diff)
            analysis = analyze_with_gemini(players, team_diff)
            save_to_sqlite(gw, analysis)
            #st.success(f"Pipeline complete for Gameweek {gw}")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

analysis = load_latest_report()
if not analysis:
    st.warning("Fetch Stats to proceed.")
    st.stop()

# pull fresh FPL again for raw view
players_raw, current_gw, fixtures, teams = fetch_fpl_data()
players_raw_df = pd.DataFrame(players_raw)

# =========================================================
# CANONICALIZE COLUMNS
# =========================================================
if "total_points" not in players_raw_df.columns:
    if "points" in players_raw_df.columns:
        players_raw_df["total_points"] = players_raw_df["points"]
    else:
        players_raw_df["total_points"] = 0

if "transfers_in_event" not in players_raw_df.columns:
    if "transfers_in" in players_raw_df.columns:
        players_raw_df["transfers_in_event"] = players_raw_df["transfers_in"]
    else:
        players_raw_df["transfers_in_event"] = 0

if "transfers_out_event" not in players_raw_df.columns:
    if "transfers_out" in players_raw_df.columns:
        players_raw_df["transfers_out_event"] = players_raw_df["transfers_out"]
    else:
        players_raw_df["transfers_out_event"] = 0

if "selected_by_percent" not in players_raw_df.columns:
    players_raw_df["selected_by_percent"] = 0

if "price" not in players_raw_df.columns:
    players_raw_df["price"] = 0

if "status" not in players_raw_df.columns:
    players_raw_df["status"] = "Available"

# =========================================================
# EXECUTIVE SUMMARY BANNER (with space after)
# =========================================================
st.subheader(f"Gameweek {analysis['gw']} — Executive Summary")
st.markdown(
    f"<p style='color:#d4d4d8; font-size:0.92rem;'>{analysis.get('commentary','No commentary')}</p>",
    unsafe_allow_html=True
)

# metrics
players_raw_df = safe_numeric_col(players_raw_df, "selected_by_percent")
players_raw_df = safe_numeric_col(players_raw_df, "transfers_in_event")
players_raw_df = safe_numeric_col(players_raw_df, "transfers_out_event")

Average_ownership = players_raw_df["selected_by_percent"].mean()
market_pulse = (players_raw_df["transfers_in_event"] - players_raw_df["transfers_out_event"]).abs().mean()
next_gw = (current_gw or analysis["gw"]) + 1

def market_mood(Average_own, vol):
    if Average_own < 5 and vol > 10000:
        return "🟢 Differential Market — Act Early on Rising Players"
    elif Average_own < 10 and vol <= 10000:
        return "🟠 Balanced Market — Quiet Before the Storm"
    elif Average_own >= 20 and vol > 15000:
        return "🔴 Hype Cycle — Beware of Overreaction"
    elif Average_own >= 20 and vol <= 10000:
        return "⚪ Stable Template — Safe Week to Hold"
    else:
        return "🟣 Mixed Sentiment — Observe Bandwagons Carefully"

market_sentiment = market_mood(Average_ownership, market_pulse)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players Scanned", f"{len(players_raw_df):,}", help=f"Total players scanned in Gameweek {analysis['gw']}.")
c2.metric("Average Ownership ", f"{Average_ownership:.1f}%", help="Average % of all managers owning each player — higher = template-heavy GW.")
c3.metric("Transfer Volatility ", f"{int(market_pulse):,}", help="Mean absolute net transfers — higher = market reacting hard.")
c4.metric("Next GW", f"{next_gw}")

st.markdown(
    f"""
    <div style='margin-top:1rem; padding:1rem; border-radius:12px;
                background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.05);
                backdrop-filter:blur(10px); text-align:center;'>
        <p style='font-size:1.1rem; color:#e0e0e0; font-weight:500;'>
            {market_sentiment}
        </p>
        <p style='font-size:0.9rem; color:#b0b0b0;'>
            Average ownership shows how consolidated the template is. Volatility shows how much managers are reacting this week.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 👇 line space you asked for
st.write("")

# =========================================================
# MAIN TABS
# =========================================================
tabs = st.tabs([
    "Transfers",
    "Price Movement",
    "Watchlist",
    "Captaincy Picks",
    "Value Picks",
    "Bandwagons",
    "Manager Insights",
    "Fixture Planner",
    "Template vs Differential",
    "Predictions",
    "Team Comparison",
    "AI Copilot",
])


# ---------------------------------------------------------
# 0. TRANSFERS (DO NOT CHANGE)
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("#### Transfers — where FPL managers are moving this week")
    st.caption("Top transfer activity based on official FPL data. Left = IN, right = OUT.")

    # Auto-normalize transfer columns
    if "transfers_in_event" not in players_raw_df.columns:
        if "transfers_in" in players_raw_df.columns:
            players_raw_df["transfers_in_event"] = players_raw_df["transfers_in"]
        else:
            players_raw_df["transfers_in_event"] = 0

    if "transfers_out_event" not in players_raw_df.columns:
        if "transfers_out" in players_raw_df.columns:
            players_raw_df["transfers_out_event"] = players_raw_df["transfers_out"]
        else:
            players_raw_df["transfers_out_event"] = 0

    # Build the dataframe
    df_transfers = players_raw_df[[
        "name", "team", "price", "transfers_in_event", "transfers_out_event",
        "selected_by_percent", "status"
    ]].rename(columns={
        "name": "Player",
        "team": "Team",
        "price":"Price",
        "transfers_in_event": "Transfers In",
        "transfers_out_event": "Transfers Out",
        "selected_by_percent": "% Selected",
        "status":"Status",
    })

    # Ensure numeric & handle missing
    df_transfers["Transfers In"] = pd.to_numeric(df_transfers["Transfers In"], errors="coerce").fillna(0)
    df_transfers["Transfers Out"] = pd.to_numeric(df_transfers["Transfers Out"], errors="coerce").fillna(0)

    # Filter out zero-transfer rows
    df_in = df_transfers[df_transfers["Transfers In"] > 0].sort_values(by="Transfers In", ascending=False).head(10)
    df_out = df_transfers[df_transfers["Transfers Out"] > 0].sort_values(by="Transfers Out", ascending=False).head(10)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Top Transfers In (Official FPL)")
        if not df_in.empty:
            fig_in = px.bar(df_in, x="Player", y="Transfers In",
                            color_discrete_sequence=["#00e676"], text="Transfers In")
            fig_in.update_traces(textposition="outside", cliponaxis=False)
            fig_in.update_layout(template="plotly_dark", height=380,
                                 margin=dict(l=8, r=8, t=48, b=65))
            st.plotly_chart(fig_in, use_container_width=True)
            st.dataframe(df_in, use_container_width=True, hide_index=True)
        else:
            st.info("No current transfer-in data found from FPL API.")

    with c2:
        st.markdown("##### Top Transfers Out (Official FPL)")
        if not df_out.empty:
            fig_out = px.bar(df_out, x="Player", y="Transfers Out",
                             color_discrete_sequence=["#ff4d4d"], text="Transfers Out")
            fig_out.update_traces(textposition="outside", cliponaxis=False)
            fig_out.update_layout(template="plotly_dark", height=380,
                                  margin=dict(l=8, r=8, t=48, b=65))
            st.plotly_chart(fig_out, use_container_width=True)
            st.dataframe(df_out, use_container_width=True, hide_index=True)
        else:
            st.info("No current transfer-out data found from FPL API.")

# ---------------------------------------------------------
# 1. PRICE MOVEMENT — Cleaned & merged with live data (robust)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. PRICE MOVEMENT — Final (Always 10 Rows)
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("#### Price Movement — expected rises and fallers")
    st.caption("Ranked by AI confidence (prob_rise / prob_fall). Always 10 entries per table, fully enriched and cleaned.")

    import difflib
    import numpy as np

    # ---------- Helper: Enrich AI Data with Live FPL ----------
    def enrich_with_live(df, players_raw_df):
        """Merge AI results with live FPL data using fuzzy matching and safe filling."""
        if df.empty:
            return df

        df = df.rename(columns=lambda x: x.strip())
        df["Player"] = df["Player"].astype(str)

        live_cols = [c for c in ["web_name", "name"] if c in players_raw_df.columns]
        live_name_col = live_cols[0] if live_cols else None

        if live_name_col:
            live_names = players_raw_df[live_name_col].str.lower().tolist()
            df["match_name"] = df["Player"].str.lower().apply(
                lambda x: difflib.get_close_matches(x, live_names, n=1, cutoff=0.6)
            )
            df["match_name"] = df["match_name"].apply(lambda x: x[0] if x else None)

            merged = df.merge(
                players_raw_df,
                how="left",
                left_on="match_name",
                right_on=live_name_col,
                suffixes=("", "_live"),
            )
        else:
            merged = df.copy()

        # Fill numeric fields with column means
        for col in ["price", "predicted_price", "prob_rise", "prob_fall", "total_points"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
                mean_val = merged[col].replace(0, np.nan).mean()
                merged[col] = merged[col].fillna(mean_val if not np.isnan(mean_val) else 0).round(1)

        # Fill categorical defaults
        merged["Team"] = merged.get("Team", merged.get("team", "N/A")).fillna("N/A")
        merged["predicted_trend"] = merged.get("predicted_trend", "Stable").fillna("Stable")
        merged["status"] = merged.get("status", merged.get("status_live", "Available")).fillna("Available")

        merged = merged.drop_duplicates(subset=["Player"], keep="first")
        return merged

    # ---------- Helper: Fill to Exactly 10 ----------
    def fill_to_ten(df, players_raw_df, field):
        """Ensure exactly 10 entries, even if enrichment dropped rows."""
        live_cols = [c for c in ["name", "web_name"] if c in players_raw_df.columns]
        price_col = (
            "now_cost" if "now_cost" in players_raw_df.columns
            else ("price" if "price" in players_raw_df.columns else None)
        )
        col_list = live_cols + ["team", "total_points"]
        if price_col:
            col_list.append(price_col)
        available_cols = [c for c in col_list if c in players_raw_df.columns]

        # Add fallback if fewer than 10
        while len(df) < 10:
            missing = 10 - len(df)
            fallback = (
                players_raw_df.nlargest(20, field)[available_cols]
                .rename(columns={
                    live_cols[0] if live_cols else "name": "Player",
                    "team": "Team",
                    price_col if price_col else "price": "price"
                })
            )
            fallback["predicted_price"] = fallback["price"]
            fallback["predicted_trend"] = "Stable"
            fallback["prob_rise"] = df["prob_rise"].replace(0, np.nan).mean() if "prob_rise" in df else 0
            fallback["prob_fall"] = df["prob_fall"].replace(0, np.nan).mean() if "prob_fall" in df else 0
            fallback["status"] = "Available"

            df = pd.concat([df, fallback.head(missing)], ignore_index=True).drop_duplicates("Player")

            # stop loop if fallback didn’t add enough (avoid infinite loop)
            if len(fallback) == 0:
                break

        return df.head(10)

    # ---------- Load and Process ----------
    df_risers = enrich_with_live(to_df(analysis.get("risers", [])), players_raw_df)
    df_fallers = enrich_with_live(to_df(analysis.get("fallers", [])), players_raw_df)

    # Final post-fix padding for guaranteed 10 rows
    df_risers = fill_to_ten(df_risers, players_raw_df, "transfers_in_event")
    df_fallers = fill_to_ten(df_fallers, players_raw_df, "transfers_out_event")

    c1, c2 = st.columns(2)

    # ---------- RISERS ----------
    with c1:
        if not df_risers.empty:
            df_risers = safe_numeric_col(df_risers, "prob_rise").sort_values("prob_rise", ascending=False).head(10)

            fig = plot_vertical_price(df_risers, "Top Price Risers (by AI confidence)", "#00d884")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_risers[
                    ["Player", "Team", "price", "predicted_price", "predicted_trend",
                     "prob_rise", "status"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No risers detected.")

    # ---------- FALLERS ----------
    with c2:
        if not df_fallers.empty:
            df_fallers = safe_numeric_col(df_fallers, "prob_fall").sort_values("prob_fall", ascending=False).head(10)

            fig = plot_vertical_price(df_fallers, "Top Price Fallers (by AI confidence)", "#ff5d5d")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_fallers[
                    ["Player", "Team", "price", "predicted_price", "predicted_trend",
                     "prob_fall", "status"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No fallers detected.")


# ---------------------------------------------------------
# 2. WATCHLIST (de-duplicate transfers_in, show 10, plot by actual price)
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("#### Watchlist — AI + Popularity")
    st.caption("Hub/Scout-style view, but driven by your AI. 10 players max in both chart and table.")

    df_w = to_df(analysis.get("watchlist", []))

    if df_w.empty:
        # fallback to real FPL
        pr = players_raw_df.copy()
        pr = safe_numeric_col(pr, "selected_by_percent")
        pr = safe_numeric_col(pr, "transfers_in_event")
        df_w = pr[["name", "team", "price", "selected_by_percent", "transfers_in_event", "status"]].copy()
        df_w = df_w.rename(columns={
            "name": "Player",
            "team": "Team",
            "transfers_in_event": "transfers_in"
        })
        df_w["predicted_price"] = df_w["price"]
    else:
        # normalize
        if "transfers_in" not in df_w.columns:
            df_w["transfers_in"] = 0
        if "predicted_price" not in df_w.columns:
            df_w["predicted_price"] = df_w.get("price", 0)

    df_w = df_w.loc[:, ~df_w.columns.duplicated()]
    df_w = safe_numeric_col(df_w, "transfers_in")
    df_w = safe_numeric_col(df_w, "predicted_price")
    df_w = safe_numeric_col(df_w, "price")
    df_w = df_w.sort_values(by=["transfers_in", "predicted_price"], ascending=[False, False]).head(10)

    fig = plot_vertical_price(df_w, "Top Watchlist Players (price view)", "#b266ff")
    if fig: st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_w[["Player", "Team", "price", "predicted_price",  "status"]],
        hide_index=True, use_container_width=True
    )

# ---------------------------------------------------------
# 3. CAPTAINCY PICKS (show opponent)
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("#### Captaincy Picks — top 10 based on ownership, points & price")
    df_c = to_df(analysis.get("captaincy_picks", []))

    if df_c.empty:
        pr = players_raw_df.copy()
        pr = safe_numeric_col(pr, "selected_by_percent")
        pr = safe_numeric_col(pr, "price")
        pr["predicted_price"] = pr["price"]
        df_c = pr.rename(columns={"name": "Player", "team": "Team"})

    df_c = safe_numeric_col(df_c, "selected_by_percent")
    df_c = safe_numeric_col(df_c, "predicted_price")
    df_c = safe_numeric_col(df_c, "price")

    # add next opponent
    opponents, diffs = [], []
    for _, r in df_c.iterrows():
        team_name = r.get("Team") or r.get("team")
        o, d = get_next_opponent(team_name, fixtures, teams, current_gw)
        opponents.append(o)
        diffs.append(d)
    df_c["Opponent"] = opponents
    df_c["Fixture Difficulty"] = diffs

    df_c = df_c.sort_values(by=["selected_by_percent", "predicted_price"], ascending=[False, False]).head(10)

    fig = plot_vertical_price(df_c, "Top Captaincy Picks (by price)", "#00b7ff")
    if fig: st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_c[["Player", "Team", "Opponent", "Fixture Difficulty", "price", "predicted_price", "status"]],
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------------
# 4. VALUE PICKS — 10 players, by points/price
# ---------------------------------------------------------
with tabs[4]:
    st.markdown("#### Value Picks — best points per £")
    st.caption("Useful for budget squads, wildcards, and bench depth.")

    df_v = to_df(analysis.get("value_picks", []))
    if df_v.empty:
        pr = players_raw_df.copy()
        pr = safe_numeric_col(pr, "total_points")
        pr = safe_numeric_col(pr, "price")
        pr["value_index"] = pr["total_points"] / (pr["price"] + 0.1)
        df_v = pr.rename(columns={"name": "Player", "team": "Team"})
        df_v["predicted_price"] = df_v["price"]
    else:
        if "value_index" not in df_v.columns:
            df_v["value_index"] = df_v.get("prob_rise", 0)
        if "predicted_price" not in df_v.columns:
            df_v["predicted_price"] = df_v.get("price", 0)

    df_v = safe_numeric_col(df_v, "value_index")
    df_v = safe_numeric_col(df_v, "price")
    df_v = safe_numeric_col(df_v, "predicted_price")
    df_v = df_v.sort_values(by="value_index", ascending=False).head(10)

    fig = plot_vertical_price(df_v, "Top Value Picks (actual price)", "#00d8ff")
    if fig: st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_v[["Player", "Team", "price", "predicted_price","status"]],
        hide_index=True, use_container_width=True
    )

# ---------------------------------------------------------
# 5. BANDWAGONS — 10 players, de-dup, price plot
# ---------------------------------------------------------
with tabs[5]:
    st.markdown("#### Bandwagons — sudden market moves")
    st.caption("Players with aggressive transfer momentum (in or out).")

    df_b = to_df(analysis.get("bandwagons", []))
    if df_b.empty:
        pr = players_raw_df.copy()
        pr = safe_numeric_col(pr, "transfers_in_event")
        pr = safe_numeric_col(pr, "transfers_out_event")
        pr["momentum"] = pr["transfers_in_event"] - pr["transfers_out_event"]
        df_b = pr.rename(columns={"name": "Player", "team": "Team"})
        df_b["predicted_price"] = df_b["price"]
    else:
        if "momentum" not in df_b.columns:
            # try to recreate
            df_b["momentum"] = df_b.get("transfers_in", 0) - df_b.get("transfers_out", 0)
        if "predicted_price" not in df_b.columns:
            df_b["predicted_price"] = df_b.get("price", 0)

    df_b = safe_numeric_col(df_b, "momentum")
    df_b = safe_numeric_col(df_b, "price")
    df_b = safe_numeric_col(df_b, "predicted_price")
    df_b = df_b.sort_values(by="momentum", ascending=False).head(10)

    fig = plot_vertical_price(df_b, "Top Bandwagons (price view)", "#ffaa33")
    if fig: st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        df_b[["Player", "Team", "price", "predicted_price", "status"]],
        hide_index=True, use_container_width=True
    )

# ---------------------------------------------------------
# 6. MANAGER INSIGHTS (your big quadrant)
# ---------------------------------------------------------
with tabs[6]:
    st.markdown("#### Manager Insights — tactical decision layer")
    st.caption("Deeper intelligence for managers: opportunity, risk, team trends, and alerts.")

    sub1, sub2, sub3, sub4 = st.tabs([
        "Opportunities",
        "Risks",
        "Team Momentum",
        "AI Alerts",
    ])

    base = players_raw_df.copy()
    base = safe_numeric_col(base, "total_points")
    base = safe_numeric_col(base, "selected_by_percent")
    base = safe_numeric_col(base, "transfers_in_event")
    base = safe_numeric_col(base, "transfers_out_event")
    team_diff_map = compute_fixture_difficulty(fixtures, teams)
    base["fixture_difficulty_next"] = base["team"].map(team_diff_map).fillna(3)

    with sub1:
        st.markdown("##### Player Opportunity Matrix")
        opp_df = base[(base["total_points"] > 0) | (base["transfers_in_event"] > 0)].copy()
        opp_df["opp_score"] = (
            (6 - opp_df["fixture_difficulty_next"]) * 0.35 +
            (opp_df["total_points"] / (opp_df["price"] + 0.1)) * 0.4 +
            (opp_df["transfers_in_event"] / (opp_df["transfers_in_event"].max() + 1)) * 0.25
        )
        fig_opp = px.scatter(
            opp_df,
            x="fixture_difficulty_next",
            y="total_points",
            size="transfers_in_event",
            color="selected_by_percent",
            hover_data=["name", "team", "price", "status"],
            title="Opportunity: Points vs Fixture Difficulty (bubble = transfers in)",
            color_continuous_scale="Viridis",
        )
        fig_opp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            xaxis_title="Next GW Difficulty (1 easy → 5 hard)",
            yaxis_title="Total Points",
        )
        st.plotly_chart(fig_opp, use_container_width=True)
        st.dataframe(
            opp_df.sort_values(by="opp_score", ascending=False).head(10)[
                ["name", "team", "price", "fixture_difficulty_next", "total_points", "transfers_in_event", "selected_by_percent", "opp_score"]
            ].rename(columns={"name":"Player","team":"Team","fixture_difficulty_next":"Next GW Diff"}),
            hide_index=True,
            use_container_width=True
        )

    with sub2:
        st.markdown("##### Risk Surface — overhyped or fragile assets")
        risk_df = base.copy()
        risk_df["inj_flag"] = risk_df["status"].isin(["Injured", "Suspended", "Doubtful"]).astype(int)
        risk_df["risk_score"] = (
            (risk_df["selected_by_percent"] / (risk_df["selected_by_percent"].max() + 1)) * 0.4 +
            (risk_df["fixture_difficulty_next"] / 5) * 0.3 +
            (risk_df["inj_flag"] * 0.3)
        )
        risky_top = risk_df.sort_values(by="risk_score", ascending=False).head(25)
        fig_risk = px.bar(
            risky_top,
            x="name",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="Reds",
            title="Risk Index (higher = more risky to own)",
        )
        fig_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45),
            height=420,
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        risky_top["Risk Reason"] = risky_top.apply(
            lambda r: ", ".join([
                reason for reason in [
                    "Injury/suspended" if r["inj_flag"] == 1 else None,
                    "Hard fixture" if r["fixture_difficulty_next"] >= 4 else None,
                    "High ownership" if r["selected_by_percent"] >= 15 else None,
                ] if reason
            ]) or "General volatility",
            axis=1
        )
        st.dataframe(
            risky_top[["name","team","status","fixture_difficulty_next","selected_by_percent","risk_score","Risk Reason"]]
            .rename(columns={"name":"Player","team":"Team","fixture_difficulty_next":"Next GW Diff","risk_score":"Risk Score","selected_by_percent":"% Selected"}),
            hide_index=True,
            use_container_width=True
        )

    with sub3:
        st.markdown("##### Team Momentum — where the market is moving")
        tm = base.groupby("team", as_index=False).agg({
            "total_points": "mean",
            "fixture_difficulty_next": "mean",
            "transfers_in_event": "sum",
            "transfers_out_event": "sum",
        })
        tm["net_transfers"] = tm["transfers_in_event"] - tm["transfers_out_event"]
        tm = tm.sort_values(by="net_transfers", ascending=False)
        fig_team = go.Figure()
        fig_team.add_trace(
            go.Bar(
                x=tm["team"],
                y=tm["net_transfers"],
                name="Net Transfers",
                marker_color="#00e676",
                yaxis="y1"
            )
        )
        fig_team.add_trace(
            go.Scatter(
                x=tm["team"],
                y=tm["total_points"],
                name="Average Points",
                mode="lines+markers",
                line=dict(color="#3399ff", width=3),
                marker=dict(size=8),
                yaxis="y2"
            )
        )
        fig_team.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Team Momentum: Net Transfers vs Average Points",
            height=460,
            xaxis=dict(tickangle=-35, title="Team"),
            yaxis=dict(title="Net Transfers", side="left", showgrid=False),
            yaxis2=dict(title="Average Points", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center"),
        )
        st.plotly_chart(fig_team, use_container_width=True)
        st.dataframe(
            tm[["team","total_points","fixture_difficulty_next","net_transfers"]]
            .rename(columns={"team":"Team","total_points":"Average Points","fixture_difficulty_next":"Next GW Diff","net_transfers":"Net Transfers"}),
            hide_index=True,
            use_container_width=True
        )

    with sub4:
        st.markdown("##### AI Alerts — manager-focused signals")
        df_alert = base.copy()
        df_alert = safe_numeric_col(df_alert, "selected_by_percent")
        df_alert = safe_numeric_col(df_alert, "total_points")
        df_alert = df_alert[df_alert["selected_by_percent"] > 5].copy()

        overhyped = df_alert[
            (df_alert["fixture_difficulty_next"] >= 4) &
            (df_alert["total_points"] <= df_alert["total_points"].median())
        ].copy()
        overhyped["AI Hint"] = "High owned + hard fixture + low points → consider selling / benching"
        overhyped = overhyped.sort_values(by=["selected_by_percent", "fixture_difficulty_next"], ascending=[False, False]).head(10)

        leverage = df_alert[
            (df_alert["fixture_difficulty_next"] <= 3) &
            (df_alert["total_points"] >= df_alert["total_points"].median())
        ].copy()
        leverage["points_per_million"] = (leverage["total_points"] / (leverage["price"] + 0.1)).round(2)
        leverage["is_differential"] = leverage["selected_by_percent"] < 10
        leverage["momentum_score"] = (
            (6 - leverage["fixture_difficulty_next"]) * 0.4 +
            (leverage["points_per_million"] / leverage["points_per_million"].max()) * 0.3 +
            (1 - (leverage["selected_by_percent"] / 100)) * 0.3
        ).round(3)

        def generate_leverage_hint(row):
            hints = []
            if row["is_differential"]:
                hints.append("Low-owned differential")
            if row["points_per_million"] > leverage["points_per_million"].median():
                hints.append("High value per £")
            if row["fixture_difficulty_next"] <= 2:
                hints.append("Excellent upcoming fixture")
            if row["momentum_score"] > 0.7:
                hints.append("Strong form trend")
            if not hints:
                hints.append("Good alternative captain pick")
            return " + ".join(hints)

        leverage["AI Hint"] = leverage.apply(generate_leverage_hint, axis=1)
        leverage = leverage.sort_values(by=["total_points"], ascending=False).head(10)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### Manager trap candidates")
            if overhyped.empty:
                st.info("No obvious traps in popular players this week.")
            else:
                st.dataframe(
                    overhyped[["name","team","fixture_difficulty_next","selected_by_percent","total_points","AI Hint"]]
                    .rename(columns={"name":"Player","team":"Team","fixture_difficulty_next":"Next GW Diff","total_points":"Total Points","selected_by_percent":"% Selected"}),
                    hide_index=True,
                    use_container_width=True
                )
        with c2:
            st.markdown("###### Manager leverage picks")
            if leverage.empty:
                st.info("No obvious leverage picks in manager pool.")
            else:
                st.dataframe(
                    leverage[["name","team","fixture_difficulty_next","selected_by_percent","total_points","AI Hint"]]
                    .rename(columns={"name":"Player","team":"Team","fixture_difficulty_next":"Next GW Diff","total_points":"Total Points","selected_by_percent":"% Selected"}),
                    hide_index=True,
                    use_container_width=True
                )


# ---------------------------------------------------------
# 7. FIXTURE PLANNER (Enhanced — Scout/Hub-style ticker)
# ---------------------------------------------------------
with tabs[7]:
    st.markdown("#### Fixture Planner — Upcoming Difficulty by Team")
    st.caption("Shows the next 5 fixtures per team, color-coded by difficulty (1=Easy → 5=Hard).")

    # Build matrix of next 5 GWs
    fixture_df = pd.DataFrame(fixtures)
    upcoming = fixture_df[fixture_df["event"].between(current_gw, current_gw + 4)]
    team_names = list(teams.values())
    data = []
    for team_id, name in teams.items():
        row = {"Team": name}
        team_fixtures = upcoming[
            (upcoming["team_h"] == team_id) | (upcoming["team_a"] == team_id)
        ]
        for _, f in team_fixtures.iterrows():
            gw = f["event"]
            is_home = f["team_h"] == team_id
            opp_id = f["team_a"] if is_home else f["team_h"]
            opp_name = teams[opp_id]
            diff = f["team_h_difficulty"] if is_home else f["team_a_difficulty"]
            loc = "(H)" if is_home else "(A)"
            row[f"GW{gw}"] = f"{opp_name} {loc} — {diff}"
        data.append(row)
    df_ticker = pd.DataFrame(data).fillna("-")

    st.dataframe(
        df_ticker.sort_values("Team").set_index("Team"),
        use_container_width=True,
        hide_index=False
    )

    st.info("Tip: Lower numbers = easier fixtures. Use this to plan transfers and captain picks ahead.")


# ---------------------------------------------------------
# 8. TEMPLATE vs DIFFERENTIAL (LiveFPL-style)
# ---------------------------------------------------------
with tabs[8]:
    st.markdown("#### Template vs Differential — ownership-driven tiers")
    st.caption("Approximate LiveFPL-style effective ownership using your player pool.")
    df_tmp = players_raw_df.copy()
    df_tmp = safe_numeric_col(df_tmp, "selected_by_percent")
    df_tmp = df_tmp.rename(columns={"name": "Player", "team": "Team"})
    template = df_tmp[df_tmp["selected_by_percent"] >= 30].sort_values(by="selected_by_percent", ascending=False).head(15)
    popular = df_tmp[(df_tmp["selected_by_percent"] >= 15) & (df_tmp["selected_by_percent"] < 30)].sort_values(by="selected_by_percent", ascending=False).head(15)
    differential = df_tmp[df_tmp["selected_by_percent"] < 15].sort_values(by="selected_by_percent", ascending=False).head(15)

    st.markdown("##### Template (≥30% owned)")
    st.dataframe(template[["Player","Team","selected_by_percent","price","status"]]
                 .rename(columns={"selected_by_percent":"% Selected"}), hide_index=True, use_container_width=True)

    st.markdown("##### Differentials (<15% owned)")
    st.dataframe(differential[["Player","Team","selected_by_percent","price","status"]]
                 .rename(columns={"selected_by_percent":"% Selected"}), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# 9. PREDICTIONS — top performers for next 3 GWs
# ---------------------------------------------------------
with tabs[9]:
    st.markdown("#### AI Predictions — Potential Top Performers")
    st.caption("Forecast of high-impact players based on recent form, fixture ease, and momentum.")

    sub1, sub2, sub3 = st.tabs(["Current GW", "Next GW", "GW + 2 Ahead"])

    def compute_form(df):
        """Approximate last 3-match form if not present."""
        if "form" in df.columns:
            return pd.to_numeric(df["form"], errors="coerce").fillna(0)
        if "total_points" in df.columns:
            return df["total_points"].clip(upper=20)
        return pd.Series([0]*len(df))

    def predict_performance(df, gw_offset=0):
        decay = [1.0, 0.85, 0.7][gw_offset]
        df = df.copy()
        df["form_score"] = compute_form(df)
        df["fixture_diff"] = df["team"].map(compute_fixture_difficulty(fixtures, teams)).fillna(3)
        df["momentum"] = df.get("transfers_in_event", 0) - df.get("transfers_out_event", 0)
        df["momentum_scaled"] = (df["momentum"] / (abs(df["momentum"]).max() + 1)) * 5
        df["potential_index"] = (
            (df["form_score"] * 0.5)
            + ((6 - df["fixture_diff"]) * 1.2)
            + (df["momentum_scaled"] * 0.3)
        ) * decay
        df["potential_index"] = df["potential_index"].round(2)
        df = df.rename(columns={"name": "Player", "team": "Team"})
        return df.sort_values(by="potential_index", ascending=False).head(10)

    # Current GW
with sub1:
    curr_pred = predict_performance(players_raw_df, 0)
    fig = px.bar(
        curr_pred, x="Player", y="potential_index", text="potential_index",
        color_discrete_sequence=["#00e676"], title=f"Potential Top 10 — Gameweek {current_gw}"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis=dict(tickangle=-45, title=None),
        yaxis=dict(title=None),
        margin=dict(l=8, r=8, t=48, b=65)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        curr_pred[["Player","Team","price","form_score","fixture_diff","momentum","potential_index","status"]]
        .rename(columns={"price":"Price  ","fixture_diff":"Fixture Diff"}),
        hide_index=True, use_container_width=True
    )

    # Next GW
    with sub2:
        next_pred = predict_performance(players_raw_df, 1)
        nxt_gw = (current_gw or analysis["gw"]) + 1
        fig = px.bar(
            next_pred, x="Player", y="potential_index", text="potential_index",
            color_discrete_sequence=["#33ccff"], title=f"Forecasted Top 10 — Gameweek {nxt_gw}"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis=dict(tickangle=-45, title=None),
            yaxis=dict(title=None),
            margin=dict(l=8, r=8, t=48, b=65)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            next_pred[["Player","Team","price","form_score","fixture_diff","momentum","potential_index","status"]]
            .rename(columns={"price":"Price  ","fixture_diff":"Fixture Diff"}),
            hide_index=True, use_container_width=True
        )

    # GW + 2
    with sub3:
        future_pred = predict_performance(players_raw_df, 2)
        fut_gw = (current_gw or analysis["gw"]) + 2
        fig = px.bar(
            future_pred, x="Player", y="potential_index", text="potential_index",
            color_discrete_sequence=["#9966ff"], title=f"Early Outlook — Gameweek {fut_gw}"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis=dict(tickangle=-45, title=None),
            yaxis=dict(title=None),
            margin=dict(l=8, r=8, t=48, b=65)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            future_pred[["Player","Team","price","form_score","fixture_diff","momentum","potential_index","status"]]
            .rename(columns={"price":"Price","fixture_diff":"Fixture Diff"}),
            hide_index=True, use_container_width=True
        )


# ---------------------------------------------------------
# 10. TEAM COMPARISON — Compact, User-Friendly Layout
# ---------------------------------------------------------
with tabs[10]:
    st.markdown("#### Team Comparison — Market Dynamics, Value & Performance")
    st.caption("Interactive team-level comparison with transfer trends, value metrics, and AI-derived performance indices.")

    base = players_raw_df.copy()
    base = safe_numeric_col(base, "total_points")
    base = safe_numeric_col(base, "selected_by_percent")
    base = safe_numeric_col(base, "transfers_in_event")
    base = safe_numeric_col(base, "transfers_out_event")
    base = safe_numeric_col(base, "price")
    base = safe_numeric_col(base, "minutes")

    # Derived proxies for attacking & defensive capability
    base["offensive_score"] = (
        base["total_points"] * 0.6 +
        base["transfers_in_event"] * 0.001 +
        base["selected_by_percent"] * 0.4
    )
    base["defensive_score"] = (
        base["total_points"] * 0.4 +
        (100 - base["selected_by_percent"]) * 0.3 +
        (base["minutes"] / (base["price"] + 0.1)) * 0.05
    )

    team_diff_map = compute_fixture_difficulty(fixtures, teams)
    base["fixture_difficulty_next"] = base["team"].map(team_diff_map).fillna(3)

    # Aggregate to team level
    tm = base.groupby("team", as_index=False).agg({
        "total_points": "mean",
        "transfers_in_event": "sum",
        "transfers_out_event": "sum",
        "selected_by_percent": "mean",
        "offensive_score": "mean",
        "defensive_score": "mean",
        "fixture_difficulty_next": "mean",
        "price": "mean",
        "minutes": "mean"
    })

    tm["net_transfers"] = tm["transfers_in_event"] - tm["transfers_out_event"]
    tm["balance_index"] = (tm["offensive_score"] - tm["defensive_score"]).abs().round(2)
    tm["value_index"] = (tm["total_points"] / (tm["price"] + 0.1)).round(2)
    tm = tm.sort_values(by="total_points", ascending=False)

    # -----------------------------------------------------
    # Compact Chart Selector
    # -----------------------------------------------------
    st.markdown("##### Team Visual Comparison")
    chart_option = st.radio(
        "Select Chart to View:",
        [
            "Transfer Momentum vs Average Points",
            "Offensive vs Defensive Strength Map",
            "Team Balance Radar",
        ],
        horizontal=True,
    )

    if chart_option == "Transfer Momentum vs Average Points":
        fig_net = go.Figure()
        fig_net.add_trace(go.Bar(
            x=tm["team"], y=tm["net_transfers"],
            name="Net Transfers", marker_color="#00e676"
        ))
        fig_net.add_trace(go.Scatter(
            x=tm["team"], y=tm["total_points"],
            name="Average Points", mode="lines+markers",
            line=dict(color="#33ccff", width=2.5), yaxis="y2"
        ))
        fig_net.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=60),
            title="Transfer Momentum vs Average Points",
            xaxis=dict(tickangle=-35, title="Team", tickfont=dict(size=10)),
            yaxis=dict(title="Net Transfers", side="left", showgrid=False),
            yaxis2=dict(title="Average Points", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_net, use_container_width=True)

    elif chart_option == "Offensive vs Defensive Strength Map":
        fig_strength = px.scatter(
            tm,
            x="offensive_score",
            y="defensive_score",
            size="total_points",
            color="fixture_difficulty_next",
            hover_name="team",
            color_continuous_scale="RdYlGn_r",
            title="Offensive vs Defensive Strength (bubble=size=Average Points, color=Fixture Diff)",
            size_max=25,
        )
        fig_strength.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=60, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Offensive Strength",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Defensive Strength",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                tickfont=dict(size=10)
            ),
            legend=dict(
                title="Fixture Difficulty",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=9)
            ),
            title_font=dict(size=13)
        )
        st.plotly_chart(fig_strength, use_container_width=True)

    elif chart_option == "Team Balance Radar":
        selected_teams = st.multiselect(
            "Select Teams to Compare:",
            options=tm["team"].tolist(),
            default=tm["team"].head(3).tolist(),
            max_selections=3
        )

        metrics = ["offensive_score", "defensive_score", "value_index"]
        fig_radar = go.Figure()

        for team_name in selected_teams:
            row = tm[tm["team"] == team_name].iloc[0]
            values = [row[m] for m in metrics]
            values += [values[0]]
            theta = metrics + [metrics[0]]

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=team_name,
                opacity=0.75
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, tm[metrics].max().max()], tickfont=dict(size=9))
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            title="Team Comparison Radar (Offense, Defense, Value)",
            legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<hr class='glow-line'>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Detailed Table
    # -----------------------------------------------------
    st.markdown("##### Detailed Team Statistics")
    st.caption("Aggregated market, performance, and AI-derived indices per team.")

    st.dataframe(
        tm[[
            "team", "total_points", "net_transfers",
            "transfers_in_event", "transfers_out_event",
            "offensive_score", "defensive_score", "balance_index",
            "fixture_difficulty_next", "selected_by_percent",
            "value_index", "price"#, "minutes"
        ]].rename(columns={
            "team": "Team",
            "total_points": "Average Points",
            "net_transfers":"Net Transfers",
            "transfers_in_event":"Transfers In",
            "transfers_out_event":"Transfers Out",
            "offensive_score":"Offensive Score",
            "defensive_score":"Defensive Score",
            "balance_index": "Next GW Diff",
            "selected_by_percent": "% Selected",
            "fixture_difficulty_next": "Next GW Difficulty",
            "price": "Average Price "
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Tip: Toggle between visuals for a cleaner, focused comparison. "
        "Use the radar chart to compare 2–3 teams directly on balance and value."
    )

# ---------------------------------------------------------
# 🧠 UNIVERSAL AI COPILOT — Gemini + AI Analysis Integration
# ---------------------------------------------------------
with tabs[11]:
    st.markdown("#### Ask FPL Advisor")
    #st.caption("Now enhanced with your AI analysis (price movement, risers/fallers, etc).")

    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = "AIzaSyA8oIR5FNUQDNu7cpb-47sZmxAW5P4CRGQ"

    GEMINI_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    @st.cache_resource(ttl=1800)
    def fetch_live_fpl_data():
        base = "https://fantasy.premierleague.com/api/"
        bootstrap = requests.get(base + "bootstrap-static/").json()
        fixtures = requests.get(base + "fixtures/").json()
        players = pd.DataFrame(bootstrap["elements"])
        teams = pd.DataFrame(bootstrap["teams"])
        teams_dict = dict(zip(teams["id"], teams["name"]))
        fixtures_df = pd.DataFrame(fixtures)
        players["team_name"] = players["team"].map(teams_dict)
        players["now_cost"] = players["now_cost"] / 10
        return players, teams_dict, fixtures_df

    # Include AI analysis (SQLite)
    ai_analysis = load_latest_report(DB_PATH)
    ai_risers = to_df(ai_analysis.get("risers", []))
    ai_fallers = to_df(ai_analysis.get("fallers", []))
    ai_bandwagons = to_df(ai_analysis.get("bandwagons", []))
    ai_watchlist = to_df(ai_analysis.get("watchlist", []))

    # Build “next fixture” map
    def build_next_fixture_map(fixtures_df, teams_dict):
        future = fixtures_df[(fixtures_df["event"].notnull()) & (~fixtures_df["finished"])]
        future = future.sort_values("event")
        next_map = {}
        for _, f in future.iterrows():
            gw = int(f["event"])
            th, ta = f["team_h"], f["team_a"]
            th_name, ta_name = teams_dict.get(th, f"Team {th}"), teams_dict.get(ta, f"Team {ta}")
            if th_name not in next_map:
                next_map[th_name] = {"opponent": ta_name, "gw": gw, "home_away": "H", "difficulty": int(f.get("team_h_difficulty", 3))}
            if ta_name not in next_map:
                next_map[ta_name] = {"opponent": th_name, "gw": gw, "home_away": "A", "difficulty": int(f.get("team_a_difficulty", 3))}
        return next_map

    def gemini_chat(prompt, players_df, teams_dict, fixtures_df, next_fixture_map, history, ai_risers, ai_fallers, ai_bandwagons):
        keep_cols = [
            "web_name", "team_name", "now_cost", "total_points",
            "selected_by_percent", "transfers_in_event", "transfers_out_event",
            "form", "ep_next", "status"
        ]
        players_df = players_df[keep_cols].rename(
            columns={"web_name": "Player", "team_name": "Team", "now_cost": "Price"}
        )

        chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])

        # NEW: Include AI Analysis data (JSON-style)
        ai_summary = {
            "price_risers": ai_risers.to_dict(orient="records") if not ai_risers.empty else [],
            "price_fallers": ai_fallers.to_dict(orient="records") if not ai_fallers.empty else [],
            "bandwagons": ai_bandwagons.to_dict(orient="records") if not ai_bandwagons.empty else [],
        }

        text_prompt = f"""
You are an expert Fantasy Premier League (FPL) advisor.

You have:
- Live official FPL data (players, teams, fixtures)
- AI-derived price movement predictions and analysis (risers, fallers, bandwagons)
- Knowledge of FPL strategy, trends, and schedule patterns

Your job:
- Use both datasets (AI + FPL) to provide actionable FPL insights
- When asked about price changes, use the AI analysis to identify top risers/fallers

DATA SNAPSHOT:
- AI_ANALYSIS: {json.dumps(ai_summary)[:3000]}  # truncated to keep prompt compact
- NEXT_FIXTURE_MAP: {json.dumps(next_fixture_map)[:1500]}
- PLAYER_SAMPLE: {players_df.head(20).to_dict(orient='records')}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{prompt}
"""

        payload = {"contents": [{"parts": [{"text": text_prompt}]}]}
        headers = {"Content-Type": "application/json"}

        resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response from Gemini.")
        )

    # Session memory
    if "advisor_chat" not in st.session_state:
        st.session_state["advisor_chat"] = []

    for msg in st.session_state["advisor_chat"]:
        color = "#00e676" if msg["role"] == "user" else "#bfbfff"
        st.markdown(f"<div style='color:{color}'><b>{msg['role'].capitalize()}:</b> {msg['content']}</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask anything (e.g. 'Who will rise in price tonight?')")
    if question:
        st.session_state["advisor_chat"].append({"role": "user", "content": question})
        with st.spinner("Analyzing with AI..."):
            players_df, teams_dict, fixtures_df = fetch_live_fpl_data()
            next_fixture_map = build_next_fixture_map(fixtures_df, teams_dict)
            answer = gemini_chat(
                question,
                players_df,
                teams_dict,
                fixtures_df,
                next_fixture_map,
                st.session_state["advisor_chat"],
                ai_risers,
                ai_fallers,
                ai_bandwagons
            )
        st.session_state["advisor_chat"].append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state["advisor_chat"] = []
        st.rerun()
