# cricket_dashboard_v13.py
# Run: streamlit run cricket_dashboard_v13.py

import ast
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Cricket Analytics — Players & Matches (2018–2025)",
    layout="wide",
    page_icon="🏏",
    initial_sidebar_state="expanded",
)

# =============================
# 1) Data load
# =============================
with st.sidebar:
    st.header("📁 Data")
    matches_file = st.file_uploader("Upload matches CSV", type=["csv"], key="matches_up")
    deliveries_file = st.file_uploader("Upload deliveries CSV", type=["csv"], key="delivs_up")

def _read_csv_fallback(uploaded, fallback_path):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    try:
        return pd.read_csv(fallback_path)
    except Exception:
        return None

matches = _read_csv_fallback(matches_file, "matches_2018_2025.csv")
deliveries = _read_csv_fallback(deliveries_file, "deliveries_2018_2025.csv")

if matches is None or deliveries is None:
    st.warning("Please upload both CSVs or place `matches_2018_2025.csv` and `deliveries_2018_2025.csv` next to this script.")
    st.stop()

# =============================
# 2) Schema assumptions
# =============================
COL_MATCH_ID_M = "match_id"
COL_DATE = "date"
COL_VENUE = "venue"
COL_CITY = "city" if "city" in matches.columns else None
COL_TOSS_WINNER = "toss_winner" if "toss_winner" in matches.columns else None
COL_TOSS_DEC = "toss_decision" if "toss_decision" in matches.columns else None
COL_WINNER = "winner" if "winner" in matches.columns else None
COL_RESULT = "result" if "result" in matches.columns else None
COL_TEAMS = "teams"
COL_SEASON = "season" if "season" in matches.columns else None

COL_MATCH_ID_D = "match_id"
COL_BATTING_TEAM = "batting_team"
COL_BOWLING_TEAM = None  # derive
COL_BATTER = "batter"
COL_BOWLER = "bowler"
COL_OVER = "over"
COL_TOTAL_RUNS = "runs_total"
COL_BAT_RUNS = "runs_batter"
COL_DISMISSAL = "wicket_type" if "wicket_type" in deliveries.columns else None
COL_PLAYER_DISMISSED = "player_out" if "player_out" in deliveries.columns else None
COL_IS_WICKET = None  # derive
COL_INNING = "innings" if "innings" in deliveries.columns else ("inning" if "inning" in deliveries.columns else None)

# =============================
# 3) Utilities
# =============================
def df_download_button(df: pd.DataFrame, label: str, filename: str):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def parse_teams(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clean_str(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"\s+", " ", regex=True)
    return s2.where(~s2.str.lower().isin(["", "nan", "none", "na"]), np.nan)

TEAM_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "rising pune supergiant": "Rising Pune Supergiant",
    "rising pune supergiants": "Rising Pune Supergiant",
}

# Default home cities (can be overridden in UI)
TEAM_HOME_CITIES_DEFAULT = {
    "Chennai Super Kings": ["Chennai"],
    "Mumbai Indians": ["Mumbai"],
    "Royal Challengers Bengaluru": ["Bengaluru", "Bangalore"],
    "Kolkata Knight Riders": ["Kolkata"],
    "Sunrisers Hyderabad": ["Hyderabad"],
    "Rajasthan Royals": ["Jaipur"],
    "Delhi Capitals": ["Delhi"],
    "Punjab Kings": ["Mohali", "Dharamsala", "Dharamshala"],
    "Lucknow Super Giants": ["Lucknow"],
    "Gujarat Titans": ["Ahmedabad"],
    "Rising Pune Supergiant": ["Pune"],
}

# Session overrides for home cities
if "home_city_overrides" not in st.session_state:
    st.session_state.home_city_overrides = {}

def get_home_cities(team_name: str):
    return st.session_state.home_city_overrides.get(team_name, TEAM_HOME_CITIES_DEFAULT.get(team_name, []))

def set_home_cities(team_name: str, cities: list[str]):
    st.session_state.home_city_overrides[team_name] = cities

def canon_team(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    key = str(x).strip().lower()
    return TEAM_ALIASES.get(key, str(x).strip())

def canon_venue(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\(.*?\)", "", s).strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    base = parts[0] if parts else s
    base = re.sub(r"\s+", " ", base)
    if "arjun jaitley" in base.lower():
        base = "Arun Jaitley Stadium"
    base = base.title()
    base = base.replace("M A ", "M. A. ").replace("M Chinnaswamy", "M Chinnaswamy")
    return base

def venue_city_from_venue(x: str):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[-1].title()
    return None

def safe_ratio(num, den):
    return (num / den) if den else 0.0

# =============================
# 4) Derivations & cleaning
# =============================
# Season from date if missing
if COL_SEASON is None and COL_DATE in matches.columns:
    matches[COL_DATE] = pd.to_datetime(matches[COL_DATE], errors="coerce")
    matches["season"] = matches[COL_DATE].dt.year
    COL_SEASON = "season"
else:
    if COL_DATE in matches.columns:
        matches[COL_DATE] = pd.to_datetime(matches[COL_DATE], errors="coerce")

# Canonical venue + derived city
if COL_VENUE in matches.columns:
    matches["venue_canon"] = clean_str(matches[COL_VENUE]).apply(canon_venue)
    if COL_CITY is None:
        matches["city_derived"] = matches[COL_VENUE].apply(venue_city_from_venue)
    else:
        matches["city_derived"] = matches[COL_CITY]

# Derive team1_canon & team2_canon from COL_TEAMS (if present)
if COL_TEAMS in matches.columns:
    teams_parsed = matches[COL_TEAMS].apply(parse_teams)
    matches["team1_canon"] = teams_parsed.apply(lambda t: canon_team(t[0]) if isinstance(t, (list, tuple)) and len(t) == 2 else np.nan)
    matches["team2_canon"] = teams_parsed.apply(lambda t: canon_team(t[1]) if isinstance(t, (list, tuple)) and len(t) == 2 else np.nan)

# bowling_team from matches.teams
if COL_TEAMS in matches.columns:
    teams_map = matches.set_index(COL_MATCH_ID_M)[COL_TEAMS].apply(parse_teams)
    def other_team(match_id, batting_team):
        try:
            ts = teams_map.loc[match_id]
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                bt = canon_team(batting_team)
                a = canon_team(ts[0]); b = canon_team(ts[1])
                if bt == a: return b
                if bt == b: return a
        except Exception:
            return np.nan
        return np.nan
    deliveries["bowling_team"] = [other_team(mid, bt) for mid, bt in zip(deliveries[COL_MATCH_ID_D], deliveries[COL_BATTING_TEAM])]
    COL_BOWLING_TEAM = "bowling_team"
else:
    deliveries["bowling_team"] = np.nan
    COL_BOWLING_TEAM = "bowling_team"

# is_wicket
if COL_IS_WICKET is None:
    cols = [c for c in [COL_DISMISSAL, COL_PLAYER_DISMISSED] if c]
    deliveries["is_wicket"] = deliveries[cols].notna().any(axis=1).astype(int) if cols else 0
    COL_IS_WICKET = "is_wicket"

# numerics
deliveries[COL_OVER] = to_num(deliveries[COL_OVER])
deliveries[COL_TOTAL_RUNS] = to_num(deliveries[COL_TOTAL_RUNS])
deliveries[COL_IS_WICKET] = to_num(deliveries[COL_IS_WICKET]).fillna(0).astype(int)

# attach season to deliveries
if COL_SEASON and COL_SEASON in matches.columns:
    deliveries = deliveries.merge(
        matches[[COL_MATCH_ID_M, COL_SEASON]],
        left_on=COL_MATCH_ID_D, right_on=COL_MATCH_ID_M, how="left", suffixes=("", "_m")
    )

# canon team on deliveries
deliveries["bat_team_canon"] = clean_str(deliveries[COL_BATTING_TEAM]).apply(canon_team)
deliveries["bowl_team_canon"] = clean_str(deliveries[COL_BOWLING_TEAM]).apply(canon_team)

# =============================
# 5) Sidebar filters (define phase BEFORE applying filter)
# =============================
st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")

teams_list = sorted(pd.unique(pd.concat([deliveries["bat_team_canon"], deliveries["bowl_team_canon"]], ignore_index=True).dropna()))
venues_list = sorted(pd.unique(matches["venue_canon"].dropna())) if "venue_canon" in matches.columns else []
seasons_list = sorted(pd.unique(deliveries[COL_SEASON].dropna())) if (COL_SEASON and COL_SEASON in deliveries.columns) else []

sel_season = st.sidebar.multiselect("Season", seasons_list, default=seasons_list if seasons_list else None)
sel_team = st.sidebar.multiselect("Team", teams_list, default=None)
sel_venue = st.sidebar.multiselect("Venue", venues_list, default=None)

# Player list depends on team & season
_d_base = deliveries.copy()
if sel_season and (COL_SEASON and COL_SEASON in _d_base.columns):
    _d_base = _d_base[_d_base[COL_SEASON].isin(sel_season)]
if sel_team:
    bat_roster = _d_base.loc[_d_base["bat_team_canon"].isin(sel_team), COL_BATTER]
    bowl_roster = _d_base.loc[_d_base["bowl_team_canon"].isin(sel_team), COL_BOWLER]
    players_list = sorted(pd.unique(pd.concat([bat_roster, bowl_roster], ignore_index=True).dropna()))
else:
    players_list = sorted(pd.unique(pd.concat([_d_base[COL_BATTER], _d_base[COL_BOWLER]], ignore_index=True).dropna()))

sel_player = st.sidebar.multiselect("Player", players_list, default=None)

# Define phase now
st.sidebar.markdown("**Phase (overs)**")
phase = st.sidebar.radio(
    label="",
    options=["All", "Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"],
    horizontal=True,
    index=0,
)

# =============================
# 6) Apply filters to deliveries DF
# =============================
df = deliveries.copy()
if sel_season and (COL_SEASON and COL_SEASON in df.columns):
    df = df[df[COL_SEASON].isin(sel_season)]
if sel_team:
    df = df[(df["bat_team_canon"].isin(sel_team)) | (df["bowl_team_canon"].isin(sel_team))]

# Join venue_canon for delivery filtering (avoid duplicate column insertion)
if sel_venue and "venue_canon" in matches.columns:
    if "venue_canon" not in df.columns:
        df = df.merge(matches[[COL_MATCH_ID_M, "venue_canon"]],
                      left_on=COL_MATCH_ID_D, right_on=COL_MATCH_ID_M, how="left")
    df = df[df["venue_canon"].isin(sel_venue)]
else:
    df = df.drop(columns=["venue_canon"], errors="ignore")

if sel_player:
    df = df[(df[COL_BATTER].isin(sel_player)) | (df[COL_BOWLER].isin(sel_player))]

def _apply_phase_filter(_df):
    if phase == "All" or COL_OVER is None:
        return _df
    if "Powerplay" in phase:
        return _df[(_df[COL_OVER] >= 1) & (_df[COL_OVER] <= 6)]
    if "Middle" in phase:
        return _df[(_df[COL_OVER] >= 7) & (_df[COL_OVER] <= 15)]
    if "Death" in phase:
        return _df[(_df[COL_OVER] >= 16)]
    return _df

df = _apply_phase_filter(df)

# =============================
# 7) UI helpers
# =============================
def metric_card(title: str, value, help_text: str | None = None):
    st.markdown(
        f"""
        <div style="background:#0f172a0d;border:1px solid #e5e7eb;border-radius:16px;padding:14px 16px;margin-bottom:8px">
            <div style="font-size:12px;color:#64748b">{title}</div>
            <div style="font-size:24px;font-weight:700;margin-top:4px">{value}</div>
            {"<div style='font-size:12px;color:#94a3b8;margin-top:6px'>"+help_text+"</div>" if help_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================
# 8) Tabs — segregated
# =============================
tab_team, tab_overview, tab_batting, tab_bowling, tab_matches, tab_players, tab_venues, tab_team_venues = st.tabs(
    ["Team Insights", "Overview", "Batting", "Bowling", "Matches", "Players", "Venues", "Team–Venues"]
)

# -----------------------------
# Team Insights
# -----------------------------
with tab_team:
    st.subheader("Team Insights — Win% • Strong Venues • Home vs Away • Opponents • Trend • H2H • Players")

    # Pick one team (defaults to first selection or first team)
    team_pick = None
    if sel_team and len(sel_team) >= 1:
        team_pick = sel_team[0]
    else:
        if teams_list:
            team_pick = teams_list[0]

    team = team_pick
    if not team:
        st.info("Select a team in the left sidebar to view insights.")
    else:
        # -- UI: editable home cities for this team
        with st.expander("⚙️ Edit home cities for this team"):
            all_cities = pd.Series(
                matches["city_derived"] if "city_derived" in matches.columns else []
            ).dropna().astype(str).str.title().unique().tolist()
            prefill = get_home_cities(team)
            edited = st.multiselect("Select home cities considered 'home' for this team",
                                    options=sorted(set(all_cities + prefill)),
                                    default=prefill,
                                    key=f"home_cities_{team}")
            st.caption("These selections affect the Home vs Away split in this tab.")
            set_home_cities(team, edited)

        # Filter matches by sidebar season & venue
        m = matches.copy()
        if COL_SEASON in m.columns and sel_season:
            m = m[m[COL_SEASON].isin(sel_season)]
        if sel_venue and "venue_canon" in m.columns:
            m = m[m["venue_canon"].isin(sel_venue)]

        # winner_canon
        if COL_WINNER and COL_WINNER in m.columns:
            m["winner_canon"] = clean_str(m[COL_WINNER]).apply(canon_team)
        else:
            m["winner_canon"] = np.nan

        # ensure team1_canon/team2_canon exist (already derived if COL_TEAMS exists)
        if "team1_canon" not in m.columns or "team2_canon" not in m.columns:
            if COL_TEAMS in m.columns:
                teams_parsed = m[COL_TEAMS].apply(parse_teams)
                m["team1_canon"] = teams_parsed.apply(lambda t: canon_team(t[0]) if isinstance(t, (list, tuple)) and len(t)==2 else np.nan)
                m["team2_canon"] = teams_parsed.apply(lambda t: canon_team(t[1]) if isinstance(t, (list, tuple)) and len(t)==2 else np.nan)
            else:
                m["team1_canon"] = np.nan
                m["team2_canon"] = np.nan

        def played_by_team(row, team_name):
            a, b = row.get("team1_canon"), row.get("team2_canon")
            return (a == team_name) or (b == team_name)

        def opponent_for(row, team_name):
            a, b = row.get("team1_canon"), row.get("team2_canon")
            if a == team_name: return b
            if b == team_name: return a
            return np.nan

        m_team = m[m.apply(lambda r: played_by_team(r, team), axis=1)].copy()

        # Overall win%
        played = len(m_team)
        wins = int((m_team["winner_canon"] == team).sum()) if played else 0
        win_pct = (wins / played * 100) if played else 0.0

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Matches Played", f"{played}")
        with c2: metric_card("Wins", f"{wins}")
        with c3: metric_card("Win %", f"{win_pct:.1f}%")

        # Venue strength
        if played and "venue_canon" in m_team.columns:
            venue_stats = (
                m_team
                .groupby("venue_canon", as_index=False)
                .agg(
                    matches=(COL_MATCH_ID_M, "count"),
                    wins=("winner_canon", lambda s: (s == team).sum()),
                )
            )
            venue_stats["win_%"] = (venue_stats["wins"] / venue_stats["matches"] * 100).round(1)
            min_games = st.slider("Minimum matches at venue", 1, 10, 3, key="min_games_team_venue")
            vt_show = venue_stats[venue_stats["matches"] >= min_games].sort_values(["win_%","matches"], ascending=[False, False])
            st.markdown("**🏟️ Best Venues (by Win%)**")
            st.dataframe(vt_show, use_container_width=True)
            df_download_button(vt_show, "⬇️ Download venues (this team)", f"{team.replace(' ', '_').lower()}_venues_winpct.csv")
            if len(vt_show):
                figv = px.bar(vt_show.head(12), x="venue_canon", y="win_%", hover_data=["matches","wins"], title="Top Venues by Win%")
                st.plotly_chart(figv, use_container_width=True)

        # Home vs Away split using city data + editable mapping
        city_col_to_use = None
        if COL_CITY and COL_CITY in m_team.columns and m_team[COL_CITY].notna().any():
            city_col_to_use = COL_CITY
        elif "city_derived" in m_team.columns:
            city_col_to_use = "city_derived"

        if city_col_to_use:
            city_series = clean_str(m_team[city_col_to_use]).str.title()
            home_cities = set([c.title() for c in get_home_cities(team)])
            m_team["_is_home"] = city_series.isin(home_cities)
            m_team["_win"] = (m_team["winner_canon"] == team).astype(int)

            home = m_team[m_team["_is_home"] == True]
            away = m_team[m_team["_is_home"] == False]
            home_played, home_wins = len(home), int(home["_win"].sum())
            away_played, away_wins = len(away), int(away["_win"].sum())
            home_wp = (home_wins / home_played * 100) if home_played else 0.0
            away_wp = (away_wins / away_played * 100) if away_played else 0.0

            c4, c5 = st.columns(2)
            with c4:
                metric_card("Home: W / Played", f"{home_wins} / {home_played}", f"Win% {home_wp:.1f}%")
            with c5:
                metric_card("Away: W / Played", f"{away_wins} / {away_played}", f"Win% {away_wp:.1f}%")

            ha_df = pd.DataFrame({"type":["Home","Away"],"played":[home_played, away_played],"wins":[home_wins, away_wins],"win_%":[round(home_wp,1),round(away_wp,1)]})
            figha = px.bar(ha_df, x="type", y="wins", text="win_%", title="Home vs Away Wins")
            figha.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(figha, use_container_width=True)

            # --- Avg runs scored per match (Home vs Away) ---
            d_aug = deliveries[[COL_MATCH_ID_D, COL_TOTAL_RUNS, "bat_team_canon"]].copy()
            d_aug = d_aug.merge(
                m_team[[COL_MATCH_ID_M, "_is_home"]],
                left_on=COL_MATCH_ID_D,
                right_on=COL_MATCH_ID_M,
                how="inner",
            )
            runs_home = d_aug[(d_aug["_is_home"] == True) & (d_aug["bat_team_canon"] == team)][COL_TOTAL_RUNS].sum()
            runs_away = d_aug[(d_aug["_is_home"] == False) & (d_aug["bat_team_canon"] == team)][COL_TOTAL_RUNS].sum()

            avg_runs_home = (runs_home / home_played) if home_played else 0.0
            avg_runs_away = (runs_away / away_played) if away_played else 0.0

            c_avg1, c_avg2 = st.columns(2)
            with c_avg1:
                metric_card("Avg Runs Scored — Home", f"{avg_runs_home:.1f}", f"Total {int(runs_home)} across {home_played} matches")
            with c_avg2:
                metric_card("Avg Runs Scored — Away", f"{avg_runs_away:.1f}", f"Total {int(runs_away)} across {away_played} matches")

        else:
            st.info("Home/Away split needs a 'city' column or a derivable city from venue.")

        # Opponent breakdown • Season trend • H2H
        st.markdown("---")
        st.subheader("Opponent Breakdown • Season Trend • Head-to-Head")

        if len(m_team):
            m_team["opponent"] = m_team.apply(lambda r: opponent_for(r, team), axis=1)
            m_team["win_flag"] = (m_team["winner_canon"] == team).astype(int)

            # per-opponent
            opp_tbl = (
                m_team
                .groupby("opponent", as_index=False)
                .agg(played=(COL_MATCH_ID_M, "count"), wins=("win_flag", "sum"))
            )
            opp_tbl["win_%"] = (opp_tbl["wins"] / opp_tbl["played"] * 100).round(1)
            opp_tbl = opp_tbl.sort_values(["win_%","played"], ascending=[False, False])

            c_opp1, c_opp2 = st.columns([1,1])
            with c_opp1:
                st.markdown("**Per-opponent Win% (filtered)**")
                st.dataframe(opp_tbl, use_container_width=True)
                df_download_button(opp_tbl, "⬇️ Download per-opponent", f"{team.replace(' ','_').lower()}_per_opponent.csv")
            with c_opp2:
                if len(opp_tbl):
                    figo = px.bar(opp_tbl.head(10), x="opponent", y="win_%", hover_data=["played","wins"], title="Top Opponents by Win%")
                    st.plotly_chart(figo, use_container_width=True)

            # season trend
            if COL_SEASON and COL_SEASON in m_team.columns:
                seas = m_team.groupby(COL_SEASON, as_index=False)["win_flag"].mean()
                seas["win_%"] = (seas["win_flag"] * 100).round(1)
                figs = px.line(seas, x=COL_SEASON, y="win_%", markers=True, title=f"Season Trend — {team} Win% by Year")
                st.plotly_chart(figs, use_container_width=True)
                df_download_button(seas[[COL_SEASON,"win_%"]], "⬇️ Download season trend", f"{team.replace(' ','_').lower()}_season_trend.csv")

            # H2H mini table
            opp_options = [o for o in opp_tbl["opponent"].dropna().tolist() if isinstance(o, str)]
            if len(opp_options):
                sel_opp = st.selectbox("Head-to-Head vs", options=opp_options, index=0, key="h2h_pick")
                h2h = m_team[m_team["opponent"] == sel_opp].copy()
                h2h_played = len(h2h)
                h2h_wins = int(h2h["win_flag"].sum())
                h2h_wp = (h2h_wins / h2h_played * 100) if h2h_played else 0.0

                c_h1, c_h2, c_h3 = st.columns(3)
                with c_h1: metric_card("H2H Played", f"{h2h_played}")
                with c_h2: metric_card("H2H Wins", f"{h2h_wins}")
                with c_h3: metric_card("H2H Win%", f"{h2h_wp:.1f}%")

                if "venue_canon" not in h2h.columns and "venue_canon" in matches.columns:
                    h2h = h2h.merge(matches[[COL_MATCH_ID_M, "venue_canon"]], on=COL_MATCH_ID_M, how="left")

                show_cols = [c for c in [COL_DATE, COL_SEASON, "team1_canon", "team2_canon", "venue_canon", COL_WINNER, COL_RESULT, COL_MATCH_ID_M] if c in h2h.columns]
                h2h_show = h2h[show_cols].copy()
                if COL_DATE in h2h_show.columns:
                    h2h_show = h2h_show.sort_values(COL_DATE, ascending=False)
                else:
                    h2h_show = h2h_show.sort_values(COL_MATCH_ID_M, ascending=False)

                st.markdown("**Recent H2H meetings**")
                st.dataframe(h2h_show.head(12), use_container_width=True)
                df_download_button(h2h_show, "⬇️ Download H2H matches", f"{team.replace(' ','_').lower()}_vs_{sel_opp.replace(' ','_').lower()}_matches.csv")

                # H2H by venue
                if "venue_canon" in h2h.columns:
                    st.markdown("**H2H by Venue**")
                    vtbl = h2h.groupby("venue_canon", as_index=False).agg(played=(COL_MATCH_ID_M, "count"), wins=("win_flag", "sum"))
                    vtbl["win_%"] = (vtbl["wins"] / vtbl["played"] * 100).round(1)
                    vtbl = vtbl.sort_values(["win_%","played"], ascending=[False, False])
                    st.dataframe(vtbl, use_container_width=True)
                    df_download_button(vtbl, "⬇️ Download H2H by venue", f"{team.replace(' ','_').lower()}_vs_{sel_opp.replace(' ','_').lower()}_venues.csv")
                    if len(vtbl):
                        fig_h2h_venue = px.bar(vtbl.head(10), x="venue_canon", y="win_%", hover_data=["played","wins"], title=f"H2H vs {sel_opp}: Best Venues by Win%")
                        st.plotly_chart(fig_h2h_venue, use_container_width=True)

                # Toss decision split inside H2H
                if COL_TOSS_WINNER and COL_TOSS_DEC and COL_TOSS_WINNER in h2h.columns and COL_TOSS_DEC in h2h.columns:
                    st.markdown("**Toss-Decision Impact (H2H)**")
                    def toss_bucket(r):
                        tw = canon_team(r.get(COL_TOSS_WINNER))
                        dec = str(r.get(COL_TOSS_DEC)).title() if pd.notna(r.get(COL_TOSS_DEC)) else "Unknown"
                        if tw == team:
                            return f"Won toss — {dec}"
                        elif tw is np.nan:
                            return "No toss data"
                        else:
                            return f"Lost toss — Opp {dec}"
                    h2h = h2h.copy()
                    h2h["toss_bucket"] = h2h.apply(toss_bucket, axis=1)
                    toss_tbl = h2h.groupby("toss_bucket", as_index=False).agg(played=(COL_MATCH_ID_M, "count"), wins=("win_flag", "sum"))
                    toss_tbl["win_%"] = (toss_tbl["wins"] / toss_tbl["played"] * 100).round(1)
                    st.dataframe(toss_tbl.sort_values(["win_%","played"], ascending=[False, False]), use_container_width=True)
                    df_download_button(toss_tbl, "⬇️ Download H2H toss split", f"{team.replace(' ','_').lower()}_vs_{sel_opp.replace(' ','_').lower()}_toss.csv")
                    if len(toss_tbl):
                        fig_toss = px.bar(toss_tbl, x="toss_bucket", y="win_%", hover_data=["played","wins"], title=f"H2H vs {sel_opp}: Win% by Toss Bucket")
                        st.plotly_chart(fig_toss, use_container_width=True)
        else:
            st.info("No matches for this team in the current filter.")

        # Players subpanel (team + seasons)
        st.markdown("---")
        st.subheader("👥 Players (Selected Team + Seasons)")
        d_team = deliveries.copy()
        if sel_season and (COL_SEASON in d_team.columns):
            d_team = d_team[d_team[COL_SEASON].isin(sel_season)]
        d_team = d_team[(d_team["bat_team_canon"] == team) | (d_team["bowl_team_canon"] == team)]

        roster_bat = d_team.loc[d_team["bat_team_canon"] == team, COL_BATTER]
        roster_bowl = d_team.loc[d_team["bowl_team_canon"] == team, COL_BOWLER]
        roster = sorted(pd.unique(pd.concat([roster_bat, roster_bowl], ignore_index=True).dropna()))

        if len(roster) == 0:
            st.info("No players found for this team in the selected seasons.")
        else:
            cpl1, cpl2 = st.columns([1,1])
            with cpl1:
                st.markdown("**Squad list (derived from deliveries)**")
                st.dataframe(pd.DataFrame({"Player": roster}), use_container_width=True)

            def batting_summary(frame: pd.DataFrame) -> pd.DataFrame:
                if not all([COL_BATTER, COL_BAT_RUNS, COL_IS_WICKET]): return pd.DataFrame()
                g = frame.groupby(COL_BATTER, as_index=False).agg(
                    runs=(COL_BAT_RUNS, "sum"),
                    balls=(COL_BAT_RUNS, "count"),
                    outs=(COL_IS_WICKET, "sum"),
                    fours=(COL_BAT_RUNS, lambda s: (s==4).sum()),
                    sixes=(COL_BAT_RUNS, lambda s: (s==6).sum()),
                )
                g["avg"] = g["runs"] / g["outs"].replace(0, np.nan)
                g["sr"] = g["runs"] / g["balls"] * 100
                return g.sort_values(["runs","sr"], ascending=[False, False])

            def bowling_summary(frame: pd.DataFrame) -> pd.DataFrame:
                if not all([COL_BOWLER, COL_TOTAL_RUNS, COL_IS_WICKET]): return pd.DataFrame()
                g = frame.groupby(COL_BOWLER, as_index=False).agg(
                    balls=(COL_TOTAL_RUNS, "count"),
                    runs_conceded=(COL_TOTAL_RUNS, "sum"),
                    wickets=(COL_IS_WICKET, "sum"),
                )
                g["overs"] = g["balls"] // 6 + (g["balls"] % 6) / 10.0
                g["econ"] = (g["runs_conceded"] / g["balls"]) * 6
                g["avg"] = g["runs_conceded"] / g["wickets"].replace(0, np.nan)
                return g.sort_values(["wickets","econ"], ascending=[False, True])

            with cpl2:
                st.markdown("**Quick Batting Snapshot**")
                bat_q = batting_summary(d_team[d_team["bat_team_canon"] == team])
                st.dataframe(bat_q, use_container_width=True)
                df_download_button(bat_q, "⬇️ Download team batting snapshot", f"{team.replace(' ','_').lower()}_batting_snapshot.csv")

                st.markdown("**Quick Bowling Snapshot**")
                bowl_q = bowling_summary(d_team[d_team["bowl_team_canon"] == team])
                st.dataframe(bowl_q, use_container_width=True)
                df_download_button(bowl_q, "⬇️ Download team bowling snapshot", f"{team.replace(' ','_').lower()}_bowling_snapshot.csv")

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.title("🏏 Cricket Analytics — Players & Matches")
    st.caption("Clean, meaningful insights for players and teams (2018–2025). Use the filters on the left.")

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        total_matches = df[COL_MATCH_ID_D].nunique() if COL_MATCH_ID_D else 0
        metric_card("Matches (filtered)", f"{total_matches:,}")
    with kpi_cols[1]:
        total_runs = int(df[COL_TOTAL_RUNS].sum()) if COL_TOTAL_RUNS else 0
        metric_card("Runs (total)", f"{total_runs:,}")
    with kpi_cols[2]:
        wkts = int(df[COL_IS_WICKET].sum()) if COL_IS_WICKET else 0
        metric_card("Wickets", f"{wkts:,}")
    with kpi_cols[3]:
        balls = df[COL_TOTAL_RUNS].count() if COL_TOTAL_RUNS else 0
        rr = (df[COL_TOTAL_RUNS].sum() / balls) if balls else 0
        metric_card("Avg Runs / Delivery", f"{rr:.2f}", "Proxy for scoring rate")

    st.markdown("---")

    left, right = st.columns([1.1, 1])
    with left:
        if COL_OVER and COL_TOTAL_RUNS and len(df):
            over_runs = df.groupby(COL_OVER, as_index=False)[COL_TOTAL_RUNS].mean().rename(columns={COL_TOTAL_RUNS: "Runs/ball (avg)"})
            fig = px.line(over_runs, x=COL_OVER, y="Runs/ball (avg)", markers=True, title="Over Trend — Average Runs per Ball")
            st.plotly_chart(fig, use_container_width=True)
        if COL_OVER and COL_TOTAL_RUNS and len(df):
            phase_bins = pd.cut(df[COL_OVER], bins=[0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"], include_lowest=True)
            phase_df = df.assign(phase=phase_bins).groupby("phase", as_index=False)[COL_TOTAL_RUNS].mean()
            fig2 = px.bar(phase_df, x="phase", y=COL_TOTAL_RUNS, title="Phase Scoring Rate (Avg Runs per Ball)")
            st.plotly_chart(fig2, use_container_width=True)
    with right:
        if "bat_team_canon" in df.columns and COL_TOTAL_RUNS and len(df):
            team_runs = df.groupby("bat_team_canon", as_index=False)[COL_TOTAL_RUNS].sum().sort_values(COL_TOTAL_RUNS, ascending=False).head(12)
            fig3 = px.bar(team_runs, x="bat_team_canon", y=COL_TOTAL_RUNS, title="Top Team Run Aggregates (Filtered)", text=COL_TOTAL_RUNS)
            fig3.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig3, use_container_width=True)
        if COL_BAT_RUNS and len(df):
            b = df[COL_BAT_RUNS].value_counts().reindex([0,1,2,3,4,6], fill_value=0).reset_index()
            b.columns = ["runs_off_bat", "balls"]
            fig4 = px.pie(b, names="runs_off_bat", values="balls", title="Runs-off-bat Distribution", hole=0.5)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("✨ Auto Insights (Filtered Slice)")
    ins = []
    if len(df):
        if COL_BAT_RUNS and COL_BATTER in df.columns:
            top_bat = df.groupby(COL_BATTER, as_index=False)[COL_BAT_RUNS].sum().sort_values(COL_BAT_RUNS, ascending=False).head(1)
            if len(top_bat):
                ins.append(f"Top run-scorer: **{top_bat.iloc[0][COL_BATTER]}** with **{int(top_bat.iloc[0][COL_BAT_RUNS])}** runs.")
        if COL_BOWLER in df.columns and COL_IS_WICKET in df.columns:
            top_bowl = df.groupby(COL_BOWLER, as_index=False)[COL_IS_WICKET].sum().sort_values(COL_IS_WICKET, ascending=False).head(1)
            if len(top_bowl) and int(top_bowl.iloc[0][COL_IS_WICKET])>0:
                ins.append(f"Top wicket-taker: **{top_bowl.iloc[0][COL_BOWLER]}** with **{int(top_bowl.iloc[0][COL_IS_WICKET])}** wickets.")
        runs = float(df[COL_TOTAL_RUNS].sum()); balls = int(df[COL_TOTAL_RUNS].count())
        rr = safe_ratio(runs, balls) * 6
        ins.append(f"Run rate: **{rr:.2f} rpo** across **{balls}** balls.")
        if COL_BAT_RUNS:
            boundaries = ((df[COL_BAT_RUNS]==4).sum() + (df[COL_BAT_RUNS]==6).sum())
            boundary_pct = safe_ratio(boundaries, balls) * 100
            ins.append(f"Boundary balls: **{boundary_pct:.1f}%** of deliveries were 4s/6s.")
        if COL_OVER in df.columns:
            phase_bins = pd.cut(df[COL_OVER], bins=[0,6,15,20], labels=["PP","Middle","Death"], include_lowest=True)
            phase_rr = df.assign(phase=phase_bins).groupby("phase")[COL_TOTAL_RUNS].mean().mul(6).sort_values(ascending=False)
            if len(phase_rr):
                ins.append(f"Best scoring phase: **{phase_rr.index[0]}** at **{phase_rr.iloc[0]:.2f} rpo**.")
        if "venue_canon" in df.columns:
            v = df["venue_canon"].value_counts().head(1)
            if len(v):
                ins.append(f"Most common venue in this slice: **{v.index[0]}**.")
    else:
        ins.append("No data in current selection.")
    for i in ins:
        st.markdown(f"- {i}")
    st.caption("Insights refresh automatically as you change filters.")

# -----------------------------
# Batting
# -----------------------------
def batting_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if not all([COL_BATTER, COL_BAT_RUNS, COL_IS_WICKET]):
        return pd.DataFrame()
    g = frame.groupby([COL_BATTER, "bat_team_canon"], as_index=False).agg(
        runs=(COL_BAT_RUNS, "sum"),
        balls=(COL_BAT_RUNS, "count"),
        fours=(COL_BAT_RUNS, lambda s: (s==4).sum()),
        sixes=(COL_BAT_RUNS, lambda s: (s==6).sum()),
        dismissals=(COL_IS_WICKET, "sum"),
    )
    g["strike_rate"] = (g["runs"] / g["balls"]) * 100
    g["avg"] = g["runs"] / g["dismissals"].replace(0, np.nan)
    g["boundary%"] = ((g["fours"] + g["sixes"]) / g["balls"]) * 100
    return g.sort_values(["runs","strike_rate"], ascending=[False, False])

with tab_batting:
    st.subheader("Batting Leaderboard (respects Team + Season filters)")
    bat_df = df.copy()
    if sel_team:
        bat_df = bat_df[bat_df["bat_team_canon"].isin(sel_team)]
    bat = batting_summary(bat_df)
    st.dataframe(bat, use_container_width=True)
    if len(bat):
        top_n = st.slider("Top batters by runs", 5, min(50, len(bat)), min(15, len(bat)))
        fig = px.bar(bat.head(top_n), x=COL_BATTER, y="runs", color="bat_team_canon",
                     hover_data=["balls","strike_rate","avg","fours","sixes"], title="Top Run Scorers (Filtered)")
        st.plotly_chart(fig, use_container_width=True)
        if COL_OVER and COL_BATTER and COL_BAT_RUNS:
            st.subheader("Batter — Scoring Rate by Over")
            sel_batters = st.multiselect("Pick batters", options=sorted(bat_df[COL_BATTER].unique()), max_selections=5, key="bat_pick")
            if sel_batters:
                bf = bat_df[bat_df[COL_BATTER].isin(sel_batters)]
                over_bat = bf.groupby([COL_OVER, COL_BATTER], as_index=False)[COL_BAT_RUNS].mean()
                figb = px.line(over_bat, x=COL_OVER, y=COL_BAT_RUNS, color=COL_BATTER, markers=True, title="Average Runs per Ball by Over")
                st.plotly_chart(figb, use_container_width=True)

# -----------------------------
# Bowling
# -----------------------------
def bowling_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if not all([COL_BOWLER, COL_TOTAL_RUNS, COL_IS_WICKET]):
        return pd.DataFrame()
    g = frame.groupby([COL_BOWLER, "bowl_team_canon"], as_index=False).agg(
        balls=(COL_TOTAL_RUNS, "count"),
        runs_conceded=(COL_TOTAL_RUNS, "sum"),
        wickets=(COL_IS_WICKET, "sum"),
    )
    g["overs"] = g["balls"] // 6 + (g["balls"] % 6) / 10.0
    g["economy"] = (g["runs_conceded"] / g["balls"]) * 6
    g["avg"] = g["runs_conceded"] / g["wickets"].replace(0, np.nan)
    return g.sort_values(["wickets","economy"], ascending=[False, True])

with tab_bowling:
    st.subheader("Bowling Leaderboard (respects Team + Season filters)")
    bowl_df = df.copy()
    if sel_team:
        bowl_df = bowl_df[bowl_df["bowl_team_canon"].isin(sel_team)]
    bowl = bowling_summary(bowl_df)
    st.dataframe(bowl, use_container_width=True)
    if len(bowl):
        top_nb = st.slider("Top bowlers by wickets", 5, min(50, len(bowl)), min(15, len(bowl)), key="bowl_top")
        figb = px.bar(bowl.head(top_nb), x=COL_BOWLER, y="wickets", color="bowl_team_canon",
                      hover_data=["economy","avg","overs","runs_conceded"], title="Top Wicket Takers (Filtered)")
        st.plotly_chart(figb, use_container_width=True)
        if COL_OVER and COL_BOWLER and COL_TOTAL_RUNS:
            st.subheader("Bowler — Economy by Over")
            sel_bowlers = st.multiselect("Pick bowlers", options=sorted(bowl_df[COL_BOWLER].unique()), max_selections=5, key="sel_bowl")
            if sel_bowlers:
                bf = bowl_df[bowl_df[COL_BOWLER].isin(sel_bowlers)]
                over_bowl = bf.groupby([COL_OVER, COL_BOWLER], as_index=False)[COL_TOTAL_RUNS].mean()
                over_bowl["economy_per_over"] = over_bowl[COL_TOTAL_RUNS] * 6
                figbo = px.line(over_bowl, x=COL_OVER, y="economy_per_over", color=COL_BOWLER, markers=True, title="Economy (Avg runs per over) by Over")
                st.plotly_chart(figbo, use_container_width=True)

# -----------------------------
# Matches
# -----------------------------
with tab_matches:
    st.subheader("Match Outcomes & Toss Influence")
    if COL_WINNER and COL_TOSS_WINNER and COL_TOSS_DEC:
        m = matches.copy()
        if COL_SEASON in m.columns and sel_season:
            m = m[m[COL_SEASON].isin(sel_season)]
        if sel_venue and "venue_canon" in m.columns:
            m = m[m["venue_canon"].isin(sel_venue)]
        if sel_team and COL_TEAMS in m.columns:
            def has_any(row, picks):
                arr = parse_teams(row[COL_TEAMS])
                if not isinstance(arr, (list, tuple)): return False
                arr_canon = [canon_team(t) for t in arr]
                return any(t in arr_canon for t in picks)
            m = m[m.apply(lambda r: has_any(r, sel_team), axis=1)]
        m["toss_helped"] = np.where(m[COL_WINNER]==m[COL_TOSS_WINNER], "Won after winning toss", "Won after losing toss")
        fig = px.histogram(m, x="toss_helped", color=COL_TOSS_DEC, barmode="group", title="Did winning the toss help? (by toss decision)")
        st.plotly_chart(fig, use_container_width=True)

        if "venue_canon" in m.columns and COL_WINNER:
            top_venues = m["venue_canon"].value_counts().head(10).index.tolist()
            mv = m[m["venue_canon"].isin(top_venues)]
            fig2 = px.histogram(mv, x="venue_canon", color=COL_WINNER, title="Venue-wise Winners (Top 10 venues)", barmode="group")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Matches Table (Filtered)")
    cols_show = [c for c in [COL_DATE, COL_SEASON, "venue_canon", COL_TOSS_WINNER, COL_TOSS_DEC, COL_WINNER, COL_RESULT] if c and (c in matches.columns or c=="venue_canon")]
    mt = matches.copy()
    if COL_SEASON in mt.columns and sel_season:
        mt = mt[mt[COL_SEASON].isin(sel_season)]
    if sel_venue and "venue_canon" in mt.columns:
        mt = mt[mt["venue_canon"].isin(sel_venue)]
    if sel_team and COL_TEAMS in mt.columns:
        def has_any(row, picks):
            arr = parse_teams(row[COL_TEAMS])
            if not isinstance(arr, (list, tuple)): return False
            arr_canon = [canon_team(t) for t in arr]
            return any(t in arr_canon for t in picks)
        mt = mt[mt.apply(lambda r: has_any(r, sel_team), axis=1)]
    if "venue_canon" not in mt.columns and COL_VENUE in matches.columns:
        mt = mt.merge(matches[[COL_MATCH_ID_M, "venue_canon"]], on=COL_MATCH_ID_M, how="left")
    st.dataframe(mt[cols_show] if cols_show else mt, use_container_width=True)

# -----------------------------
# Players
# -----------------------------
with tab_players:
    st.subheader("Player Drilldown")
    player = st.selectbox("Select a player", options=sorted(set(df[COL_BATTER]).union(set(df[COL_BOWLER]))) if len(df) else [])
    if player:
        p_delivs = df[(df[COL_BATTER]==player) | (df[COL_BOWLER]==player)]

        def _batting_summary(frame: pd.DataFrame) -> pd.DataFrame:
            if not all([COL_BATTER, COL_BAT_RUNS, COL_IS_WICKET]): return pd.DataFrame()
            g = frame.groupby([COL_BATTER, "bat_team_canon"], as_index=False).agg(
                runs=(COL_BAT_RUNS, "sum"),
                balls=(COL_BAT_RUNS, "count"),
                fours=(COL_BAT_RUNS, lambda s: (s==4).sum()),
                sixes=(COL_BAT_RUNS, lambda s: (s==6).sum()),
                dismissals=(COL_IS_WICKET, "sum"),
            )
            g["strike_rate"] = (g["runs"] / g["balls"]) * 100
            g["avg"] = g["runs"] / g["dismissals"].replace(0, np.nan)
            return g.sort_values(["runs","strike_rate"], ascending=[False, False])

        def _bowling_summary(frame: pd.DataFrame) -> pd.DataFrame:
            if not all([COL_BOWLER, COL_TOTAL_RUNS, COL_IS_WICKET]): return pd.DataFrame()
            g = frame.groupby([COL_BOWLER, "bowl_team_canon"], as_index=False).agg(
                balls=(COL_TOTAL_RUNS, "count"),
                runs_conceded=(COL_TOTAL_RUNS, "sum"),
                wickets=(COL_IS_WICKET, "sum"),
            )
            g["overs"] = g["balls"] // 6 + (g["balls"] % 6) / 10.0
            g["economy"] = (g["runs_conceded"] / g["balls"]) * 6
            return g.sort_values(["wickets","economy"], ascending=[False, True])

        c1, c2, c3 = st.columns(3)
        bat_p = _batting_summary(p_delivs)
        if len(bat_p) and player in bat_p[COL_BATTER].values:
            row = bat_p[bat_p[COL_BATTER]==player].iloc[0]
            with c1: metric_card("Batting — Runs", int(row["runs"]), f"SR {row['strike_rate']:.1f}")
        bowl_p = _bowling_summary(p_delivs)
        if len(bowl_p) and player in bowl_p[COL_BOWLER].values:
            rowb = bowl_p[bowl_p[COL_BOWLER]==player].iloc[0]
            with c2: metric_card("Bowling — Wickets", int(rowb["wickets"]), f"Econ {rowb['economy']:.2f}")
        with c3:
            metric_card("Matches involved", p_delivs[COL_MATCH_ID_D].nunique() if COL_MATCH_ID_D else 0)

        if COL_OVER and COL_BATTER and COL_BAT_RUNS:
            bat_over = p_delivs[p_delivs[COL_BATTER]==player].groupby(COL_OVER, as_index=False)[COL_BAT_RUNS].mean()
            fig = px.line(bat_over, x=COL_OVER, y=COL_BAT_RUNS, markers=True, title="Batting — Avg Runs per Ball by Over")
            st.plotly_chart(fig, use_container_width=True)
        if COL_OVER and COL_BOWLER and COL_TOTAL_RUNS:
            bowl_over = p_delivs[p_delivs[COL_BOWLER]==player].groupby(COL_OVER, as_index=False)[COL_TOTAL_RUNS].mean()
            bowl_over["economy_per_over"] = bowl_over[COL_TOTAL_RUNS] * 6
            fig2 = px.line(bowl_over, x=COL_OVER, y="economy_per_over", markers=True, title="Bowling — Economy by Over")
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Venues
# -----------------------------
with tab_venues:
    st.subheader("Venue Insights")
    if "venue_canon" in matches.columns and COL_MATCH_ID_D and COL_MATCH_ID_M:
        only_first_inns = st.checkbox("Approximate using only 1st innings (if innings column present)", value=False)
        d2 = deliveries.merge(
            matches[[COL_MATCH_ID_M, "venue_canon"] + ([COL_SEASON] if (COL_SEASON and COL_SEASON in matches.columns) else [])],
            left_on=COL_MATCH_ID_D, right_on=COL_MATCH_ID_M, how="left"
        )
        if sel_season and (COL_SEASON and COL_SEASON in d2.columns):
            d2 = d2[d2[COL_SEASON].isin(sel_season)]
        if sel_team:
            d2 = d2[(d2["bat_team_canon"].isin(sel_team)) | (d2["bowl_team_canon"].isin(sel_team))]
        if only_first_inns and COL_INNING and COL_INNING in d2.columns:
            d2 = d2[d2[COL_INNING].astype(str).str.strip().isin(["1","1.0","1*","1.0*"])]

        if len(d2):
            venue_scores = d2.groupby("venue_canon", as_index=False)[COL_TOTAL_RUNS].sum().rename(columns={COL_TOTAL_RUNS: "Total Runs"})
            fig = px.bar(venue_scores.sort_values("Total Runs", ascending=False).head(12), x="venue_canon", y="Total Runs", title="Venues — Aggregate Runs (Filtered)")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Venue Table")
            st.dataframe(venue_scores.sort_values("Total Runs", ascending=False), use_container_width=True)
            df_download_button(venue_scores, "⬇️ Download venue totals", "venue_totals.csv")
        else:
            st.info("No deliveries matched these filters. Showing venue counts from matches instead.")
            m2 = matches.copy()
            if COL_SEASON and sel_season:
                m2 = m2[m2[COL_SEASON].isin(sel_season)]
            if sel_team and COL_TEAMS in m2.columns:
                def has_any(row, picks):
                    arr = parse_teams(row[COL_TEAMS])
                    if not isinstance(arr, (list, tuple)): return False
                    arr_canon = [canon_team(t) for t in arr]
                    return any(t in arr_canon for t in picks)
                m2 = m2[m2.apply(lambda r: has_any(r, sel_team), axis=1)]
            if "venue_canon" in m2.columns:
                venue_counts = m2.groupby("venue_canon", as_index=False).agg(matches=(COL_MATCH_ID_M, "count"))
                st.dataframe(venue_counts.sort_values("matches", ascending=False), use_container_width=True)
                df_download_button(venue_counts, "⬇️ Download venue match counts", "venue_match_counts.csv")
    else:
        st.info("No 'venue' information available in matches to build this view.")

# -----------------------------
# Team–Venues (global view for every team)
# -----------------------------
with tab_team_venues:
    st.subheader("Team–Venues Leaderboard — Best Venues for Each Team (Win% + Matches)")

    # Filter matches by season/venue (global slice)
    m_all = matches.copy()
    if COL_SEASON in m_all.columns and sel_season:
        m_all = m_all[m_all[COL_SEASON].isin(sel_season)]
    if sel_venue and "venue_canon" in m_all.columns:
        m_all = m_all[m_all["venue_canon"].isin(sel_venue)]

    # Ensure required columns exist
    if COL_WINNER and COL_WINNER in m_all.columns:
        m_all["winner_canon"] = clean_str(m_all[COL_WINNER]).apply(canon_team)
    else:
        m_all["winner_canon"] = np.nan

    if "team1_canon" not in m_all.columns or "team2_canon" not in m_all.columns:
        if COL_TEAMS in m_all.columns:
            teams_parsed = m_all[COL_TEAMS].apply(parse_teams)
            m_all["team1_canon"] = teams_parsed.apply(lambda t: canon_team(t[0]) if isinstance(t, (list, tuple)) and len(t) == 2 else np.nan)
            m_all["team2_canon"] = teams_parsed.apply(lambda t: canon_team(t[1]) if isinstance(t, (list, tuple)) and len(t) == 2 else np.nan)
        else:
            m_all["team1_canon"] = np.nan
            m_all["team2_canon"] = np.nan

    # Build per-team, per-venue rows (each match contributes one row for each participant)
    a = m_all[[COL_MATCH_ID_M, "venue_canon", "team1_canon", "winner_canon"]].rename(columns={"team1_canon":"team"})
    b = m_all[[COL_MATCH_ID_M, "venue_canon", "team2_canon", "winner_canon"]].rename(columns={"team2_canon":"team"})
    long = pd.concat([a, b], ignore_index=True)
    long["team"] = long["team"].apply(canon_team)
    long = long.dropna(subset=["team", "venue_canon"])

    # Optional team filter from sidebar (if user set Team filter, honor it)
    if sel_team and len(sel_team):
        long = long[long["team"].isin(sel_team)]

    # Aggregate
    tv = (
        long
        .assign(win=(long["winner_canon"] == long["team"]).astype(int))
        .groupby(["team", "venue_canon"], as_index=False)
        .agg(played=(COL_MATCH_ID_M, "count"), wins=("win", "sum"))
    )
    tv["win%"] = (tv["wins"] / tv["played"] * 100).round(1)

    # Controls
    min_p = st.slider("Minimum matches at a venue", 1, 15, 3, key="tv_min_matches")
    teams_in_table = sorted(tv["team"].dropna().unique().tolist())
    default_teams = sel_team if (sel_team and len(sel_team)) else teams_in_table
    choice_teams = st.multiselect("Teams to show", options=teams_in_table, default=default_teams)

    tv_show = tv[(tv["played"] >= min_p) & (tv["team"].isin(choice_teams))].copy()
    tv_show = tv_show.sort_values(["team", "win%", "played"], ascending=[True, False, False])

    st.dataframe(tv_show, use_container_width=True)
    df_download_button(tv_show, "⬇️ Download Team–Venues table", "team_venues_winpct.csv")

    # If exactly one team is selected, show a chart of their best venues
    if len(choice_teams) == 1:
        tname = choice_teams[0]
        tdf = tv_show[tv_show["team"] == tname].sort_values(["win%", "played"], ascending=[False, False]).head(12)
        if len(tdf):
            fig_tv = px.bar(tdf, x="venue_canon", y="win%", hover_data=["played","wins"], title=f"{tname} — Best Venues by Win% (min {min_p} matches)")
            st.plotly_chart(fig_tv, use_container_width=True)

