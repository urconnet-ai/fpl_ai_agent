###########################################
## 1. Update: Fetch & Enrich FPL Data
#we now return fixtures and teams so we can tell the AI “these clubs have good fixtures, thats why people are buying them.
import requests
import json
import sqlite3
import os
from datetime import datetime

FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

def get_fpl_data():
    # base data
    r = requests.get(FPL_API_URL, timeout=15)
    r.raise_for_status()
    data = r.json()

    # fixtures for context
    fx = requests.get(FIXTURES_URL, timeout=15)
    fx.raise_for_status()
    fixtures = fx.json()

    # current GW
    current_gw = next((e["id"] for e in data["events"] if e.get("is_current")), None)
    if current_gw is None:
        current_gw = next((e["id"] for e in data["events"] if e.get("is_next")), 1) - 1

    # team map
    team_map = {t["id"]: t["name"] for t in data["teams"]}

    # enrich players
    players = []
    for p in data["elements"]:
        current_price = round(p["now_cost"] / 10, 1)
        if p["cost_change_event"] > 0:
            predicted_price = round(current_price + 0.1, 1)
            price_trend = "Likely to rise"
        elif p["cost_change_event"] < 0:
            predicted_price = round(current_price - 0.1, 1)
            price_trend = "Likely to fall"
        else:
            predicted_price = current_price
            price_trend = "Stable"

        players.append({
            "player_name": f"{p['first_name']} {p['second_name']}".strip(),
            "team": team_map.get(p["team"], "Unknown"),
            "position": p["element_type"],      # 1 GK, 2 DEF, 3 MID, 4 FWD
            "status": p["status"],              # 'a', 'i', 'd', 's'
            "price": current_price,
            "predicted_price": predicted_price,
            "price_trend": price_trend,
            "transfers_in_event": p["transfers_in_event"],
            "transfers_out_event": p["transfers_out_event"],
            "cost_change_event": p["cost_change_event"],
            "selected_by_percent": p["selected_by_percent"],
            "total_points": p["total_points"],
            "raw": p,
        })

    return players, current_gw, fixtures, data["teams"]

###########################################
## 2. Extended AI model
import time
import json
import requests

GEMINI_API_KEY = "AIzaSyA8oIR5FNUQDNu7cpb-47sZmxAW5P4CRGQ"
GEMINI_URL = (
  f"https://generativelanguage.googleapis.com/v1beta/"
  f"models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

def analyze_players_gemini(players, fixtures, teams):
    # build fixture difficulty per team (very simple)
    # FPL fixture object: team_h, team_a, event, team_h_difficulty, team_a_difficulty
    team_fixture_map = {}
    for fx in fixtures:
        gw = fx.get("event")
        if not gw:
            continue
        th = fx["team_h"]
        ta = fx["team_a"]
        th_diff = fx["team_h_difficulty"]
        ta_diff = fx["team_a_difficulty"]
        team_fixture_map.setdefault(th, []).append(th_diff)
        team_fixture_map.setdefault(ta, []).append(ta_diff)

    # make team id -> name
    team_id_to_name = {t["id"]: t["name"] for t in teams}

    # minimal player data for LLM
    minimal_data = []
    for p in players:
        minimal_data.append({
            "player_name": p["player_name"],
            "team": p["team"],
            "position": p["position"],
            "status": p["status"],
            "price": p["price"],
            "predicted_price": p["predicted_price"],
            "price_trend": p["price_trend"],
            "transfers_in_event": p["transfers_in_event"],
            "transfers_out_event": p["transfers_out_event"],
            "cost_change_event": p["cost_change_event"],
            "selected_by_percent": p["selected_by_percent"],
            "total_points": p["total_points"],
        })

    prompt = (
        "You are an advanced Fantasy Premier League analyst. "
        "You will receive:\n"
        "1) full player list (reduced fields), and\n"
        "2) team fixture difficulty by team.\n\n"
        "Your job:\n"
        "- Return EXACTLY 6 keys in JSON: top_ins, top_outs, risers, fallers, watchlist, commentary.\n"
        "- top_ins: 10 players most transferred IN this GW.\n"
        "- top_outs: 10 players most transferred OUT this GW.\n"
        "- risers: 10 players with positive cost_change_event or 'Likely to rise'.\n"
        "- fallers: 10 players with negative cost_change_event or 'Likely to fall'.\n"
        "- watchlist: 10 players that are close to a rise/fall, or have good fixtures coming, or returning from injury.\n"
        "- commentary: a short paragraph tying transfers to fixtures and prices.\n"
        "For every player object IN ANY LIST include:\n"
        "player_name, team, position, price, predicted_price, price_trend, transfers_in_event, transfers_out_event.\n"
        "Prefer players with good upcoming fixtures (low difficulty) when ties occur.\n"
        "Use multiple clubs, not just one.\n"
        "Respond with JSON only.\n\n"
        f"PLAYERS: {json.dumps(minimal_data)}\n\n"
        f"TEAMS_FIXTURES: {json.dumps(team_fixture_map)}\n"
        f"TEAM_NAMES: {json.dumps(team_id_to_name)}\n"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    for _ in range(3):
        resp = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=payload)
        if resp.status_code == 429:
            time.sleep(30)
            continue
        resp.raise_for_status()
        data = resp.json()
        break

    # parse
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])
    except Exception as e:
        print("LLM parsing failed:", e)
        parsed = {
            "top_ins": [],
            "top_outs": [],
            "risers": [],
            "fallers": [],
            "watchlist": [],
            "commentary": "LLM failed to produce JSON."
        }

    # hard-enforce 10 per list
    for key in ["top_ins", "top_outs", "risers", "fallers", "watchlist"]:
        if key in parsed and isinstance(parsed[key], list):
            parsed[key] = parsed[key][:10]
        else:
            parsed[key] = []

    return parsed

###########################################
## 3. Update SQLite to store new fields
def save_to_sqlite(current_gw, analysis):
    conn = sqlite3.connect("fpl_ai.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS weekly_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gw INTEGER,
        commentary TEXT,
        top_ins TEXT,
        top_outs TEXT,
        risers TEXT,
        fallers TEXT,
        watchlist TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    INSERT INTO weekly_reports (gw, commentary, top_ins, top_outs, risers, fallers, watchlist)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        current_gw,
        analysis.get("commentary", ""),
        json.dumps(analysis.get("top_ins", [])),
        json.dumps(analysis.get("top_outs", [])),
        json.dumps(analysis.get("risers", [])),
        json.dumps(analysis.get("fallers", [])),
        json.dumps(analysis.get("watchlist", [])),
    ))
    conn.commit()
    conn.close()


###########################################
## 4. Make the output nicer
def display_latest_summary():
    conn = sqlite3.connect("fpl_ai.db")
    c = conn.cursor()
    c.execute("""
        SELECT gw, commentary, top_ins, top_outs, risers, fallers, watchlist
        FROM weekly_reports
        ORDER BY id DESC LIMIT 1;
    """)
    row = c.fetchone()
    conn.close()
    if not row:
        print("No reports yet.")
        return

    gw, commentary, top_ins, top_outs, risers, fallers, watchlist = row
    print(f"\n=== Gameweek {gw} Report ===\n")
    print(commentary, "\n")

    top_ins = json.loads(top_ins)
    top_outs = json.loads(top_outs)
    risers = json.loads(risers)
    fallers = json.loads(fallers)
    watchlist = json.loads(watchlist)

    def show(title, items):
        print(title)
        for i, p in enumerate(items, start=1):
            print(f" {i:2}. {p['player_name']} ({p['team']}) £{p['price']} → £{p['predicted_price']} [{p['price_trend']}]")
        print()

    show("Top Transfers IN", top_ins)
    show("Top Transfers OUT", top_outs)
    show("Top Price Risers", risers)
    show("Top Price Fallers", fallers)
    show("Watchlist (near change / good fixtures)", watchlist)


def main():
    print("Fetching FPL data...")
    players, current_gw, fixtures, teams = get_fpl_data()

    print(f"Analyzing Gameweek {current_gw} data via Gemini...")
    analysis = analyze_players_gemini(players, fixtures, teams)

    print(f"Saving Gameweek {current_gw} report to SQLite...")
    save_to_sqlite(current_gw, analysis)

    print("Displaying latest summary...\n")
    display_latest_summary()


if __name__ == "__main__":
    main()
