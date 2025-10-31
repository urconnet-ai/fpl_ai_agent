"""
FPL AI Agent — Phase 2.5
Enhanced filtering for active players only.
Keeps ML + Gemini structure, filters out irrelevant players (transferred/unavailable/0-activity)
"""

import os, json, time, sqlite3, requests, numpy as np, re
import openai
from openai import OpenAI
from openai._exceptions import RateLimitError
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# -----------------------
# CONFIG
# -----------------------
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
DB_PATH = "fpl_ai.db"


GEMINI_API_KEY = "AIzaSyA8oIR5FNUQDNu7cpb-47sZmxAW5P4CRGQ"
GEMINI_URL = (
  f"https://generativelanguage.googleapis.com/v1beta/"
  f"models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

OPENAI_API_KEY = "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"  # ← your actual OpenAI API key
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
#openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------
# FETCH & FILTER
# -----------------------
def fetch_fpl_data():
    r = requests.get(FPL_API_URL, timeout=15)
    r.raise_for_status()
    data = r.json()
    rfx = requests.get(FIXTURES_URL, timeout=15)
    rfx.raise_for_status()
    fixtures = rfx.json()

    teams = {t["id"]: t["name"] for t in data["teams"]}
    current_gw = next((e["id"] for e in data["events"] if e.get("is_current")), None)
    if current_gw is None:
        current_gw = next((e["id"] for e in data["events"] if e.get("is_next")), 1) - 1

    # Teams actually playing this GW
    valid_team_ids = {fx["team_h"] for fx in fixtures if fx.get("event") == current_gw} | {
        fx["team_a"] for fx in fixtures if fx.get("event") == current_gw
    }
    valid_teams = [teams[i] for i in valid_team_ids]

    players = []
    for p in data["elements"]:
        team = teams[p["team"]]
        price = round(p["now_cost"] / 10, 1)
        status_map = {"i": "Injured", "d": "Doubtful", "s": "Suspended", "u": "Unavailable", "a": "Available"}
        status = status_map.get(p["status"], "Available")
        players.append({
            "id": p["id"],
            "name": f"{p['first_name']} {p['second_name']}".strip(),
            "team": team,
            "position": p["element_type"],
            "price": price,
            "cost_change": p["cost_change_event"],
            "transfers_in": p["transfers_in_event"],
            "transfers_out": p["transfers_out_event"],
            "selected_by_percent": float(p["selected_by_percent"]),
            "points": p["total_points"],
            "status": status,
        })

    # 🧹 FILTER: remove inactive or irrelevant players
    players = [
        p for p in players
        if (
            p["team"] in valid_teams
            and p["status"] not in ("Unavailable", "u", "na")
            and not (p["points"] == 0 and p["transfers_in"] == 0 and p["transfers_out"] == 0)
        )
    ]

    return players, current_gw, fixtures, teams


# -----------------------
# FIXTURE DIFFICULTY
# -----------------------
def compute_fixture_difficulty(fixtures, teams):
    td = {t: [] for t in teams.values()}
    for fx in fixtures:
        if not fx.get("event"):
            continue
        th, ta = fx["team_h"], fx["team_a"]
        td[teams[th]].append(fx["team_h_difficulty"])
        td[teams[ta]].append(fx["team_a_difficulty"])
    return {t: round(sum(v)/len(v), 2) if v else 3.0 for t, v in td.items()}

# -----------------------
# ML MODEL
# -----------------------
def train_price_model(players):
    X, y = [], []
    for p in players:
        label = 1 if p["cost_change"] > 0 else -1 if p["cost_change"] < 0 else 0
        y.append(label)
        X.append([p["transfers_in"], p["transfers_out"], p["selected_by_percent"], p["points"]])
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=80, random_state=42).fit(Xs, y)
    return model, scaler

def apply_price_predictions(players, model, scaler):
    Xp = np.array([[p["transfers_in"], p["transfers_out"], p["selected_by_percent"], p["points"]] for p in players])
    Xs = scaler.transform(Xp)
    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)
    classes = list(model.classes_)
    for p, pred, prob in zip(players, preds, probs):
        pm = {c: float(pr) for c, pr in zip(classes, prob)}
        p["predicted_trend"] = "Likely to rise" if pred == 1 else "Likely to fall" if pred == -1 else "Stable"
        p["predicted_price"] = round(p["price"] + 0.1, 1) if pred == 1 else round(p["price"] - 0.1, 1) if pred == -1 else p["price"]
        p["prob_rise"], p["prob_fall"], p["prob_stable"] = round(pm.get(1,0)*100,1), round(pm.get(-1,0)*100,1), round(pm.get(0,0)*100,1)

# -----------------------
# ANALYTICS METRICS
# -----------------------
def compute_player_metrics(players, team_difficulty):
    for p in players:
        momentum = (p["transfers_in"] - p["transfers_out"]) / max(p["transfers_out"]+1, 1)
        td = team_difficulty.get(p["team"], 3.0)
        injury_flag = 1.0 if p["status"] in ("Injured","Suspended") else 0.0
        p["momentum"] = round(momentum,3)
        p["opportunity_index"] = round(((max(momentum,0))*0.4 + (p.get("prob_rise",0)/100)*0.4 + (1/(td+0.1))*0.2),3)
        p["risk_index"] = round((injury_flag*0.5 + (td/5)*0.25 + (p.get("prob_fall",0)/100)*0.25),3)

# -----------------------
# GEMINI ANALYSIS
# -----------------------

def analyze_with_gemini(players, team_difficulty):

    # ----------------------------------------------------
    # 1️⃣ Prepare compact player payload (exclude transfer counts)
    # ----------------------------------------------------
    payload_players = []
    for p in players:
        subset = {k: p.get(k, None) for k in [
            "name", "team", "position", "price",
            "predicted_price", "predicted_trend",
            "prob_rise", "prob_fall", "selected_by_percent",
            "points", "status", "momentum",
            "opportunity_index", "risk_index"
        ]}
        payload_players.append(subset)

    # ----------------------------------------------------
    # 2️⃣ Build structured prompt for Gemini
    # ----------------------------------------------------
    """
    prompt = (
        "You are an advanced FPL data analyst.\n"
        "Use the provided data to identify insights about player price movements, "
        "captaincy opportunities, and value picks.\n"
        "Do NOT include or fabricate transfer counts; these come from official FPL data.\n\n"
        "Return valid JSON with these keys:\n"
        "List at least 10 players in each category including risers and fallers based on my initial idea.\n"
        "risers, fallers, watchlist, captaincy_picks, value_picks, bandwagons, commentary.\n\n"
        "Top risers and top fallers should be sorted on form and predicted price.\n"
        "Each player object must include: player_name, team, price, predicted_price, "
        "predicted_trend, prob_rise, prob_fall, status.\n"
        "Rank intelligently by form, probability, or opportunity.\n\n"
        f"PLAYERS: {json.dumps(payload_players)[:10000]}\n"
        f"TEAM_DIFFICULTY: {json.dumps(team_difficulty)}"
    )
    """

    prompt = (
        "You are an FPL data analyst.\n"
        "Ignore any player not active this season or not playing in the current gameweek.\n"
        "Return JSON with keys: top_ins, top_outs, risers, fallers, watchlist, captaincy_picks, value_picks, bandwagons, commentary.\n"
        "Each player object must have: player_name, team, price, predicted_price, predicted_trend, prob_rise, prob_fall, status.\n"
        "Include **as many relevant players as possible** (at least 10 limit). Rank them by form, activity, probability, or opportunity.\n"
        f"PLAYERS: {json.dumps(payload_players)}\nTEAM_DIFFICULTY: {json.dumps(team_difficulty)}"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    # ----------------------------------------------------
    # 3️⃣ Call Gemini safely with retries & backoff
    # ----------------------------------------------------
    resp = None
    for attempt in range(4):
        try:
            r = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"},
                              json=payload, timeout=35)
            if r.status_code == 200:
                resp = r
                break
            elif r.status_code == 429:
                wait = (attempt + 1) * random.uniform(4, 7)
                print(f"[Gemini] Rate limited — waiting {wait:.1f}s before retry {attempt+1}")
                time.sleep(wait)
                continue
            else:
                raise Exception(f"[Gemini] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            if attempt < 3:
                wait = (attempt + 1) * 5
                print(f"[Gemini] Retry {attempt+1} due to {e} (waiting {wait}s)")
                time.sleep(wait)
                continue
            print(f"[Gemini] Final failure: {e}")
            break

    # ----------------------------------------------------
    # 4️⃣ Parse response or return fallback
    # ----------------------------------------------------
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            txt = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            json_text = txt[txt.find("{"):txt.rfind("}")+1]
            result = json.loads(json_text)
        except Exception as e:
            print(f"[Gemini] JSON parse error: {e}")
            result = {k: [] for k in [
                "risers", "fallers", "watchlist",
                "captaincy_picks", "value_picks", "bandwagons"
            ]}
            result["commentary"] = "Parsing error — Gemini response incomplete."
    else:
        # fallback narrative when Gemini unavailable
        result = {
            "risers": [], "fallers": [], "watchlist": [],
            "captaincy_picks": [], "value_picks": [], "bandwagons": [],
            "commentary": (
                "⚠️ Gemini quota limit reached. "
                "AI insights temporarily unavailable — using live FPL and ML data only."
            ),
        }

    return result

    """payload_players = [
        {k:p[k] for k in [
            "name","team","position","price","predicted_price","predicted_trend",
            "prob_rise","prob_fall","transfers_in","transfers_out",
            "selected_by_percent","points","status","momentum","opportunity_index","risk_index"
        ]} for p in players
    ]
    
    prompt = (
        "You are an FPL data analyst.\n"
        "Ignore any player not active this season or not playing in the current gameweek.\n"
        "Return JSON with keys: top_ins, top_outs, risers, fallers, watchlist, captaincy_picks, value_picks, bandwagons, commentary.\n"
        "Each player object must have: player_name, team, price, predicted_price, predicted_trend, prob_rise, prob_fall, status.\n"
        "Include **as many relevant players as possible** (at least 10 limit). Rank them by form, activity, probability, or opportunity.\n"
        f"PLAYERS: {json.dumps(payload_players)}\nTEAM_DIFFICULTY: {json.dumps(team_difficulty)}"
    )
    
    resp = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json={"contents":[{"parts":[{"text":prompt}]}]})
    resp.raise_for_status()
    data = resp.json()
    
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        json_text = txt[txt.find("{"):txt.rfind("}")+1]
        result = json.loads(json_text)
    except Exception:
        result = {k:[] for k in ["top_ins","top_outs","risers","fallers","watchlist","captaincy_picks","value_picks","bandwagons"]}
        result["commentary"] = "Parsing error."
    return result
"""
"""

import json, re, requests
from openai import OpenAI
from openai._exceptions import RateLimitError

client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_with_gemini(players, team_difficulty):


    import json, requests, time

    essentials = [
        "name","team","position","price","predicted_price","predicted_trend",
        "prob_rise","prob_fall","transfers_in","transfers_out",
        "selected_by_percent","points","status","momentum",
        "opportunity_index","risk_index"
    ]

    # 🔹 compress numeric values aggressively
    payload_players = [
        {k: (round(p[k], 2) if isinstance(p[k], float)
             else int(p[k] / 100) if isinstance(p[k], int) and abs(p[k]) > 100 else p[k])
         for k in essentials if k in p}
        for p in players
    ]

    payload_json = json.dumps(payload_players, ensure_ascii=False)
    CHUNK_SIZE = 12000  # reduce latency per API call
    chunks = [payload_json[i:i + CHUNK_SIZE] for i in range(0, len(payload_json), CHUNK_SIZE)]

    all_results = {k: [] for k in [
        "top_ins","top_outs","risers","fallers",
        "watchlist","captaincy_picks","value_picks","bandwagons"
    ]}
    commentary = ""

    # 🔹 Parallel chunk processing (non-blocking async-style batching)
    for idx, chunk in enumerate(chunks):
        
        prompt = (
            f"You are an FPL data analyst. Chunk {idx+1}/{len(chunks)}.\n"
            "Generate JSON with keys: top_ins, top_outs, risers, fallers, watchlist, "
            "captaincy_picks, value_picks, bandwagons, commentary.\n"
            "Include ~10–15 top players per key ranked by activity.\n"
            "Respond with pure JSON only.\n\n"
            f"PLAYERS: {chunk}\nTEAM_DIFFICULTY: {json.dumps(team_difficulty)}"
        )
        

        # ⚡ try OpenAI first, with shorter timeout
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an FPL data summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25,
                max_tokens=2200,
                timeout=25,  # shorter
            )
            txt = response.choices[0].message.content
            json_text = txt[txt.find("{"):txt.rfind("}") + 1]
            sub_result = json.loads(json_text)
            for k in all_results.keys():
                all_results[k].extend(sub_result.get(k, []))
            commentary += sub_result.get("commentary", "")
            print(f"✅ GPT processed chunk {idx+1}/{len(chunks)} quickly.")
            continue
        except Exception as e:
            print(f"⚠️ OpenAI timeout or quota issue: {e}")

        # ⚙️ fallback to Gemini with shorter prompt and timeout
        try:
            GEMINI_URL = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            )
            resp = requests.post(
                GEMINI_URL,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=40,
            )
            data = resp.json()
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            json_text = txt[txt.find("{"):txt.rfind("}") + 1]
            sub_result = json.loads(json_text)
            for k in all_results.keys():
                all_results[k].extend(sub_result.get(k, []))
            commentary += sub_result.get("commentary", "")
            print(f"✅ Gemini processed chunk {idx+1}/{len(chunks)} successfully.")
        except Exception as e2:
            print(f"❌ Gemini failed for chunk {idx+1}: {e2}")
            continue

    all_results["commentary"] = commentary.strip() or "Aggregated analysis from all chunks."
    return all_results
"""
# -----------------------
# SQLITE SAVE (Optional)
# -----------------------
def save_to_sqlite(gw, analysis):
    conn = sqlite3.connect(DB_PATH)
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
      captaincy_picks TEXT,
      value_picks TEXT,
      bandwagons TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""
      INSERT INTO weekly_reports (
        gw, commentary, top_ins, top_outs, risers, fallers, watchlist,
        captaincy_picks, value_picks, bandwagons
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (gw, analysis.get("commentary",""),
          json.dumps(analysis.get("top_ins",[])), json.dumps(analysis.get("top_outs",[])),
          json.dumps(analysis.get("risers",[])), json.dumps(analysis.get("fallers",[])),
          json.dumps(analysis.get("watchlist",[])), json.dumps(analysis.get("captaincy_picks",[])),
          json.dumps(analysis.get("value_picks",[])), json.dumps(analysis.get("bandwagons",[]))))
    conn.commit()
    conn.close()

# -----------------------
# DISPLAY
# -----------------------
def display_report(gw, analysis):
    print(f"\n=== Gameweek {gw} — Market Analysis ===\n{analysis.get('commentary','')}\n")
    def show(title, items):
        print(title)
        if not items: print("  (no data)\n"); return
        for i,p in enumerate(items,1):
            print(f" {i}. {p.get('player_name',p.get('name'))} ({p.get('team')}) "
                  f"£{p.get('price')} → £{p.get('predicted_price')} [{p.get('predicted_trend')}] "
                  f"(rise {p.get('prob_rise')}%, fall {p.get('prob_fall')}%) - {p.get('status')}")
        print()
    for s in ["top_ins","top_outs","risers","fallers","watchlist","captaincy_picks","value_picks","bandwagons"]:
        show(s.replace("_"," ").title(), analysis.get(s, []))

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    players, gw, fixtures, teams = fetch_fpl_data()
    team_diff = compute_fixture_difficulty(fixtures, teams)
    model, scaler = train_price_model(players)
    apply_price_predictions(players, model, scaler)
    compute_player_metrics(players, team_diff)
    analysis = analyze_with_gemini(players, team_diff)
    save_to_sqlite(gw, analysis)
    display_report(gw, analysis)
    print("✅ Done.")
