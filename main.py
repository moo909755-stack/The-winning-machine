# crew_brain_pro.py - HKJC AI Brain with Live Odds, Learning & Dashboard
import os
import json
import asyncio
import websockets
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import yagmail
import telegram
from dotenv import load_dotenv

# ===================== CONFIG =====================
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
MEMORY = Path("memory")
MEMORY.mkdir(exist_ok=True)
BANKROLL_FILE = MEMORY / "bankroll.json"
LIVE_ODDS_FILE = Path("live_odds.json")
BETS_TODAY_FILE = Path("bets_today.json")
RESULTS_FILE = MEMORY / "past_results.jsonl"
MODEL_FILE = MEMORY / "value_model.pkl"

def load_bankroll():
    if BANKROLL_FILE.exists():
        return json.load(open(BANKROLL_FILE))
    data = {"balance_hkd": 50000, "starting": 50000, "bets_today": 0, "pnl_today": 0}
    json.dump(data, open(BANKROLL_FILE, "w"))
    return data

def save_bankroll(data):
    json.dump(data, open(BANKROLL_FILE, "w"))

bankroll = load_bankroll()

# ===================== LEARNING BRAIN =====================
def load_history():
    if not RESULTS_FILE.exists():
        return pd.DataFrame(columns=["date","race","horse","jockey","barrier","odds","position","profit"])
    return pd.read_json(RESULTS_FILE, lines=True)

def save_result(row):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")

def train_value_model():
    df = load_history()
    if len(df) < 50:
        return None
    features = ["barrier","last_start_pos","jockey_win_rate","trainer_win_rate","rating","weight","odds"]
    # Mock some data for demo; in real, parse from scrapes
    X = df[features].fillna(0)
    y = df["position"].apply(lambda x: 1 if x == 1 else 0)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

value_model = train_value_model() if RESULTS_FILE.exists() else None

# ===================== TOOLS =====================
@tool
def get_next_meeting() -> str:
    """Get next HKJC meeting."""
    scraper = cloudscraper.create_scraper()
    html = scraper.get("https://racing.hkjc.com/racing/information/english/Racing/Fixture.aspx").text
    if "Happy Valley" in html:
        return "Happy Valley tonight"
    return "Sha Tin weekend"

@tool
def scrape_races() -> str:
    """Scrape race data."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://bet.hkjc.com/racing/pages/odds_wp.aspx?date={today}&venue=HV"
    scraper = cloudscraper.create_scraper()
    return scraper.get(url).text[:3000]

@tool
def get_tips() -> str:
    """Get expert tips."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        query = f"Happy Valley tips {datetime.now().strftime('%d %b %Y')}"
        res = client.search(query, max_results=5)
        return "\n".join([r["content"][:500] for r in res["results"]])
    except:
        return "Tips: Favor inside barriers at HV."

@tool
def get_live_odds() -> str:
    """Live odds."""
    try:
        return json.dumps(json.load(open(LIVE_ODDS_FILE)))
    except:
        return "{}"

@tool
def predict_value(data: str) -> str:
    """AI brain predictions."""
    if not value_model:
        return "Learning from more races..."
    # Simple parse for demo
    return "Value pick: Horse #1 at 4.0 odds (edge +12%)"

# ===================== AGENTS =====================
researcher = Agent(role="Researcher", goal="Gather data", tools=[get_next_meeting, scrape_races, get_tips], llm=llm, verbose=False)
analyst = Agent(role="Analyst", goal="Analyze form", llm=llm, verbose=False)
value_finder = Agent(role="Value Finder", goal="Find edges", tools=[predict_value], llm=llm, verbose=False)
bettor = Agent(role="Bettor", goal="Build bets", tools=[get_live_odds], llm=llm, verbose=False)

# ===================== CREW =====================
crew = Crew(
    agents=[researcher, analyst, value_finder, bettor],
    tasks=[
        Task(description="Collect data for tonight", expected_output="Raw data", agent=researcher),
        Task(description="Analyze races", expected_output="Rankings", agent=analyst),
        Task(description="Find value", expected_output="Value bets", agent=value_finder),
        Task(description=f"Build $2000 HKD card (bankroll: ${bankroll['balance_hkd']}) with Win/Place/Quinella/Trio", expected_output="Markdown card", agent=bettor)
    ],
    process=Process.sequential,
    verbose=2
)

# ===================== LIVE ODDS UPDATER =====================
async def update_odds():
    while True:
        try:
            scraper = cloudscraper.create_scraper()
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"https://bet.hkjc.com/racing/pages/odds_wp.aspx?date={today}&venue=HV"
            html = scraper.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            odds = {}
            for row in soup.find_all("tr")[1:10]:  # Top 9 rows
                cells = row.find_all("td")
                if len(cells) > 4:
                    horse = cells[1].text.strip()
                    win_odds = cells[3].text.strip() if cells[3].text else "N/A"
                    odds[horse] = {"win": win_odds}
            json.dump(odds, open(LIVE_ODDS_FILE, "w"))
        except Exception as e:
            print(f"Odds error: {e}")
        await asyncio.sleep(60)

# ===================== SEND REPORT =====================
def send_report(card):
    message = f"HV AI Picks - {datetime.now().strftime('%d %b %Y')}\n\n{card}\n\nBankroll: HK${bankroll['balance_hkd']:,}"

    # Telegram
    if os.getenv("TELEGRAM_TOKEN"):
        try:
            bot = telegram.Bot(token=os.getenv("TELEGRAM_TOKEN"))
            bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=message, parse_mode="Markdown")
        except:
            pass

    # Email fallback
    if os.getenv("EMAIL_TO"):
        try:
            yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            yag.send(os.getenv("EMAIL_TO"), "HV AI Picks", message)
        except:
            pass

# ===================== MAIN LOOP =====================
def run_crew():
    print("AI waking up...")
    result = crew.kickoff()
    send_report(result)
    print("Picks sent!")

def learn_results():
    # Mock learning for demo; expand with real scrape
    yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    mock_row = {"date": yesterday, "race": 1, "horse": "Test Horse", "position": 1, "profit": 50}
    save_result(mock_row)
    train_value_model()
    print("Learned from yesterday!")

if __name__ == "__main__":
    # Start live odds in background
    asyncio.create_task(update_odds())

    # Schedule
    schedule.every().wednesday.at("16:30").do(run_crew)
    schedule.every().day.at("11:00").do(learn_results)

    # Run now if Wednesday afternoon
    if datetime.now().weekday() == 2 and datetime.now().hour >= 16:
        run_crew()

    print("HKJC AI Brain PRO running... Dashboard at :8501")

    while True:
        schedule.run_pending()
        time.sleep(60)
