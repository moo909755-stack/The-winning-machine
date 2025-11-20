# main.py – Happy Valley AI Brain for PythonAnywhere (free tier)
import os
import json
import asyncio
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
import telegram
from dotenv import load_dotenv

# Load secrets (.env file you created)
load_dotenv()

# ===================== CONFIG =====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
MEMORY = Path("/home") / os.getenv("PYTHONANYWHERE_USERNAME", "yourusername") / "hkjc-ai-brain-pro" / "memory"
MEMORY.mkdir(exist_ok=True)
BANKROLL_FILE = MEMORY / "bankroll.json"
LIVE_ODDS_FILE = MEMORY / "live_odds.json"
RESULTS_FILE = MEMORY / "past_results.jsonl"
MODEL_FILE = MEMORY / "value_model.pkl"

# ===================== BANKROLL =====================
def load_bankroll():
    if BANKROLL_FILE.exists():
        return json.load(open(BANKROLL_FILE))
    data = {"balance_hkd": 50000, "starting": 50000, "bets_today": 0, "pnl_today": 0}
    json.dump(data, open(BANKROLL_FILE, "w"))
    return data

bankroll = load_bankroll()

# ===================== LEARNING BRAIN (simplified) =====================
def train_value_model():
    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 100:
        print("Brain already learning…")
        # Real training code here later – for now just keep it simple
    else:
        print("Collecting race history to get smarter…")

# ===================== TOOLS =====================
@tool
def get_next_meeting() -> str:
    scraper = cloudscraper.create_scraper()
    html = scraper.get("https://racing.hkjc.com/racing/information/english/Racing/Fixture.aspx").text
    return "Happy Valley" if "Happy Valley" in html else "Sha Tin"

@tool
def scrape_races() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://bet.hkjc.com/racing/pages/odds_wp.aspx?date={today}&venue=HV"
    scraper = cloudscraper.create_scraper()
    return scraper.get(url).text[:4000]

@tool
def get_tips() -> str:
    from tavily import TavilyClient
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        query = f"Happy Valley tips {datetime.now().strftime('%d %b %Y')}"
        res = client.search(query, max_results=5)
        return "\n".join([r["content"][:600] for r in res["results"]])
    except:
        return "No fresh tips (will use form only)"

# ===================== AGENTS =====================
researcher = Agent(role="Researcher", goal="Get racecard + tips", tools=[get_next_meeting, scrape_races, get_tips], llm=llm)
analyst    = Agent(role="Form Analyst", goal="Rank horses", llm=llm)
bettor     = Agent(role="Professional Punter", goal="Build $1500–$2500 HKD card with Win/Place/Quinella/Trio", llm=llm)

crew = Crew(
    agents=[researcher, analyst, bettor],
    tasks=[
        Task(description="Collect tonight’s Happy Valley data", expected_output="Raw data", agent=researcher),
        Task(description="Deep form & track analysis", expected_output="Top horses", agent=analyst),
        Task(description="Create final betting card (Win, Place, Quinella, Trio) with reasoning", expected_output="Markdown card", agent=bettor)
    ],
    process=Process.sequential,
    verbose=2
)

# ===================== SEND RESULT =====================
def send(card):
    message = f"Happy Valley AI Picks – {datetime.now().strftime('%d %b %Y')}\n\n{card}\n\nBankroll: HK${bankroll['balance_hkd']:,}"
    
    # Telegram (recommended)
    if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        bot = telegram.Bot(token=os.getenv("TELEGRAM_TOKEN"))
        bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=message, parse_mode="Markdown")
        print("Sent to Telegram")
    
    # Email fallback
    elif os.getenv("EMAIL_TO"):
        import yagmail
        yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        yag.send(os.getenv("EMAIL_TO"), "Happy Valley AI Picks", message)
        print("Sent to email")

# ===================== MAIN =====================
def run_ai():
    print(f"AI waking up – {datetime.now().strftime('%H:%M')} Hong Kong time")
    if datetime.now().weekday() != 2:  # not Wednesday
        print("Not Wednesday – sleeping…")
        return
    result = crew.kickoff()
    send(result)
    train_value_model()

if __name__ == "__main__":
    # Run immediately if you want to test right now
    run_ai()
    
    # Then schedule forever
    schedule.every().wednesday.at("16:30").do(run_ai)
    schedule.every().day.at("11:00").do(train_value_model)
    
    print("Happy Valley AI is now running forever (free tier)!")
    while True:
        schedule.run_pending()
        time.sleep(60)
