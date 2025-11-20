# main.py – Your Winning Machine (Happy Valley AI)
import os, json, time, schedule
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import cloudscraper
from bs4 import BeautifulSoup
from tavily import TavilyClient
import telegram
from dotenv import load_dotenv

load_dotenv()  # reads .env file

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Tools
@tool
def get_races_today() -> str:
    scraper = cloudscraper.create_scraper()
    url = f"https://bet.hkjc.com/racing/pages/odds_wp.aspx?date={datetime.now():%Y-%m-%d}&venue=HV"
    return scraper.get(url).text[:5000]

@tool
def get_expert_tips() -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"Happy Valley tips {datetime.now().strftime('%d %b %Y')}"
    res = client.search(query, max_results=6)
    return "\n".join([r["content"][:800] for r in res["results"]])

# Agents
researcher = Agent(role="Data Collector", goal="Get racecard + tips", tools=[get_races_today, get_expert_tips], llm=llm)
analyst    = Agent(role="Form Expert", goal="Rank horses", llm=llm)
bettor     = Agent(role="Pro Punter", goal="Build $2000 HKD card with Win/Place/Quinella/Trio", llm=llm)

crew = Crew(
    agents=[researcher, analyst, bettor],
    tasks=[
        Task(description="Collect tonight’s Happy Valley data + tips", expected_output="All data", agent=researcher),
        Task(description="Deep analysis of every race", expected_output="Top picks", agent=analyst),
        Task(description="Final betting card – Win, Place, Quinella, Trio – with reasoning", expected_output="Markdown card", agent=bettor)
    ],
    process=Process.sequential,
    verbose=2
)

def send_to_telegram(card):
    bot = telegram.Bot(token=os.getenv("TELEGRAM_TOKEN"))
    message = f"Happy Valley Picks – {datetime.now().strftime('%d %b')}\n\n{card}"
    bot.send_message(chat_id=os.getenv("TELE Gram_CHAT_ID"), text=message, parse_mode="Markdown")

def run():
    if datetime.now().weekday() != 2:  # 0=Mon, 2=Wed
        print("Not Wednesday – sleeping")
        return
    print("WEDNESDAY! Running the AI...")
    result = crew.kickoff()
    send_to_telegram(result)

if __name__ == "__main__":
    run()  # runs once now so you see it works
    schedule.every().wednesday.at("16:30").do(run)
    print("Winning Machine is ALIVE forever!")
    while True:
        schedule.run_pending()
        time.sleep(60)
