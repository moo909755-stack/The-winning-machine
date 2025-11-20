# main.py – The Winning Machine (Happy Valley → Email only)
import os, json, time, schedule
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import cloudscraper
from bs4 import BeautifulSoup
from tavily import TavilyClient
import yagmail
from dotenv import load_dotenv

load_dotenv()  # reads .env file

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# ===================== TOOLS =====================
@tool
def get_races_today() -> str:
    scraper = cloudscraper.create_scraper()
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://bet.hkjc.com/racing/pages/odds_wp.aspx?date={today}&venue=HV"
    return scraper.get(url).text[:6000]

@tool
def get_expert_tips() -> str:
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        query = f"Happy Valley horse racing tips {datetime.now().strftime('%d %B %Y')}"
        res = client.search(query, max_results=6)
        return "\n\n".join([r["content"][:900] for r in res["results"]])
    except:
        return "No external tips found – using form only."

# ===================== AGENTS =====================
researcher = Agent(
    role="Racing Researcher",
    goal="Collect racecard, odds and expert tips",
    tools=[get_races_today, get_expert_tips],
    llm=llm
)

analyst = Agent(
    role="Form & Track Analyst",
    goal="Rank every horse in every race",
    llm=llm
)

bettor = Agent(
    role="Professional Punter",
    goal="Create a $1800–$2500 HKD betting card (Win, Place, Quinella, Trio)",
    llm=llm
)

crew = Crew(
    agents=[researcher, analyst, bettor],
    tasks=[
        Task(description="Collect tonight’s Happy Valley data + tips", expected_output="Full data", agent=researcher),
        Task(description="Deep analysis of every race and horse", expected_output="Ranked contenders", agent=analyst),
        Task(description="Final betting card with stakes, reasoning and bet types", expected_output="Beautiful markdown card", agent=bettor)
    ],
    process=Process.sequential,
    verbose=2
)

# ===================== SEND BY EMAIL =====================
def send_by_email(card):
    try:
        yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        subject = f"Winning Machine – Happy Valley Picks {datetime.now().strftime('%d %b %Y')}"
        body = f"Your AI just ran!\n\n{card}\n\nBankroll will be added soon."
        yag.send(os.getenv("EMAIL_TO"), subject, body)
        print("Picks successfully sent to email!")
    except Exception as e:
        print("Email failed:", e)

# ===================== MAIN =====================
def run_wednesday():
    if datetime.now().weekday() != 2:  # 2 = Wednesday
        print("Not Wednesday – sleeping")
        return
    print("IT'S WEDNESDAY! Running the Winning Machine...")
    result = crew.kickoff()
    send_by_email(result)

if __name__ == "__main__":
    run_wednesday()                                   # runs once now (test)
    schedule.every().wednesday.at("16:30").do(run_wednesday)
    schedule.every().day.at("11:00").do(lambda: print("Daily health check"))
    print("The Winning Machine is now LIVE forever (email version)!")
    while True:
        schedule.run_pending()
        time.sleep(60)
