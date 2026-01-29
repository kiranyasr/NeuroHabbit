from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime, timedelta
from nlp_parser import parse_habit_text

load_dotenv()
app = FastAPI()

# 🛡️ FIX: This allows your Frontend to talk to the Backend safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

try:
    model = joblib.load("habit_predictor.pkl")
except:
    model = None

class HabitRequest(BaseModel):
    text: str

@app.post("/add-habit")
async def add_habit(request: HabitRequest):
    structured_data = parse_habit_text(request.text)
    try:
        # FIX: Sending 'name' to satisfy Supabase constraints
        supabase.table("habits").insert({
            "name": structured_data["activity"], 
            "activity": structured_data["activity"],
            "duration": structured_data["duration"],
            "frequency": structured_data["frequency"],
            "raw_text": request.text
        }).execute()
        return {"status": "Success", "saved_data": structured_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-risk")
async def predict_risk():
    if not model:
        return {"risks": []}
    tomorrow = datetime.now() + timedelta(days=1)
    day_num = tomorrow.weekday()
    # List of habits the AI studied in Day 3
    habit_names = ["Deep Work", "Morning Run", "Meditation"]
    risks = []
    for i, name in enumerate(habit_names):
        # FIX: Using DataFrame to stop the 'Feature Names' warning
        input_df = pd.DataFrame([[day_num, i]], columns=['day_num', 'habit_code'])
        probs = model.predict_proba(input_df)[0]
        risk_percent = round(probs[0] * 100) # Prob of failure
        risks.append({"habit": name, "risk": risk_percent})
    return {"tomorrow": tomorrow.strftime("%Y-%m-%d"), "risks": risks}

@app.get("/get-habits")
async def get_habits():
    # CRT: Pulling all habit records from Supabase for the History Manifest
    response = supabase.table("habits").select("*").order("created_at", desc=True).execute()
    return {"habits": response.data}

@app.get("/habit-analytics")
async def habit_analytics():
    # CRT: Fetching risk data from your ML model for the visual dashboard
    response = supabase.table("habits").select("activity, risk_score").limit(10).execute()
    # Format data for the Recharts component
    chart_data = [{"name": h['activity'], "risk": h['risk_score']} for h in response.data]
    return {"analytics": chart_data}