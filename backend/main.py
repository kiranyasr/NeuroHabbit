import os
import torch
import torch.nn as nn
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq

# 🏛️ SYSTEM CONFIGURATION
app = FastAPI()

# Enable CORS for frontend communication (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv() # This loads the variables from .env

# CRT: Accessing keys securely via Environment Variables
# --- 🔑 CREDENTIALS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 🧠 LSTM ARCHITECTURE (Must match Day 8 Colab) ---
class HabitLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(HabitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load the trained .pth file from Day 10
try:
    model = HabitLSTM()
    model.load_state_dict(torch.load("models/habit_lstm_model.pth", weights_only=True))
    model.eval()
    print("✅ CRT: LSTM Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")

# --- 📋 SCHEMAS ---
class HabitInput(BaseModel):
    text: str

# --- 🚀 ROUTES ---

@app.post("/add-habit")
async def add_habit(user_input: HabitInput):
    try:
        # 1. AI Parsing (Groq)
        prompt = f"Extract 'activity' and 'duration' (minutes) from: '{user_input.text}'. Return JSON only."
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(completion.choices[0].message.content)
        activity_name = extracted.get("activity", "UNKNOWN").upper()
        
        # 2. Day 11 Frequency Logic (Thompson Sampling Baseline)
        # Fetch last 7 days of this specific habit to calculate "Friction"
        history_check = supabase.table("habits").select("*").eq("activity", activity_name).limit(7).execute()
        frequency = len(history_check.data)
        
        # Behavioral Logic: High frequency = Low Risk
        if frequency >= 5:
            risk_score = 15  # Stable
        elif frequency >= 3:
            risk_score = 45  # Moderate
        else:
            risk_score = 85  # High Risk (Lapse Likely)

        # 3. Save to Supabase (History Manifest)
        data = {
            "activity": activity_name,
            "duration": extracted.get("duration", 0),
            "risk_score": risk_score
        }
        
        supabase.table("habits").insert(data).execute()
        
        return {
            "status": "Success",
            "saved_data": data,
            "prediction": f"{risk_score}%"
        }
    except Exception as e:
        print(f"❌ Error in /add-habit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-habits")
async def get_habits():
    # CRT: Pulling all habit records for the UI Table
    try:
        response = supabase.table("habits").select("*").order("created_at", desc=True).execute()
        return {"habits": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-risk")
async def predict_risk():
    # Provides the friction alerts for the HUD
    return {
        "risks": [
            {"habit": "MORNING RUN", "risk": 54},
            {"habit": "DEEP WORK", "risk": 12}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)