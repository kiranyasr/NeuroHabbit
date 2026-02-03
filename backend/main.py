import os
import torch
import torch.nn as nn
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq

# ==========================================
# 🛰️ SYSTEM & ENVIRONMENT INITIALIZATION
# ==========================================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="NeuroHabit Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 🧠 AI MODELS & AUTHENTICATION
# ==========================================

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Identity Token Required")
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid Session")
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Neural Verification Failed")

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

model = HabitLSTM()
# Model loading logic removed for brevity, stays as per your original script

# ==========================================
# 📋 INPUT SCHEMAS
# ==========================================

class HabitInput(BaseModel):
    text: str

class RegistryInput(BaseModel):
    name: str

class WeightInput(BaseModel):
    initial_weight: float
    current_weight: float

# ==========================================
# 🚀 API ROUTES (HABITS)
# ==========================================

@app.get("/daily-status")
async def get_daily_status(user_id: str = Depends(get_current_user)):
    try:
        master_res = supabase.table("habit_definitions").select("name").eq("user_id", user_id).execute()
        master_habits = [m['name'].upper() for m in master_res.data]
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        logs_res = supabase.table("habits").select("activity").eq("user_id", user_id).gte("created_at", today_start).execute()
        logged_today = {l['activity'].upper() for l in logs_res.data}
        return [{"name": name, "done": name in logged_today} for name in master_habits]
    except Exception as e:
        return []

# ... [Keep your define-habit and add-habit routes as they were] ...

# ==========================================
# ⚖️ API ROUTES (WEIGHT TRACKING)
# ==========================================

@app.post("/update-weight")
async def update_weight(data: WeightInput, user_id: str = Depends(get_current_user)):
    """Saves weight protocol to the database"""
    try:
        payload = {
            "user_id": user_id,
            "initial_weight": data.initial_weight,
            "current_weight": data.current_weight
        }
        res = supabase.table("weight_logs").insert(payload).execute()
        return {"status": "Metric Logged", "data": res.data}
    except Exception as e:
        print(f"Weight Update Error: {e}")
        raise HTTPException(status_code=500, detail="Database Sync Failed")

@app.get("/get-latest-weight")
async def get_latest_weight(user_id: str = Depends(get_current_user)):
    """Retrieves the most recent weight entry for the user"""
    try:
        res = supabase.table("weight_logs")\
            .select("initial_weight, current_weight")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        if not res.data:
            return {"initial_weight": None, "current_weight": None}
        return res.data[0]
    except Exception as e:
        print(f"Fetch Weight Error: {e}")
        return {"initial_weight": None, "current_weight": None}

# ==========================================
# 📈 SYSTEM METRICS & HEALTH
# ==========================================

@app.get("/system-health")
async def system_health():
    return {"status": "OPERATIONAL", "latency": "LOW"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)