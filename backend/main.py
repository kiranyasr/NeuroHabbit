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

# Force load .env from the same directory as this script
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="NeuroHabit Engine", version="2.0.0")

# 🛡️ CORS Handshake - Critical for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🔑 CREDENTIALS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Pre-flight Console Diagnostics
print("--- 🔍 SYSTEM DIAGNOSTICS ---")
if not GROQ_API_KEY: print("❌ ERROR: GROQ_API_KEY missing")
if not SUPABASE_URL: print("❌ ERROR: SUPABASE_URL missing")
print("✅ Services Initialized")
print("----------------------------")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 🧠 AI MODELS & AUTHENTICATION
# ==========================================

async def get_current_user(authorization: str = Header(None)):
    """Verifies the User Identity via Supabase JWT Token from the Frontend"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Identity Token Required")
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid Session")
        return user.user.id
    except Exception as e:
        print(f"Auth Failure: {e}")
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

# Load AI Brain State
model = HabitLSTM()
try:
    model_path = "models/habit_lstm_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("✅ LSTM Model Loaded")
except Exception as e:
    print(f"ℹ️ LSTM Mode: Heuristic Backup active ({e})")

# ==========================================
# 📋 INPUT SCHEMAS
# ==========================================

class HabitInput(BaseModel):
    text: str

class RegistryInput(BaseModel):
    name: str

# ==========================================
# 🚀 API ROUTES
# ==========================================

@app.get("/daily-status")
async def get_daily_status(user_id: str = Depends(get_current_user)):
    """Compares Master Habits vs Today's Logs to build the checklist"""
    try:
        # 1. Fetch defined habits (Your Goals)
        master_res = supabase.table("habit_definitions").select("name").eq("user_id", user_id).execute()
        master_habits = [m['name'].upper() for m in master_res.data]

        # 2. Fetch logs created today (Your Actions)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        logs_res = supabase.table("habits").select("activity").eq("user_id", user_id).gte("created_at", today_start).execute()
        
        logged_today = {l['activity'].upper() for l in logs_res.data}

        # 3. Build Checklist
        return [{"name": name, "done": name in logged_today} for name in master_habits]
    except Exception as e:
        print(f"Checklist Error: {e}")
        return []

@app.post("/define-habit")
async def define_habit(habit: RegistryInput, user_id: str = Depends(get_current_user)):
    """Registers a 'Master Habit' that the user wants to track daily"""
    try:
        data = {"name": habit.name.upper(), "user_id": user_id}
        supabase.table("habit_definitions").insert(data).execute()
        return {"status": "Goal Registered", "habit": habit.name.upper()}
    except Exception as e:
        print(f"Define Error: {e}")
        raise HTTPException(status_code=500, detail="Could not save habit definition")

@app.post("/add-habit")
async def add_habit(user_input: HabitInput, user_id: str = Depends(get_current_user)):
    """Analyzes text via Groq, evaluates risk, and saves to database"""
    try:
        # 1. NLP Extraction via Groq
        prompt = f"System: Extract 'activity' and 'duration' (integer minutes, default 30) from: '{user_input.text}'. Return ONLY JSON."
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(completion.choices[0].message.content)
        activity_name = str(extracted.get("activity", "UNKNOWN")).upper()
        
        # 2. Heuristic Risk Score (Example logic)
        history = supabase.table("habits").select("*").eq("activity", activity_name).eq("user_id", user_id).execute()
        risk_score = 15 if len(history.data) >= 3 else 88 
        
        # 3. Commit to Supabase
        data = {
            "activity": activity_name, 
            "duration": extracted.get("duration", 30), 
            "risk_score": risk_score, 
            "user_id": user_id
        }
        supabase.table("habits").insert(data).execute()
        
        return {"status": "Success", "saved_data": data, "prediction": f"{risk_score}.00%"}
    except Exception as e:
        print(f"Log Error: {e}")
        raise HTTPException(status_code=500, detail="Neural Link Failure")

@app.get("/activity-heatmap")
async def get_activity_heatmap(user_id: str = Depends(get_current_user)):
    """Returns 24-hour activity distribution"""
    try:
        res = supabase.table("habits").select("created_at").eq("user_id", user_id).execute()
        heatmap = {i: 0 for i in range(24)}
        for r in res.data:
            dt = datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))
            heatmap[dt.hour] += 1
        return [{"hour": h, "count": c} for h, c in heatmap.items()]
    except Exception as e:
        return [{"hour": i, "count": 0} for i in range(24)]

@app.get("/habit-streak")
async def get_streak(user_id: str = Depends(get_current_user)):
    """Calculates active daily streak"""
    try:
        res = supabase.table("habits").select("created_at").eq("user_id", user_id).execute()
        dates = sorted({datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')).date() for r in res.data}, reverse=True)
        if not dates: return {"streak": 0, "level": "RECRUIT"}
        
        today = datetime.utcnow().date()
        streak, curr_date = 0, dates[0]
        
        if dates[0] >= today - timedelta(days=1):
            for d in dates:
                if d == curr_date:
                    streak += 1
                    curr_date -= timedelta(days=1)
                else: break
        
        level = "COMMANDER" if streak >= 7 else "INITIATE" if streak >= 3 else "RECRUIT"
        return {"streak": streak, "level": level}
    except Exception:
        return {"streak": 0, "level": "OFFLINE"}

@app.get("/neural-performance-score")
async def get_nps(user_id: str = Depends(get_current_user)):
    """Overall efficiency score"""
    try:
        res = supabase.table("habits").select("id").eq("user_id", user_id).execute()
        score = min(len(res.data) * 5 + 20, 100)
        return {"nps_score": score, "rating": "OPTIMAL" if score > 75 else "STABLE"}
    except:
        return {"nps_score": 0, "rating": "ERROR"}

@app.get("/get-habits")
async def get_habits(user_id: str = Depends(get_current_user)):
    """Fetches log history for the Bar Chart"""
    res = supabase.table("habits").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(15).execute()
    return {"habits": res.data}

@app.get("/system-health")
async def system_health():
    return {"status": "OPERATIONAL", "latency": "LOW"}

@app.delete("/delete-habit/{habit_name}")
async def delete_habit(habit_name: str, user_id: str = Depends(get_current_user)):
    """Removes a permanent habit from the user's checklist"""
    try:
        supabase.table("habit_definitions")\
            .delete()\
            .eq("user_id", user_id)\
            .eq("name", habit_name.upper())\
            .execute()
        return {"status": "Goal Terminated"}
    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete habit")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)