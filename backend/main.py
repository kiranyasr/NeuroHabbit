import os
import torch
import torch.nn as nn
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq

# 🏛️ SYSTEM CONFIGURATION
app = FastAPI(title="NeuroHabit Engine", version="2.0.0")

# 🛡️ CORS Handshake - Critical for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# --- 🔑 CREDENTIALS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 🧠 AUTHENTICATION DEPENDENCY ---
async def get_current_user(authorization: str = Header(None)):
    """Verifies the User Identity via Supabase JWT Token from the Frontend"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Identity Token Required")
    try:
        # Extract Bearer token
        token = authorization.replace("Bearer ", "")
        # Validate with Supabase Auth
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid Session")
        return user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Neural Verification Failed")

# --- 🧠 LSTM ARCHITECTURE (AI Brain) ---
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
try:
    model = HabitLSTM()
    if os.path.exists("models/habit_lstm_model.pth"):
        model.load_state_dict(torch.load("models/habit_lstm_model.pth", weights_only=True))
        model.eval()
        print("✅ CRT: LSTM Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ AI Model Load Error: {e}")

# --- 📋 INPUT MODELS ---
class HabitInput(BaseModel):
    text: str

class RegistryInput(BaseModel):
    name: str

# --- 🚀 ROUTES (Day 24: 7-Day Efficiency Stats) ---

@app.get("/daily-progress-stats")
async def get_progress_stats(user_id: str = Depends(get_current_user)):
    """Calculates completion percentage of registered habits for the last 7 days"""
    try:
        # 1. Fetch Registered Goals for the user
        registry = supabase.table("habit_registry").select("habit_name").eq("user_id", user_id).execute()
        goal_names = [goal['habit_name'] for goal in registry.data]
        goal_count = len(goal_names)
        
        if goal_count == 0:
            return []

        # 2. Fetch logs for the last 7 days
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        logs_res = supabase.table("habits").select("activity, created_at").eq("user_id", user_id).gte("created_at", seven_days_ago).execute()
        
        stats = []
        # 3. Iterate through each of the last 7 days to calculate efficiency
        for i in range(7):
            target_date = (datetime.utcnow() - timedelta(days=i)).date()
            
            # Count unique registered habits completed on this target day
            completed_on_day = {
                log['activity'] for log in logs_res.data 
                if datetime.fromisoformat(log['created_at'].replace('Z', '+00:00')).date() == target_date
                and log['activity'] in goal_names
            }
            
            # Efficiency Score = (Done / Total Goals) * 100
            success_rate = (len(completed_on_day) / goal_count) * 100
            stats.append({
                "day": target_date.strftime("%a"), 
                "score": round(success_rate)
            })
            
        return stats[::-1] # Sort Chronologically for Recharts
    except Exception as e:
        print(f"Stats Calculation Error: {e}")
        return []

# --- 🚀 ROUTES (Day 23: Neural Habit Registry) ---

@app.post("/register-habit")
async def register_habit(goal: RegistryInput, user_id: str = Depends(get_current_user)):
    """Creates a permanent target in the habit_registry table"""
    try:
        data = {"habit_name": goal.name.upper(), "user_id": user_id}
        supabase.table("habit_registry").insert(data).execute()
        return {"status": "REGISTERED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registry Error: {str(e)}")

@app.get("/get-registry")
async def get_registry(user_id: str = Depends(get_current_user)):
    """Fetches checklist goals for the frontend UI"""
    res = supabase.table("habit_registry").select("*").eq("user_id", user_id).execute()
    return {"registry": res.data}

@app.delete("/unregister-habit/{habit_id}")
async def unregister_habit(habit_id: str, user_id: str = Depends(get_current_user)):
    """Removes a permanent goal"""
    supabase.table("habit_registry").delete().eq("id", habit_id).eq("user_id", user_id).execute()
    return {"status": "DELETED"}

# --- 🚀 LOGGING & AI ANALYSIS ROUTES ---

@app.post("/add-habit")
async def add_habit(user_input: HabitInput, user_id: str = Depends(get_current_user)):
    """Analyzes text with Groq, evaluates risk, and saves to habits table"""
    try:
        # NLP Extraction with Llama 3
        prompt = f"Extract 'activity' and 'duration' (int, default 0) from: '{user_input.text}'. Return JSON only."
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        extracted = json.loads(completion.choices[0].message.content)
        activity_name = str(extracted.get("activity", "UNKNOWN")).upper()
        
        # Simple Risk Logic (Can be replaced by LSTM inference)
        history = supabase.table("habits").select("*").eq("activity", activity_name).eq("user_id", user_id).limit(5).execute()
        risk_score = 20 if len(history.data) >= 3 else 80 
        
        data = {
            "activity": activity_name, 
            "duration": extracted.get("duration", 0), 
            "risk_score": risk_score, 
            "user_id": user_id
        }
        supabase.table("habits").insert(data).execute()
        return {"status": "Success", "saved_data": data}
    except Exception as e:
        print(f"Log Error: {e}")
        raise HTTPException(status_code=500, detail="Neural Link Failure")

@app.get("/habit-streak")
async def get_streak(user_id: str = Depends(get_current_user)):
    """Calculates consecutive days of activity"""
    try:
        res = supabase.table("habits").select("created_at").eq("user_id", user_id).execute()
        dates = sorted({datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')).date() for r in res.data}, reverse=True)
        
        if not dates: return {"streak": 0, "level": "RECRUIT"}
        
        today = datetime.utcnow().date()
        if dates[0] < today - timedelta(days=1): return {"streak": 0, "level": "BROKEN"}
        
        streak, curr_date = 0, dates[0]
        for d in dates:
            if d == curr_date:
                streak += 1
                curr_date -= timedelta(days=1)
            else: break
            
        level = "NEURAL COMMANDER" if streak >= 7 else "INITIATE" if streak >= 3 else "RECRUIT"
        return {"streak": streak, "level": level}
    except Exception: return {"streak": 0, "level": "SYSTEM_OFFLINE"}

@app.get("/neural-performance-score")
async def get_nps(user_id: str = Depends(get_current_user)):
    """Calculates overall system performance based on volume"""
    res = supabase.table("habits").select("id").eq("user_id", user_id).execute()
    score = min(len(res.data) * 5 + 20, 100)
    return {"nps_score": score, "rating": "OPTIMAL" if score > 75 else "STABLE"}

@app.get("/get-habits")
async def get_habits(user_id: str = Depends(get_current_user)):
    """Fetches the raw log history"""
    res = supabase.table("habits").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return {"habits": res.data}

@app.delete("/purge-data")
async def purge_data(confirm: bool = False, user_id: str = Depends(get_current_user)):
    """Sanitizes user history"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required.")
    supabase.table("habits").delete().eq("user_id", user_id).execute()
    return {"status": "WIPE_COMPLETE"}

@app.get("/system-health")
async def system_health():
    return {"status": "OPERATIONAL", "latency": "LOW"}

if __name__ == "__main__":
    import uvicorn
    # Make sure port matches your frontend fetch calls
    uvicorn.run(app, host="0.0.0.0", port=8000)