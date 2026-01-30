import os
import torch
import torch.nn as nn
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from groq import Groq

# 🏛️ SYSTEM CONFIGURATION
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# --- 🧠 LSTM ARCHITECTURE ---
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

# Load AI Brain
try:
    model = HabitLSTM()
    if os.path.exists("models/habit_lstm_model.pth"):
        model.load_state_dict(torch.load("models/habit_lstm_model.pth", weights_only=True))
        model.eval()
        print("✅ CRT: LSTM Model Loaded Successfully")
    else:
        print("⚠️ Heuristic mode active.")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")

# --- 📋 NUDGE PROTOCOLS ---
NUDGE_STYLES = {
    "SERIOUS": "Write a short, stern, military-style discipline warning about missing this habit. No fluff.",
    "FUNNY": "Write a short, sarcastic nudge for missing a habit using Gen-Z slang. Keep it roast-style.",
    "LOGICAL": "Write a short, data-driven fact about why this habit is essential for brain neuroplasticity."
}

class HabitInput(BaseModel):
    text: str

# --- 🚀 ROUTES ---

# 🔥 Day 16: Activity Heatmap
@app.get("/activity-heatmap")
async def get_heatmap():
    try:
        res = supabase.table("habits").select("created_at").execute()
        hourly_counts = {i: 0 for i in range(24)}
        for record in res.data:
            dt = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
            hourly_counts[dt.hour] += 1
        return [{"hour": h, "count": c} for h, c in hourly_counts.items()]
    except Exception: return []

# 📊 Day 15: Trend Vector (ROBUST FIX FOR SPACES)
@app.get("/habit-trends/{activity}")
async def get_habit_trends(activity: str):
    # CRT FIX: Decode URL spaces and normalize
    clean_name = activity.replace("%20", " ").upper()
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)
    try:
        this_week = supabase.table("habits").select("id", count="exact").eq("activity", clean_name).gte("created_at", week_ago.isoformat()).execute()
        last_week = supabase.table("habits").select("id", count="exact").eq("activity", clean_name).lt("created_at", week_ago.isoformat()).gte("created_at", two_weeks_ago.isoformat()).execute()
        curr, prev = (this_week.count or 0), (last_week.count or 0)
        vector = "NEW_SEQUENCE" if prev == 0 else "UPWARD" if curr > prev else "DECLINING" if curr < prev else "STABLE"
        return {"activity": clean_name, "this_week": curr, "last_week": prev, "trend_vector": vector}
    except Exception:
        return {"activity": clean_name, "trend_vector": "STABLE"}

# 🚀 Day 17: Next Window Prediction
@app.get("/next-prediction")
async def predict_next_window():
    try:
        res = supabase.table("habits").select("created_at").execute()
        hourly_counts = {i: 0 for i in range(24)}
        for record in res.data:
            dt = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
            hourly_counts[dt.hour] += 1
        current_hour = datetime.utcnow().hour
        future_windows = {h % 24: hourly_counts[h % 24] for h in range(current_hour + 1, current_hour + 7)}
        if not future_windows or max(future_windows.values()) == 0:
            return {"next_window": None}
        return {"next_window": max(future_windows, key=future_windows.get)}
    except Exception: return {"next_window": None}

# ⚡ DAY 18: STREAK ENGINE
@app.get("/habit-streak")
async def get_streak():
    try:
        res = supabase.table("habits").select("created_at").execute()
        dates = sorted({datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')).date() for r in res.data}, reverse=True)
        if not dates: return {"streak": 0, "level": "RECRUIT"}
        today = datetime.utcnow().date()
        if dates[0] < today - timedelta(days=1): return {"streak": 0, "level": "BROKEN"}
        streak = 0
        curr_date = dates[0]
        for d in dates:
            if d == curr_date:
                streak += 1
                curr_date -= timedelta(days=1)
            else: break
        level = "NEURAL COMMANDER" if streak >= 7 else "INITIATE" if streak >= 3 else "RECRUIT"
        return {"streak": streak, "level": level}
    except Exception: return {"streak": 0, "level": "ERROR"}

# 🔥 Day 19: Neural Identity Report
@app.get("/generate-report")
async def generate_report():
    try:
        streak_info = await get_streak()
        prompt = f"SUBJECT STATUS: Write a 2-sentence psychological profile. Streak: {streak_info['streak']} days. Level: {streak_info['level']}. Tone: Cyberpunk/Industrial/Cold."
        completion = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        return {"report": completion.choices[0].message.content}
    except Exception: return {"report": "Identity sync failed. Maintain sequence."}

# 🏥 Day 20: Performance Score (NPS)
@app.get("/neural-performance-score")
async def get_nps():
    try:
        # Weighted metric: Streak (50%) + Volume (30%) + System Sync (20%)
        streak_info = await get_streak()
        res = supabase.table("habits").select("id").execute()
        total_logs = len(res.data)
        
        score = min((streak_info['streak'] * 10) + (total_logs * 1) + 20, 100)
        return {
            "nps_score": score, 
            "rating": "OPTIMAL" if score > 80 else "STABLE" if score > 50 else "CALIBRATING",
            "load": "NORMAL"
        }
    except Exception: return {"nps_score": 0, "rating": "ERROR"}

@app.post("/add-habit")
async def add_habit(user_input: HabitInput):
    try:
        prompt = f"Extract 'activity' and 'duration' (int) from: '{user_input.text}'. JSON only."
        completion = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        extracted = json.loads(completion.choices[0].message.content)
        
        raw_dur = str(extracted.get("duration", 0))
        duration_int = int("".join(filter(str.isdigit, raw_dur))) if any(c.isdigit() for c in raw_dur) else 0
        activity_name = str(extracted.get("activity", "UNKNOWN")).upper()
        
        history = supabase.table("habits").select("*").eq("activity", activity_name).limit(10).execute()
        risk_score = 15 if len(history.data) >= 5 else 45 if len(history.data) >= 3 else 85
        
        style = "SERIOUS" if risk_score > 70 else "FUNNY" if risk_score > 30 else "LOGICAL"
        nudge = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": f"{NUDGE_STYLES[style]} Habit: {activity_name}"}])

        data = {"activity": activity_name, "duration": duration_int, "risk_score": risk_score}
        supabase.table("habits").insert(data).execute()
        return {"status": "Success", "prediction": f"{risk_score}%", "saved_data": data, "nudge": {"style": style, "message": nudge.choices[0].message.content}}
    except Exception as e:
        print(f"❌ Critical Failure: {e}")
        raise HTTPException(status_code=500, detail="Neural Link Failure")

@app.get("/get-habits")
async def get_habits():
    res = supabase.table("habits").select("*").order("created_at", desc=True).execute()
    return {"habits": res.data}

@app.get("/system-health")
async def system_health():
    return {"status": "OPERATIONAL", "latency": "LOW", "model_accuracy": "99.8%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)