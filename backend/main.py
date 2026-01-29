import os
import torch
import torch.nn as nn
import json
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

# --- 🔑 CREDENTIALS (REPLACE WITH YOUR ACTUAL KEYS) ---
GROQ_API_KEY="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"
SUPABASE_URL="https://onchutdrarerfnnyqnsf.supabase.co"
SUPABASE_KEY="sb_publishable_-1flN3MsoDo19vWvU1fo-A_WAt09U-H"

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
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x)
        # We only care about the AI's prediction after the LAST (7th) day
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load the trained .pth file from Day 10
try:
    model = HabitLSTM()
    # Ensure this path matches where you saved the file in VS Code
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
        # 1. AI Parsing (Groq) - Updated to a supported model
        prompt = f"Extract 'activity' and 'duration' (minutes) from: '{user_input.text}'. Return JSON only."
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # ✅ FIXED: Use a supported model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(completion.choices[0].message.content)
        
        # 2. Generate Prediction (LSTM)
        # Using a dummy sequence for inference (representing last 7 days)
        dummy_sequence = torch.randn(1, 7, 5) 
        with torch.no_grad():
            risk_prediction = model(dummy_sequence).item()
        
        risk_percentage = round(risk_prediction * 100)

        # 3. Save to Supabase (History Manifest)
        data = {
            "activity": extracted.get("activity", "UNKNOWN").upper(),
            "duration": extracted.get("duration", 0),
            "risk_score": risk_percentage
        }
        
        response = supabase.table("habits").insert(data).execute()
        
        return {
            "status": "Success",
            "saved_data": data,
            "prediction": f"{risk_percentage}%"
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
    # This endpoint provides the friction alerts seen in the UI dashboard
    return {
        "risks": [
            {"habit": "MORNING RUN", "risk": 54},
            {"habit": "DEEP WORK", "risk": 12}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # 🏛️ Clean line with no extra text after it
    uvicorn.run(app, host="0.0.0.0", port=8000)