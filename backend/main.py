import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from nlp_parser import parse_habit_text  # This is the AI script you updated
from dotenv import load_dotenv

# 1. Load keys from your .env file
load_dotenv()

app = FastAPI()

# 2. Allow your modern Frontend to talk to this Python code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Connect to your Supabase project
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Safety check: Prevent server start if keys are missing
if not url or not key:
    print("❌ Error: SUPABASE_URL or SUPABASE_KEY missing in .env!")
else:
    supabase: Client = create_client(url, key)

@app.post("/add-habit")
async def handle_habit_request(data: dict):
    """
    Receives text from your Light-Themed UI, 
    asks Llama 3 to parse it, and saves it to Supabase.
    """
    user_text = data.get("text")
    if not user_text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Step A: AI translates the human text into data
        structured_data = parse_habit_text(user_text)
        
        # Step B: Insert the data into your existing 'habits' table
        response = supabase.table("habits").insert({
            "name": structured_data.get("activity"),
            "duration": structured_data.get("duration"),
            "frequency": structured_data.get("frequency")
        }).execute()
        
        # Step C: Send the success response back to the UI
        return {
            "status": "Success",
            "saved_data": structured_data,
            "supabase_id": response.data[0]['id'] if response.data else None
        }

    except Exception as e:
        print(f"❌ Backend Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save habit to database.")