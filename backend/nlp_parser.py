import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def parse_habit_text(user_input):
    prompt = f"""
    Extract the following from the text: activity, duration (in minutes), frequency.
    Text: "{user_input}"
    Return ONLY a JSON object. Example: {{"activity": "Reading", "duration": 30, "frequency": "daily"}}
    """
    completion = client.chat.completions.create(
       model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)