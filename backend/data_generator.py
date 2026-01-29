import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_behavioral_data():
    # 1. Pick a start date (about 5 years ago to get 5000 rows)
    start_date = datetime.now() - timedelta(days=1666) 
    data = []
    
    # 2. Define our habits and how hard they are
    habits = [
        {"name": "Deep Work", "base_success": 0.8},
        {"name": "Morning Run", "base_success": 0.6},
        {"name": "Meditation", "base_success": 0.5}
    ]

    print("Generating 5,000 rows of human behavior...")

    for i in range(5000):
        # Pick which habit we are talking about for this row
        current_date = start_date + timedelta(days=i // 3)
        habit = habits[i % 3]
        day_of_week = current_date.weekday() # 0 is Monday, 6 is Sunday
        
        # 🧪 The "Lazy Logic" (Failure Patterns)
        success_prob = habit["base_success"]
        
        # Make it harder on weekends!
        if day_of_week >= 5:
            success_prob -= 0.20
            
        # Decide if the "fake person" finished the habit
        completed = 1 if np.random.random() < success_prob else 0
        
        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "habit_name": habit["name"],
            "completed": completed,
            "day_type": "Weekend" if day_of_week >= 5 else "Weekday"
        })

    # 3. Save it to a file
    df = pd.DataFrame(data)
    df.to_csv("habit_training_data.csv", index=False)
    print("✅ All done! I created 'habit_training_data.csv' in your backend folder.")

if __name__ == "__main__":
    generate_behavioral_data()