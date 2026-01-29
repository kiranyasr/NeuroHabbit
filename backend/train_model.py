import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_neural_engine():
    # 1. Load the "textbook" we made on Day 3
    df = pd.read_csv("habit_training_data.csv")
    
    # 2. Convert text to numbers so the AI understands
    # (AI doesn't know "Monday", it knows "0")
    df['day_num'] = pd.to_datetime(df['date']).dt.dayofweek
    df['habit_code'] = df['habit_name'].astype('category').cat.codes
    
    # 3. Pick what to learn (Features) and what to predict (Target)
    X = df[['day_num', 'habit_code']] # The Day and The Habit
    y = df['completed']               # Did they succeed? (0 or 1)
    
    # 4. Split data: 80% for studying, 20% for the "Final Exam"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 5. Initialize and Train the AI
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 6. Check the "Score" (How smart is it?)
    accuracy = model.score(X_test, y_test)
    print(f"✅ AI Training Complete! Accuracy: {accuracy * 100:.2f}%")
    
    # 7. Save the "Brain" to a file so we can use it later
    joblib.dump(model, "habit_predictor.pkl")
    print("🧠 Model saved as 'habit_predictor.pkl'")

if __name__ == "__main__":
    train_neural_engine()