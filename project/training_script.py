"""
Simple script to train the speech evaluation model
"""
import os
import json
from speech_evaluator import speech_evaluator

def main():
    print("Training the speech evaluation model...")
    
    # Check if training data exists
    if not os.path.exists("training_data.json"):
        print("Error: training_data.json not found")
        return
    
    # Load training data
    with open("training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training samples")
    
    # Extract speeches, scores, and topics
    speeches = [item["speech"] for item in training_data]
    scores = [item["score"] for item in training_data]
    
    if all("topic" in item for item in training_data):
        topics = [item["topic"] for item in training_data]
        print(f"Training with {len(set(topics))} unique topics")
        
        # Train with topics
        success = speech_evaluator.train_model(speeches, scores, topics)
    else:
        print("Training without topics")
        # Train without topics
        success = speech_evaluator.train_model(speeches, scores)
    
    if success:
        print("Model trained successfully and saved")
    else:
        print("Error training model")

if __name__ == "__main__":
    main() 