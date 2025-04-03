# MUN Speech Evaluator

An AI-powered tool for analyzing, evaluating, and providing feedback on Model United Nations (MUN) speeches.

## Features

- Automated evaluation of MUN speeches on a 1-10 scale
- Detailed feedback on content, structure, and language
- Analysis of key entities mentioned in speeches
- Specific improvement suggestions tailored to each speech
- Machine learning model trained on exemplary MUN speeches
- Web interface for easy speech submission and evaluation

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the required SpaCy model:
   ```
   python -m spacy download en_core_web_md
   ```

## Usage

### Web Interface

Start the web application:
```
python app.py
```

Then open your browser to http://localhost:5000 to use the web interface.

### Data Collection

To build a custom dataset of MUN speeches for training:
```
python collect_training_data.py
```

This interactive tool allows you to:
- Input MUN speeches
- Rate them on a scale of 1-10
- Add metadata like topic, type, and committee context
- Save all data to a JSON file for model training

### Model Training

To train the model on your collected data:
```
python train_model.py --data training_data.json --output mun_speech_evaluator.joblib
```

The training script:
- Splits data into training and testing sets
- Builds and trains the evaluator model
- Evaluates performance with metrics (MSE, RMSE, R²)
- Generates a visualization of actual vs. predicted scores
- Provides detailed analysis of prediction accuracy

### Programmatic Usage

```python
from main import MUNSpeechEvaluator

# Create an evaluator
evaluator = MUNSpeechEvaluator()

# If you have a pre-trained model:
# evaluator.load_model("mun_speech_evaluator.joblib")

# Or train a new model with labeled data:
evaluator.train(speech_samples, scores)

# Evaluate a new speech
speech = "Mr. Chair, distinguished delegates, our delegation believes..."
score = evaluator.evaluate(speech)
feedback = evaluator.generate_feedback(speech)

# Save your trained model
evaluator.save_model("mun_speech_evaluator.joblib")
```

## Model Features

The AI evaluates speeches based on:
- Content quality and relevance
- Speech structure and flow
- Language complexity and appropriateness
- Use of diplomatic language and MUN terminology
- Entity recognition for countries, organizations, and issues

## Web Interface

The web interface provides:
- A simple form to input MUN speeches
- Immediate feedback with an overall score
- Detailed analysis of content, structure, and language
- Specific suggestions for improvement
- Visual representation of results

## Future Improvements

- Integration with speech recognition for real-time feedback
- Expanded training dataset with speeches from international competitions
- Support for different types of MUN speeches (opening, moderated caucus, etc.)
- Advanced analytics dashboard showing speech strengths and weaknesses
