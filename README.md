# MUN Speech Analyzer

## Features

- **Instant Scoring** - Get a quality score from 1-10 for your speech
- **Smart Analysis** - Detailed breakdown of content, structure, and language
- **Entity Recognition** - Identifies countries, organizations, and policies
- **Coherence Check** - Evaluates logical flow between ideas
- **Actionable Tips** - Personalized suggestions to improve your speech
- **Voice Analysis** - Analyzes recordings of speeches for modulation, pace, and clarity

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Individual-Project.git
cd Individual-Project

# If you are using github to access the project you will need to download the dataset from: https://www.kaggle.com/datasets/unitednations/un-general-debates
# If you are using github to access the project you will need to train the model first, as model files are too large to commit to GitHub.

# Install FFmpeg (required for voice analysis)
# Windows: 
# 1. Download from https://www.gyan.dev/ffmpeg/builds/
# 2. Download the ffmpeg-git-full.7z version
# 3. Extract the downloaded archive
# 4. Add the bin folder to your system PATH

# Linux:
sudo apt update && sudo apt install ffmpeg

# Run the setup script
python setup.py
```

## Quick Start

```bash
# Ensure you are in the project directory
cd project
python main.py

# Access the web interface in your browser
# http://localhost:5000
```

## Usage Guide

### Text Analysis

The text analysis component evaluates written speeches based on:

- Content relevance and depth
- Structural coherence
- Language quality and vocabulary usage
- Policy suggestions and resolution alignment

#### Using Text Analysis

1. **Type your speech** or copy and paste into the text box
2. **Select a topic** (optional)
3. **Click "Analyze"**
4. **Review detailed feedback** on your speech

### Voice Analysis

The voice analysis component evaluates speech recordings based on:

- Voice modulation and pitch variation
- Speech rate (words per minute)
- Strategic pauses and emphasis
- Fluency and stuttering detection
- Clarity and pronunciation
- Vocabulary complexity
- Confidence markers
- Emotional tone

#### Using Voice Analysis

1. **Record your speech** or upload an audio file
2. **Click "Analyze"**
3. **Review detailed feedback** on your speaking performance


## Technical Details

### Architecture

The application uses a combination of natural language processing and audio analysis techniques:

- **NLP Pipeline**: Transformer-based models analyze speech content, structure, and language
- **Audio Processing**: Advanced audio analysis tools evaluate vocal delivery and presentation
- **Web Interface**: Flask-based web application for easy access to analysis tools

### Training & Configuration

The text analysis model uses transfer learning to achieve higher accuracy for MUN speech evaluation:

1. **Base Model Training**: A transformer-based (BERT) model is pretrained on a large corpus of UN General Debates speeches to learn diplomatic language patterns
2. **Fine-tuning**: The base model is then fine-tuned on specialized MUN speech datasets to adapt to MUN-specific characteristics
3. **Performance Improvement**: This approach yields lower error rates and better handling of MUN-specific language compared to models trained solely on MUN data

To train the model yourself:
```bash
python run_transfer_learning.py --un_data_path un_general_debates.csv --mun_data_path training_data.json
```

### Supported Audio Formats

- WAV
- MP3
- M4A

## Project Structure

```
Individual-Project/
├── project/                  # Main application directory
│   ├── main.py               # Application entry point
│   ├── templates/            # Web interface templates
│   │   ├── index.html        # Main page for text analysis
│   │   └── voice_analysis.html # Voice analysis interface
│   ├── static/               # Static assets for web interface
│   │   ├── css/              # Stylesheets
│   │   └── js/               # JavaScript files
│   ├── uploads/              # Directory for uploaded audio files
│   ├── output/               # Analysis output files
│   ├── voice_analyzer.py     # Voice analysis functionality
│   ├── text_processor.py     # Text processing functionality
│   ├── speech_evaluator.py   # Main speech evaluation logic
│   ├── evaluation.py         # Evaluation metrics and scoring
│   ├── feedback_generation.py # Feedback creation functionality
│   ├── feature_extraction.py # Extract features from text/audio
│   ├── keyword_analysis.py   # Keyword and entity recognition
│   ├── data_processor.py     # Data preprocessing utilities
│   ├── base_model.py         # Base model definition
│   ├── transfer_model.py     # Transfer learning implementation
│   ├── run_transfer_learning.py # Script to run transfer learning
│   ├── evaluate_model.py     # Model evaluation utilities
│   ├── mun_speech_evaluator.joblib # Pre-trained model
│   ├── training_data.json    # Training data for MUN speeches
│   ├── un_general_debates.csv # UN General Debates dataset
│   └── requirements.txt      # Project dependencies
└── README.md                 # Project documentation
```

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- Internet connection for API services (Google Speech Recognition)
- Libraries: (see requirements.txt for full list)
  - Flask
  - TensorFlow/PyTorch
  - librosa
  - transformers
  - numpy
  - pandas

