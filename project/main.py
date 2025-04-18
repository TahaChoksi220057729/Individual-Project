"http://192.168.1.184:5000 --> Not on Local Machine"
"http://127.0.0.1:5000 --> On Local Machine"

import os
import json

# Set environment variable to silence Flask development server warnings
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, request, jsonify, render_template
from speech_evaluator import SpeechEvaluator

# Initialize speech evaluator
speech_evaluator = SpeechEvaluator()

# Initialise Flask app
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')

# Helper function for error handling
def handle_error(e, status_code=400):
    """
    Standardized error handling for API endpoints.
    
    Args:
        e (Exception): The exception that occurred
        status_code (int): HTTP status code to return
        
    Returns:
        tuple: (JSON response with error info, status code)
    """
    error_message = str(e)
    
    # Simplify common error messages
    if "features" in error_message:
        error_message = "Topic mismatch. Try a different topic or use safe mode."
    
    return jsonify({
        'error': error_message,
        'success': False
    }), status_code

@app.route('/')
def index():
    """Render the main application page."""
    # Get available topics for the dropdown
    topics = speech_evaluator.get_topics()
    return render_template('index.html', topics=topics)

@app.route('/evaluate', methods=['POST'])
def evaluate_form():
    """
    Form endpoint for speech evaluation.
    
    Accepts form data or JSON and passes it to the API endpoint.
    This supports both form submissions and API calls.
    """
    if request.is_json:
        # If JSON data is sent, use it directly
        data = request.get_json()
    else:
        # If form data is sent, convert it to the expected format
        data = {
            'speech': request.form.get('speech', '').strip(),
            'topic': request.form.get('topic', ''),
            'safe_mode': request.form.get('safe_mode') == 'true'
        }
        
        # Clean up empty topic
        if not data['topic']:
            data['topic'] = None
    
    # Validate input
    if not data.get('speech'):
        return jsonify({'error': 'Speech text is required'}), 400
    
    try:
        # Evaluate the speech
        result = speech_evaluator.evaluate_speech(
            data['speech'], 
            data['topic'], 
            data.get('safe_mode', False)
        )
        return jsonify(result)
    except Exception as e:
        return handle_error(e)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_speech():
    """
    API endpoint for speech evaluation.
    
    Expects JSON with:
    - speech: text of speech to evaluate
    - topic (optional): topic of the speech
    - safe_mode (optional): boolean, whether to use safe mode
    
    Returns:
    - score: numerical score (1-10)
    - feedback: detailed feedback object
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    # Validate input
    if 'speech' not in data or not data['speech'].strip():
        return jsonify({'error': 'Speech text is required'}), 400
    
    speech = data['speech'].strip()
    topic = data.get('topic')
    safe_mode = data.get('safe_mode', False)
    
    # Clean up topic if it's empty
    if topic and not topic.strip():
        topic = None
    
    try:
        # Evaluate the speech
        result = speech_evaluator.evaluate_speech(speech, topic, safe_mode)
        return jsonify(result)
    except Exception as e:
        return handle_error(e)

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """API endpoint to retrieve available topics."""
    try:
        topics = speech_evaluator.get_topics()
        return jsonify({'topics': topics})
    except Exception as e:
        return handle_error(e)

@app.route('/api/training/add', methods=['POST'])
def add_training_sample():
    """
    API endpoint to add a training sample.
    
    Expects JSON with:
    - speech: text of speech
    - score: numerical score (1-10)
    - topic (optional): topic of the speech
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    # Validate input
    if 'speech' not in data or not data['speech'].strip():
        return jsonify({'error': 'Speech text is required'}), 400
    
    if 'score' not in data:
        return jsonify({'error': 'Score is required'}), 400
    
    try:
        # Add the training sample
        speech = data['speech'].strip()
        score = float(data['score'])
        topic = data.get('topic')
        
        # Clean up topic if it's empty
        if topic and not topic.strip():
            topic = None
        
        success = speech_evaluator.add_training_sample(speech, score, topic)
        
        if success:
            return jsonify({'success': True, 'message': 'Training sample added successfully'})
        else:
            return handle_error(Exception('Failed to add training sample'), 500)
    except ValueError as e:
        return handle_error(e, 400)
    except Exception as e:
        return handle_error(e, 500)

@app.route('/api/training/data', methods=['GET'])
def get_training_data():
    """API endpoint to retrieve training data."""
    try:
        data = speech_evaluator.load_training_data()
        return jsonify({'data': data})
    except Exception as e:
        return handle_error(e)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """
    API endpoint to trigger model training.
    
    Can be called without parameters to train with all available data,
    or with specific training data provided.
    """
    try:
        if request.is_json:
            data = request.get_json()
            if 'speeches' in data and 'scores' in data:
                speeches = data['speeches']
                scores = data['scores']
                topics = data.get('topics')
                
                if len(speeches) != len(scores):
                    return handle_error(ValueError('Number of speeches must match number of scores'), 400)
                
                success = speech_evaluator.train_model(speeches, scores, topics)
            else:
                # Train with all available data
                training_data = speech_evaluator.load_training_data()
                if not training_data:
                    return handle_error(ValueError('No training data available'), 400)
                
                speeches = [item['speech'] for item in training_data]
                scores = [item['score'] for item in training_data]
                
                if all('topic' in item for item in training_data):
                    topics = [item['topic'] for item in training_data]
                    success = speech_evaluator.train_model(speeches, scores, topics)
                else:
                    success = speech_evaluator.train_model(speeches, scores)
        else:
            # Train with all available data
            training_data = speech_evaluator.load_training_data()
            if not training_data:
                return handle_error(ValueError('No training data available'), 400)
            
            speeches = [item['speech'] for item in training_data]
            scores = [item['score'] for item in training_data]
            
            if all('topic' in item for item in training_data):
                topics = [item['topic'] for item in training_data]
                success = speech_evaluator.train_model(speeches, scores, topics)
            else:
                success = speech_evaluator.train_model(speeches, scores)
        
        if success:
            return jsonify({'success': True, 'message': 'Model trained successfully'})
        else:
            return handle_error(Exception('Failed to train model'), 500)
    except Exception as e:
        return handle_error(e, 500)

@app.route('/debug/topics')
def debug_topics():
    """Debug endpoint to directly view topics."""
    topics = speech_evaluator.get_topics()
    return jsonify({
        'topics_count': len(topics),
        'topics': topics
    })

if __name__ == '__main__':
    # Determine port for running the app
    port = int(os.environ.get('PORT', 5000))
    
    # Minimize console output
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Create static/templates directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    print(f"MUN Speech Evaluator running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)