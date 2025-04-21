"http://192.168.1.184:5000 --> Not on Local Machine"
"http://127.0.0.1:5000 --> On Local Machine"

import os
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*NVML.*")

# Set environment variable to silence Flask development server warnings
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, request, jsonify, render_template, url_for, redirect, send_from_directory
from speech_evaluator import SpeechEvaluator
from voice_analyser import process_audio_file
import tempfile
import uuid
import werkzeug.utils

# Initialise speech evaluator
speech_evaluator = SpeechEvaluator()

# Initialise Flask app
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')

# Create upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Helper function for error handling
def handle_error(e, status_code=400):
    """
    Standardised error handling for API endpoints.
    
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

@app.route('/debug/topics')
def debug_topics():
    """Debug endpoint to directly view topics."""
    topics = speech_evaluator.get_topics()
    return jsonify({
        'topics_count': len(topics),
        'topics': topics
    })

@app.route('/voice', methods=['GET'])
def voice_analysis_page():
    """Render the voice analysis page."""
    topics = speech_evaluator.get_topics()
    return render_template('voice_analysis.html', topics=topics)

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    """
    API endpoint for audio file upload.
    
    Expects:
    - audio: audio file (wav, mp3, etc.)
    - topic (optional): speech topic
    
    Returns:
    - voice_analysis: detailed voice analysis results
    - text_analysis: text-based analysis of transcribed speech
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Log upload info
        print(f"Processing audio upload: {audio_file.filename} ({audio_file.content_type})")
        
        # Secure the filename
        filename = werkzeug.utils.secure_filename(audio_file.filename)
        
        # Add unique ID to prevent filename collisions
        file_id = str(uuid.uuid4())[:8]
        base, ext = os.path.splitext(filename)
        
        # Ensure we have a valid extension
        if not ext:
            # Try to determine extension from content type
            if 'webm' in audio_file.content_type:
                ext = '.webm'
            elif 'ogg' in audio_file.content_type:
                ext = '.ogg'
            elif 'mp3' in audio_file.content_type:
                ext = '.mp3'
            elif 'mp4' in audio_file.content_type:
                ext = '.mp4'
            else:
                ext = '.audio'  # Generic fallback
                
        unique_filename = f"{base}_{file_id}{ext}"
        
        # Create full path to save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        audio_file.save(filepath)
        print(f"Saved audio file to: {filepath}")
        
        # Process the audio file
        voice_results = process_audio_file(filepath)
        
        # Check if there was an error
        if 'error' in voice_results:
            print(f"Error processing audio: {voice_results['error']}")
        
        # Get topic if provided
        topic = request.form.get('topic')
        safe_mode = request.form.get('safe_mode') == 'true'
        
        # Use the transcript for text analysis
        transcript = voice_results.get('transcript', '')
        
        # Only do text analysis if we have a transcript
        text_results = {}
        if transcript:
            try:
                text_results = speech_evaluator.evaluate_speech(transcript, topic, safe_mode)
            except Exception as e:
                text_results = {
                    'error': str(e),
                    'score': 5.0,
                    'feedback': {}
                }
        
        # Combine results
        combined_results = {
            'voice_analysis': voice_results,
            'text_analysis': text_results,
            'success': True
        }
        
        return jsonify(combined_results)
        
    except Exception as e:
        print(f"Exception in upload_audio: {str(e)}")
        return handle_error(e)

@app.route('/api/record_audio', methods=['POST'])
def record_audio():
    """
    API endpoint for audio recording from browser.
    
    Expects:
    - audio: audio blob (wav format)
    - topic (optional): speech topic
    
    Identical to upload_audio but with a different route name for clarity.
    """
    return upload_audio()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Determine port for running the app
    port = int(os.environ.get('PORT', 5000))
    
    # Minimise console output
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Create static/templates/uploads directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Run the app
    print(f"MUN Speech Evaluator running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)