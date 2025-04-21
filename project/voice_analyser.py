# Configure FFmpeg path
import os

# Set path to your FFmpeg installation
FFMPEG_PATH = r"C:\Users\Decay\Downloads\ffmpeg-2025-04-17-git-7684243fbe-full_build\bin"

# Add FFmpeg to system PATH
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

print(f"FFmpeg configured at: {FFMPEG_PATH}")

import numpy as np
import librosa
import speech_recognition as sr
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import parselmouth
from parselmouth.praat import call
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
import json
import soundfile as sf
from pydub import AudioSegment

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Helper function to convert NumPy types to Python native types
def convert_to_serialisable(obj: Any) -> Any:
    """Convert NumPy types to Python native types for JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serialisable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serialisable(item) for item in obj]
    else:
        return obj

class VoiceAnalyser:
    """
    Analyses speech recordings for various aspects of speech quality.
    """
    
    def __init__(self):
        """Initialise the voice analyser with necessary resources."""
        self.recognizer = sr.Recognizer()
        
    def process_audio_file(self, audio_file_path: str) -> Dict:
        """
        Process an audio file and extract various speech metrics.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dict: Dictionary containing speech analysis results
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                print(f"Error: Audio file not found at {audio_file_path}")
                return {
                    'error': f"Audio file not found at {audio_file_path}",
                    'transcript': "",
                    'overall_score': 5.0
                }
                
            # Get file extension
            _, file_extension = os.path.splitext(audio_file_path)
            
            # Convert audio if it's not in WAV format
            temp_wav_path = None
            if file_extension.lower() not in ['.wav']:
                try:
                    # Set FFmpeg path explicitly
                    ffmpeg_path = r"C:\Users\Decay\Downloads\ffmpeg-2025-04-17-git-7684243fbe-full_build\bin\ffmpeg"  # Change this to your actual path
                    AudioSegment.converter = ffmpeg_path
                    
                    # Create temp file
                    temp_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd)
                    
                    # Convert to WAV with explicit parameters
                    print(f"Converting {file_extension} to WAV format...")
                    sound = AudioSegment.from_file(audio_file_path)
                    sound.export(temp_wav_path, format="wav")
                    
                    # Use the converted file instead
                    audio_file_path = temp_wav_path
                except Exception as e:
                    print(f"Error converting audio format: {e}")
            
            # Load the audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            
            # Transcribe the audio
            transcript = self._transcribe_audio(audio_file_path)
            
            # Analyse different aspects of the speech
            results = {
                'transcript': transcript,
                'voice_modulation': self._analyse_voice_modulation(y, sr, audio_file_path),
                'speech_rate': self._analyse_speech_rate(y, sr, transcript),
                'pauses': self._analyse_pauses(y, sr),
                'stuttering': self._analyse_stuttering(transcript),
                'clarity': self._analyse_clarity(transcript),
                'vocabulary': self._analyse_vocabulary(transcript),
                'confidence': self._analyse_confidence(y, sr, transcript),
                'emotion': self._analyse_emotion(y, sr, transcript),
                'pitch_tone': self._analyse_pitch_tone(y, sr, audio_file_path),
            }
            
            # Add overall score
            results['overall_score'] = self._calculate_overall_score(results)
            
            # Clean up temp file if created
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                
            # Convert any NumPy types to Python native types
            return convert_to_serialisable(results)
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {
                'error': str(e),
                'transcript': "",
                'overall_score': 5.0
            }
    
    def _transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe the audio file to text.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Ensure the file exists and is readable
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
                
            # Try to open as wav file with speech_recognition
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language="en-GB")
                return text
        except FileNotFoundError as e:
            print(f"Error transcribing audio: {e}")
            return ""
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            
            # If it fails with standard WAV format, try with alternative methods
            try:
                # Try with a different approach - convert to wav first using pydub
                try:
                    # Create temp file
                    temp_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd)
                    
                    # Export as WAV
                    sound = AudioSegment.from_file(audio_file_path)
                    sound.export(temp_wav_path, format="wav")
                    
                    # Try transcribing again
                    with sr.AudioFile(temp_wav_path) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data, language="en-GB")
                        
                    # Clean up
                    os.remove(temp_wav_path)
                    return text
                except Exception as e2:
                    print(f"Alternative transcription method failed: {e2}")
                    return ""
            except Exception as e3:
                print(f"All transcription methods failed: {e3}")
                return ""
    
    def _analyse_voice_modulation(self, y: np.ndarray, sr: int, audio_file_path: str) -> Dict:
        """
        Analyse pitch variation throughout the speech.
        
        Args:
            y: Audio time series
            sr: Sample rate
            audio_file_path: Path to the audio file for Praat analysis
            
        Returns:
            Dict: Voice modulation metrics
        """
        try:
            # First, check if we have valid audio data
            if len(y) < sr:  # Less than 1 second of audio
                raise ValueError("Audio file too short or empty")
                
            # Try to convert to WAV for Praat if needed
            temp_file = None
            try:
                # Create a temporary file for Praat
                temp_fd, temp_file = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                
                # Save audio using soundfile
                sf.write(temp_file, y, sr, format='WAV')
                
                # Use the temp file for Praat analysis
                sound = parselmouth.Sound(temp_file)
            except Exception as e:
                print(f"Error preparing audio for Praat: {e}")
                # Fall back to original file
                sound = parselmouth.Sound(audio_file_path)
            
            # Get pitch statistics
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
            std_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            
            # Calculate pitch range and variation coefficient
            pitch_range = max_pitch - min_pitch
            variation_coefficient = (std_pitch / mean_pitch) * 100 if mean_pitch > 0 else 0
            
            # Cleanup temp file if created
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                
            return {
                'mean_pitch': mean_pitch,
                'pitch_range': pitch_range,
                'variation_coefficient': variation_coefficient,
                'score': min(10, max(1, variation_coefficient / 10))
            }
        except Exception as e:
            print(f"Error analysing voice modulation: {e}")
            return {
                'mean_pitch': 0,
                'pitch_range': 0,
                'variation_coefficient': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_speech_rate(self, y: np.ndarray, sr: int, transcript: str) -> Dict:
        """
        Calculate speech rate in words per minute.
        
        Args:
            y: Audio time series
            sr: Sample rate
            transcript: Transcribed text
            
        Returns:
            Dict: Speech rate metrics
        """
        try:
            # Calculate duration in minutes
            duration_sec = librosa.get_duration(y=y, sr=sr)
            duration_min = duration_sec / 60
            
            # Count words
            words = word_tokenize(transcript)
            word_count = len(words)
            
            # Calculate words per minute
            wpm = word_count / duration_min if duration_min > 0 else 0
            
            # Score based on optimal speech rate (120-160 WPM is considered ideal)
            if 120 <= wpm <= 160:
                score = 10.0
            elif 100 <= wpm < 120 or 160 < wpm <= 180:
                score = 8.0
            elif 80 <= wpm < 100 or 180 < wpm <= 200:
                score = 6.0
            else:
                score = 4.0
                
            return {
                'words_per_minute': wpm,
                'word_count': word_count,
                'duration_sec': duration_sec,
                'score': score
            }
        except Exception as e:
            print(f"Error analysing speech rate: {e}")
            return {
                'words_per_minute': 0,
                'word_count': 0,
                'duration_sec': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_pauses(self, y: np.ndarray, sr: int) -> Dict:
        """
        Detect pauses in speech.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dict: Pause analysis metrics
        """
        try:
            # Convert to mono if needed
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            
            # Set a threshold for silence detection
            threshold = 0.03
            
            # Compute the root mean square energy
            frame_length = int(sr * 0.025)  # 25ms frames
            hop_length = int(sr * 0.010)    # 10ms hop
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find silent frames
            silent_frames = (rms < threshold)
            
            # Convert frame indices to time
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            
            # Find pauses (consecutive silent frames)
            pause_starts = []
            pause_ends = []
            in_pause = False
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    pause_starts.append(times[i])
                    in_pause = True
                elif not is_silent and in_pause:
                    pause_ends.append(times[i])
                    in_pause = False
            
            # Handle the case where the recording ends during a pause
            if in_pause:
                pause_ends.append(times[-1])
            
            # Calculate pause durations
            pause_durations = [end - start for start, end in zip(pause_starts, pause_ends)]
            
            # Count pauses longer than 0.5 seconds
            significant_pauses = [d for d in pause_durations if d >= 0.5]
            pause_count = len(significant_pauses)
            
            # Calculate total pause time
            total_pause_time = sum(significant_pauses)
            
            # Calculate pause frequency (pauses per minute)
            duration_min = librosa.get_duration(y=y, sr=sr) / 60
            pause_frequency = pause_count / duration_min if duration_min > 0 else 0
            
            # Score based on pause frequency (3-4 pauses/min is considered good)
            if 2 <= pause_frequency <= 6:
                score = 9.0
            elif 1 <= pause_frequency < 2 or 6 < pause_frequency <= 8:
                score = 7.0
            elif pause_frequency > 8:
                score = 5.0
            else:
                score = 3.0  # Too few pauses
                
            return {
                'pause_count': pause_count,
                'pause_frequency': pause_frequency,
                'average_pause_duration': np.mean(pause_durations) if pause_durations else 0,
                'total_pause_time': total_pause_time,
                'score': score
            }
        except Exception as e:
            print(f"Error analysing pauses: {e}")
            return {
                'pause_count': 0,
                'pause_frequency': 0,
                'average_pause_duration': 0,
                'total_pause_time': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_stuttering(self, transcript: str) -> Dict:
        """
        Detect stuttering patterns in transcript.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Dict: Stuttering analysis metrics
        """
        try:
            words = word_tokenize(transcript.lower())
            
            # Count repeated words
            repetitions = 0
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    repetitions += 1
            
            # Look for common filler words
            filler_words = ['um', 'uh', 'ah', 'er', 'like', 'you know', 'actually', 'basically', 'literally']
            filler_count = sum(1 for word in words if word in filler_words)
            
            # Calculate stutter metrics
            word_count = len(words)
            repetition_rate = (repetitions / word_count) * 100 if word_count > 0 else 0
            filler_rate = (filler_count / word_count) * 100 if word_count > 0 else 0
            
            # Calculate combined stuttering score (lower is better)
            stutter_score = repetition_rate + filler_rate
            
            # Convert to 1-10 scale (10 is best, i.e., no stuttering)
            normalised_score = max(1, min(10, 10 - (stutter_score / 2)))
            
            return {
                'repetition_count': repetitions,
                'repetition_rate': repetition_rate,
                'filler_count': filler_count,
                'filler_rate': filler_rate,
                'score': normalised_score
            }
        except Exception as e:
            print(f"Error analysing stuttering: {e}")
            return {
                'repetition_count': 0,
                'repetition_rate': 0,
                'filler_count': 0,
                'filler_rate': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_clarity(self, transcript: str) -> Dict:
        """
        Analyse clarity based on transcript completeness.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Dict: Clarity analysis metrics
        """
        try:
            # Count words and complete sentences
            words = word_tokenize(transcript)
            sentences = sent_tokenize(transcript)
            
            word_count = len(words)
            
            # Look for incomplete sentences or transcription artifacts
            incomplete_markers = ['...', '..', '.....', '[inaudible]', '[unclear]']
            incomplete_count = sum(1 for marker in incomplete_markers if marker in transcript)
            
            # Calculate average words per sentence
            words_per_sentence = word_count / len(sentences) if sentences else 0
            
            # Flag potentially incomplete transcription
            is_incomplete = (incomplete_count > 0) or (words_per_sentence > 25)
            
            # Simple scoring based on completeness
            if not is_incomplete and words_per_sentence >= 5 and words_per_sentence <= 20:
                score = 9.0
            elif not is_incomplete and (words_per_sentence < 5 or words_per_sentence > 20):
                score = 7.0
            else:
                score = 5.0
                
            return {
                'word_count': word_count,
                'sentence_count': len(sentences),
                'words_per_sentence': words_per_sentence,
                'incomplete_markers': incomplete_count,
                'score': score
            }
        except Exception as e:
            print(f"Error analysing clarity: {e}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'words_per_sentence': 0,
                'incomplete_markers': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_vocabulary(self, transcript: str) -> Dict:
        """
        Analyse vocabulary complexity.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Dict: Vocabulary analysis metrics
        """
        try:
            # Calculate readability scores
            flesch_reading_ease = textstat.flesch_reading_ease(transcript)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(transcript)
            
            # Calculate lexical diversity
            words = word_tokenize(transcript.lower())
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if words else 0
            
            # Score based on readability and lexical diversity
            readability_score = min(10, max(1, flesch_reading_ease / 10))
            diversity_score = min(10, max(1, lexical_diversity * 10))
            
            combined_score = (readability_score + diversity_score) / 2
            
            return {
                'flesch_reading_ease': flesch_reading_ease,
                'grade_level': flesch_kincaid_grade,
                'lexical_diversity': lexical_diversity,
                'unique_word_count': len(unique_words),
                'total_word_count': len(words),
                'score': combined_score
            }
        except Exception as e:
            print(f"Error analysing vocabulary: {e}")
            return {
                'flesch_reading_ease': 0,
                'grade_level': 0,
                'lexical_diversity': 0,
                'unique_word_count': 0,
                'total_word_count': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_confidence(self, y: np.ndarray, sr: int, transcript: str) -> Dict:
        """
        Analyse speaker confidence based on audio and transcript features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            transcript: Transcribed text
            
        Returns:
            Dict: Confidence analysis metrics
        """
        try:
            # Look for filler words in transcript
            words = word_tokenize(transcript.lower())
            filler_words = ['um', 'uh', 'ah', 'er', 'like', 'you know', 'actually', 'basically', 'literally']
            filler_count = sum(1 for word in words if word in filler_words)
            filler_rate = (filler_count / len(words)) * 100 if words else 0
            
            # Calculate average amplitude (energy)
            energy = np.mean(np.abs(y))
            
            # Calculate score based on filler rate and energy
            filler_score = max(1, min(10, 10 - (filler_rate / 2)))
            energy_score = min(10, max(1, energy * 100))
            
            combined_score = (filler_score * 0.7) + (energy_score * 0.3)
            
            return {
                'filler_count': filler_count,
                'filler_rate': filler_rate,
                'energy_level': energy,
                'score': combined_score
            }
        except Exception as e:
            print(f"Error analysing confidence: {e}")
            return {
                'filler_count': 0,
                'filler_rate': 0,
                'energy_level': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_emotion(self, y: np.ndarray, sr: int, transcript: str) -> Dict:
        """
        Analyse emotional tone in speech.
        
        Args:
            y: Audio time series
            sr: Sample rate
            transcript: Transcribed text
            
        Returns:
            Dict: Emotion analysis metrics
        """
        try:
            # Extract audio features related to emotion
            # Note: A full emotion detection system would use a trained model
            # This is a simplified approach using audio features
            
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Extract energy (volume) features
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            
            # Simple mapping of features to emotion indicators
            # High energy + high pitch variation might indicate excitement
            emotion_score = (energy_std / energy_mean) * 10 if energy_mean > 0 else 5
            emotion_score = min(10, max(1, emotion_score))
            
            # Determine dominant emotion (very simplified)
            emotions = {
                'neutral': 0.5,
                'passionate': 0,
                'confident': 0,
                'uncertain': 0
            }
            
            # Simple heuristics for emotion classification
            if energy_mean > 0.1 and pitch_mean > 200:
                emotions['passionate'] = 0.7
                emotions['confident'] = 0.6
            elif energy_mean > 0.08:
                emotions['confident'] = 0.8
            elif energy_std < 0.02:
                emotions['neutral'] = 0.9
            else:
                emotions['uncertain'] = 0.6
            
            # Normalise emotion scores
            total = sum(emotions.values())
            for key in emotions:
                emotions[key] = emotions[key] / total if total > 0 else 0
            
            return {
                'energy_mean': float(energy_mean),
                'energy_variation': float(energy_std),
                'pitch_mean': float(pitch_mean),
                'spectral_centroid': float(spectral_centroid),
                'emotions': emotions,
                'dominant_emotion': max(emotions, key=emotions.get),
                'score': float(emotion_score)
            }
        except Exception as e:
            print(f"Error analysing emotion: {e}")
            return {
                'energy_mean': 0,
                'energy_variation': 0,
                'pitch_mean': 0,
                'spectral_centroid': 0,
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'score': 5.0,
                'error': str(e)
            }
    
    def _analyse_pitch_tone(self, y: np.ndarray, sr: int, audio_file_path: str) -> Dict:
        """
        Analyse pitch and tone characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            audio_file_path: Path to the audio file for Praat analysis
            
        Returns:
            Dict: Pitch and tone analysis metrics
        """
        try:
            # First, check if we have valid audio data
            if len(y) < sr:  # Less than 1 second of audio
                raise ValueError("Audio file too short or empty")
                
            # Try to convert to WAV for Praat if needed
            temp_file = None
            try:
                # Create a temporary file for Praat
                temp_fd, temp_file = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                
                # Save audio using soundfile
                sf.write(temp_file, y, sr, format='WAV')
                
                # Use the temp file for Praat analysis
                sound = parselmouth.Sound(temp_file)
            except Exception as e:
                print(f"Error preparing audio for Praat: {e}")
                # Fall back to original file
                sound = parselmouth.Sound(audio_file_path)
                
            # Get pitch statistics
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
            std_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            
            # Calculate jitter and shimmer (measures of voice quality)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Calculate harmonics-to-noise ratio
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            # Cleanup temp file if created
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Score based on voice quality metrics
            # Lower jitter and shimmer values indicate better voice quality
            jitter_score = min(10, max(1, 10 - (jitter * 1000)))
            shimmer_score = min(10, max(1, 10 - (shimmer * 100)))
            hnr_score = min(10, max(1, hnr / 2))  # Higher HNR is better
            
            combined_score = (jitter_score + shimmer_score + hnr_score) / 3
            
            return {
                'mean_pitch': float(mean_pitch),
                'min_pitch': float(min_pitch),
                'max_pitch': float(max_pitch),
                'pitch_range': float(max_pitch - min_pitch),
                'jitter': float(jitter),
                'shimmer': float(shimmer),
                'harmonics_to_noise': float(hnr),
                'score': float(combined_score)
            }
        except Exception as e:
            print(f"Error analysing pitch and tone: {e}")
            return {
                'mean_pitch': 0,
                'min_pitch': 0,
                'max_pitch': 0,
                'pitch_range': 0,
                'jitter': 0,
                'shimmer': 0,
                'harmonics_to_noise': 0,
                'score': 5.0,
                'error': str(e)
            }
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """
        Calculate an overall voice quality score based on individual metrics.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            float: Overall score (1-10)
        """
        # Define weights for different factors
        weights = {
            'voice_modulation': 0.15,
            'speech_rate': 0.1,
            'pauses': 0.05,
            'stuttering': 0.1,
            'clarity': 0.15,
            'vocabulary': 0.1,
            'confidence': 0.15,
            'emotion': 0.1,
            'pitch_tone': 0.1
        }
        
        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0
        
        for factor, weight in weights.items():
            if factor in results and 'score' in results[factor]:
                weighted_sum += results[factor]['score'] * weight
                total_weight += weight
        
        # Return normalised score
        return weighted_sum / total_weight if total_weight > 0 else 5.0

# Create a helper function to handle file uploads
def process_audio_file(file_path: str) -> Dict:
    """
    Process an audio file and return analysis results.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict: Analysis results
    """
    analyser = VoiceAnalyser()
    return analyser.process_audio_file(file_path) 