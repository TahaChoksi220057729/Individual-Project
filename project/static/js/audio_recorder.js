/**
 * Audio Recorder Module
 * 
 * Provides functionality for recording audio, playing it back, and uploading it.
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.audioUrl = null;
        this.stream = null;
        this.isRecording = false;
        this.context = null;
        this.processor = null;
    }

    /**
     * Request microphone access and initialize the recorder
     */
    async initialize() {
        try {
            // Get audio stream
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,          // mono
                    sampleRate: 44100,        // standard sample rate
                    echoCancellation: true,   // reduce echo
                    noiseSuppression: true,   // reduce background noise
                }
            });
            
            // Create audio context
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.context = new AudioContext();
            
            // Use MediaRecorder with specific MIME type for WAV
            const options = { 
                audioBitsPerSecond: 128000,
                mimeType: 'audio/webm' // We'll convert this to WAV on the server
            };
            
            // Check if mimeType is supported
            if (MediaRecorder.isTypeSupported(options.mimeType)) {
                this.mediaRecorder = new MediaRecorder(this.stream, options);
            } else {
                console.warn(`${options.mimeType} is not supported, using default format.`);
                this.mediaRecorder = new MediaRecorder(this.stream);
            }
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                // Create Blob with proper MIME type
                this.audioBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                this.isRecording = false;
                
                console.log(`Recording complete: ${this.audioBlob.size} bytes, type: ${this.audioBlob.type}`);
                
                // Trigger finished event
                const event = new CustomEvent('recordingFinished', { 
                    detail: { audioUrl: this.audioUrl }
                });
                document.dispatchEvent(event);
            };
            
            return true;
        } catch (error) {
            console.error('Error initializing audio recorder:', error);
            return false;
        }
    }
    
    /**
     * Start recording audio
     */
    startRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'recording') {
            this.audioChunks = [];
            
            // Start with small timeslice for more frequent ondataavailable events
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            
            console.log('Recording started');
            
            // Trigger started event
            const event = new CustomEvent('recordingStarted');
            document.dispatchEvent(event);
            
            return true;
        }
        return false;
    }
    
    /**
     * Stop recording audio
     */
    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            console.log('Recording stopped');
            return true;
        }
        return false;
    }
    
    /**
     * Play recorded audio
     */
    playRecording() {
        if (this.audioUrl) {
            const audio = new Audio(this.audioUrl);
            audio.play();
            return true;
        }
        return false;
    }
    
    /**
     * Get the recorded audio as a File object
     */
    getAudioFile(filename = 'recording.wav') {
        if (this.audioBlob) {
            // Use the detected MIME type from the recorder
            const mimeType = this.mediaRecorder ? this.mediaRecorder.mimeType : 'audio/webm';
            
            // Adjust filename extension based on MIME type
            let extension = '.webm';
            if (mimeType.includes('ogg')) extension = '.ogg';
            else if (mimeType.includes('mp4')) extension = '.mp4';
            
            const finalFilename = filename.endsWith(extension) ? filename : `${filename}${extension}`;
            console.log(`Creating file: ${finalFilename} (${mimeType})`);
            
            return new File([this.audioBlob], finalFilename, { type: mimeType });
        }
        return null;
    }
    
    /**
     * Upload the recorded audio to the server
     */
    async uploadRecording(url, additionalData = {}) {
        if (!this.audioBlob) {
            throw new Error('No recording available to upload');
        }
        
        const formData = new FormData();
        const audioFile = this.getAudioFile();
        
        if (audioFile) {
            formData.append('audio', audioFile);
            console.log(`Uploading audio: ${audioFile.name} (${audioFile.type}), size: ${audioFile.size} bytes`);
        } else {
            throw new Error('Failed to create audio file');
        }
        
        // Add any additional data to the form
        for (const [key, value] of Object.entries(additionalData)) {
            formData.append(key, value);
        }
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error uploading recording:', error);
            throw error;
        }
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.audioUrl) {
            URL.revokeObjectURL(this.audioUrl);
        }
        
        if (this.context) {
            this.context.close();
        }
        
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.audioUrl = null;
        this.stream = null;
        this.isRecording = false;
        this.context = null;
        this.processor = null;
    }
}

// Create a global instance
const audioRecorder = new AudioRecorder();

// Export the recorder instance
window.audioRecorder = audioRecorder; 