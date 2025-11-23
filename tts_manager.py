import os
import wave
import threading
import queue
import pygame
import tempfile
from piper.voice import PiperVoice
from piper.config import SynthesisConfig

class PiperTTSManager:
    # Map emotions to Piper parameters
    # length_scale: Speed (Lower is faster)
    # noise_scale:  Volatility (Higher is more 'breath/randomness')
    EMOTION_TTS_PARAMS = {
        'angry':    {'speed': 0.85, 'pitch': 0.15, 'noise': 0.60},  # Fast, aggressive tone
        'disgust':  {'speed': 1.25, 'pitch': -0.05, 'noise': 0.70}, # Slow, drawn-out delivery
        'fear':     {'speed': 0.80, 'pitch': 0.20, 'noise': 0.90},  # Rapid, unstable speech
        'happy':    {'speed': 0.90, 'pitch': 0.10, 'noise': 0.75},  # Upbeat, variable tone
        'sad':      {'speed': 1.30, 'pitch': -0.10, 'noise': 0.50}, # Very slow, flat delivery
        'surprise': {'speed': 0.85, 'pitch': 0.25, 'noise': 0.95},  # Fast, highly variable
        'neutral':  {'speed': 1.00, 'pitch': 0.0, 'noise': 0.66}    # Standard baseline
    }

    def __init__(self, model_path: str, config_path: str):
        self.queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.model_path = model_path
        
        try:
            # Initialize Piper Voice
            # Ensure the .onnx and .json files are in the same folder!
            self.voice = PiperVoice.load(model_path, config_path)
            
            # Initialize Pygame Mixer (Standard 22050Hz usually works best for Piper)
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=2048)
            print(f"[PiperTTS] Loaded model: {model_path}")
            self.audio_available = True
        except Exception as e:
            print(f"[PiperTTS] Init Error: {e}")
            self.audio_available = False

    def is_available(self) -> bool:
        return self.audio_available

    def speak(self, text: str, emotion: str = 'neutral'):
        """Non-blocking speak function"""
        if not self.audio_available:
            return

        # Get settings or default to neutral
        settings = self.EMOTION_TTS_PARAMS.get(emotion.lower(), self.EMOTION_TTS_PARAMS['neutral'])
        
        # Add to queue
        self.queue.put((text, settings))
        
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def _worker_loop(self):
        while True:
            try:
                text, settings = self.queue.get(timeout=1)
                
                # Generate Audio to Temp File
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                    temp_path = tf.name
                
                # Create synthesis config with emotion parameters
                syn_config = SynthesisConfig(
                    length_scale=settings['speed'],
                    noise_scale=settings['noise']
                )
                
                # Generate audio chunks
                audio_chunks = list(self.voice.synthesize(text, syn_config))
                
                # Write audio chunks to WAV file
                with wave.open(temp_path, "wb") as wav_file:
                    if audio_chunks:
                        # Configure wave file parameters from first chunk
                        first_chunk = audio_chunks[0]
                        wav_file.setnchannels(getattr(first_chunk, 'sample_channels', 1))
                        wav_file.setsampwidth(getattr(first_chunk, 'sample_width', 2))
                        wav_file.setframerate(getattr(first_chunk, 'sample_rate', 22050))
                        
                        # Write all audio data
                        for chunk in audio_chunks:
                            wav_file.writeframes(chunk.audio_int16_bytes)
                
                # Play Audio
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
                
                # Cleanup
                pygame.mixer.music.unload() # Important to release file lock
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
                self.queue.task_done()
                
            except queue.Empty:
                self.is_running = False
                break
            except Exception as e:
                print(f"[PiperTTS] Worker Error: {e}")

    def stop(self):
        try:
            pygame.mixer.music.stop()
            # Clear queue to stop pending sentences
            with self.queue.mutex:
                self.queue.queue.clear()
        except:
            pass

    # Sync method for download button (Streamlit safe)
    def generate_audio_bytes(self, text: str, emotion: str, speed_multiplier: float = 1.0) -> bytes:
        """Generates audio bytes immediately for download buttons"""
        settings = self.EMOTION_TTS_PARAMS.get(emotion.lower(), self.EMOTION_TTS_PARAMS['neutral'])
        
        # Apply speed multiplier to emotion-based speed
        adjusted_speed = settings['speed'] * speed_multiplier
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            temp_path = tf.name

        try:
            # Create synthesis config with emotion parameters
            syn_config = SynthesisConfig(
                length_scale=adjusted_speed,
                noise_scale=settings['noise']
            )
            
            # Generate audio chunks
            audio_chunks = list(self.voice.synthesize(text, syn_config))
            
            # Write audio chunks to WAV file
            with wave.open(temp_path, "wb") as wav_file:
                if audio_chunks:
                    # Configure wave file parameters from first chunk
                    first_chunk = audio_chunks[0]
                    wav_file.setnchannels(getattr(first_chunk, 'sample_channels', 1))
                    wav_file.setsampwidth(getattr(first_chunk, 'sample_width', 2))
                    wav_file.setframerate(getattr(first_chunk, 'sample_rate', 22050))
                    
                    # Write all audio data
                    for chunk in audio_chunks:
                        wav_file.writeframes(chunk.audio_int16_bytes)
            
            with open(temp_path, "rb") as f:
                data = f.read()
                
            return data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @staticmethod
    def get_available_voices():
        """Return available voice options"""
        return {
            'Sam (Male - Medium)': 'en_US-sam-medium.onnx',
            'Amy (Female - Medium)': 'en_US-amy-medium.onnx',
            'Arctic (Male - Medium)': 'en_US-arctic-medium.onnx',
            'Bryce (Male - Medium)': 'en_US-bryce-medium.onnx',
            'HFC Female (Medium)': 'en_US-hfc_female-medium.onnx',
            'HFC Male (Medium)': 'en_US-hfc_male-medium.onnx',
            'Lessac (Male - High)': 'en_US-lessac-high.onnx',
            'LibriTTS (Mixed - High)': 'en_US-libritts-high.onnx',
            'Norman (Male - Medium)': 'en_US-norman-medium.onnx',
            'Ryan (Male - High)': 'en_US-ryan-high.onnx'
        }
    
    @staticmethod
    def get_emotion_options():
        """Return available emotion options for manual override"""
        return ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']