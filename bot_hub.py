import os
import fitz  
import ollama 
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import time
import threading
import queue
from piper.voice import PiperVoice # For Piper TTS
from faster_whisper import WhisperModel # For ASR
import soundfile as sf # For reading/writing audio files
import sounddevice as sd # For playing audio
import pyaudio # For microphone input
import torch # For Silero VAD
import torchaudio # For Silero VAD

# --- Configuration ---
OLLAMA_MODEL = "mistral" 
PDF_DATA_DIR = "data"    
TTS_MODEL_DIR = "tts_models" 
TTS_MODEL_BASENAME = "en_US-lessac-high" 
WHISPER_MODEL_SIZE = "small.en" 

# Microphone/Audio Stream Configuration
MIC_RATE = 16000  # Sample rate for microphone (Hz)
MIC_CHUNK_SIZE = 512 # Number of frames per buffer
VAD_THRESHOLD = 0.65 # Voice Activity Detection threshold (0-1, higher means less sensitive)
SILENCE_TIMEOUT_SECONDS = 0.7 # How long to wait for continuous silence before marking speech as ended

CHUNK_SIZE = 500         # Max characters per text chunk for RAG
CHUNK_OVERLAP = 50       # Overlap between chunks
TOP_K_RAG = 3            # Number of top relevant chunks to retrieve for RAG

# --- Global Variables for TTS Playback and Interruption ---
tts_playback_thread = None
stop_speaking_event = threading.Event() # Event to signal TTS thread to stop
mic_input_queue = queue.Queue() # Queue to pass audio chunks from mic thread to ASR
vad_active = False # Flag to control VAD listening
speech_detected_event = threading.Event() # Event to signal speech detection

# --- Services ---

class MicrophoneStream:
    """
    Opens a recording stream from a microphone and puts audio chunks into a queue.
    Integrates Silero VAD for voice activity detection.
    """
    def __init__(self, rate, chunk_size, vad_threshold, silence_timeout):
        self.rate = rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.vad_model, self.vad_utils = self._load_vad_model()
        self.speech_buffer = [] # Buffer to store speech segments for ASR
        self.is_speaking = False # State of VAD
        self.last_speech_time = time.time() # To track silence duration

    def _load_vad_model(self):
        """Loads the Silero VAD model."""
        print("Loading Silero VAD model...")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        print("Silero VAD model loaded.")
        return model, (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)

    def start_stream(self):
        """Starts the microphone stream in a non-blocking mode."""
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk_size,
                                      stream_callback=self._callback)
        self.stream.start_stream()
        print("Microphone stream started.")

    def stop_stream(self):
        """Stops and closes the microphone stream."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Microphone stream stopped.")

    def _callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream."""
        global mic_input_queue, vad_active, speech_detected_event

        if vad_active:
            # Convert raw audio data to PyTorch tensor for VAD
            audio_int16 = np.frombuffer(in_data, dtype=np.int16).copy() # Added .copy() to make it writable
            audio_float32 = torch.from_numpy(audio_int16).float() / 32768.0

            # Perform VAD inference
            speech_prob = self.vad_model(audio_float32, self.rate).item()

            if speech_prob > self.vad_threshold:
                # Speech is detected
                self.last_speech_time = time.time() # Reset silence timer
                if not self.is_speaking:
                    # Speech just started
                    self.is_speaking = True
                    speech_detected_event.set() # Signal that speech has started
                    print("\n[VAD: Speech Detected]")
                self.speech_buffer.append(in_data)
            else:
                # Silence is detected
                if self.is_speaking:
                    # We were speaking, now in silence. Check if it's sustained silence.
                    if (time.time() - self.last_speech_time) > self.silence_timeout:
                        # Sustained silence, speech has truly ended
                        self.is_speaking = False
                        if self.speech_buffer:
                            full_speech_audio = b''.join(self.speech_buffer)
                            mic_input_queue.put(full_speech_audio)
                            self.speech_buffer = [] # Clear buffer
                        print("[VAD: Speech Ended]")
                    else:
                        # Still within silence timeout, keep buffering
                        self.speech_buffer.append(in_data)
                else:
                    # Already not speaking, just discard silence
                    pass
        
        return (in_data, pyaudio.paContinue)

class ASRService:
    """
    Handles Speech-to-Text (ASR) using Faster Whisper.
    """
    def __init__(self, model_size="small"):
        print(f"Initializing Faster Whisper model ({model_size})... This may take a moment.")
        #For better accuracy, consider using a larger model size like "base.en" or "small.en".
        # "tiny.en" is fast but has lower accuracy.
        # Use 'cpu' for CPU-only, 'cuda' for NVIDIA GPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Faster Whisper initialized.")

    def transcribe_audio_from_bytes(self, audio_bytes, sample_rate):
        """
        Transcribes raw audio bytes directly using Faster Whisper.
        No temporary file is generated.
        """
        if not audio_bytes:
            return None

        # Convert raw bytes (int16) to a float32 numpy array as expected by Faster Whisper
        audio_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np_float32 = audio_np_int16.astype(np.float32) / 32768.0 # Normalize to [-1, 1]

        print("Transcribing audio...")
        try:
            start_time = time.time() # Start timing ASR
            segments, info = self.model.transcribe(audio_np_float32, beam_size=5) # Pass numpy array directly
            end_time = time.time() # End timing ASR
            print(f"DEBUG: ASR Transcription took {end_time - start_time:.2f} seconds.") # DEBUG LOG

            transcribed_text = " ".join([segment.text for segment in segments])
            print(f"DEBUG: ASR Raw Segments: {list(segments)}") # DEBUG LOG
            print(f"DEBUG: ASR Info: {info}") # DEBUG LOG
            print(f"Transcription complete. Language: {info.language}, Probability: {info.language_probability:.2f}")
            return transcribed_text.strip()
        except Exception as e:
            print(f"Error during ASR transcription: {e}")
            return None


class RAGService:
    """
    Handles PDF processing, text chunking, embedding generation, and retrieval.
    Uses Sentence Transformers for embeddings and cosine similarity for search.
    """
    def __init__(self, pdf_dir, chunk_size, chunk_overlap):
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = [] # Stores {'text': chunk, 'embedding': np.array}
        print("Initializing Sentence Transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Good balance of size/performance
        print("Sentence Transformer initialized.")

    def _load_and_chunk_pdf(self, file_path):
        """Loads a PDF, extracts text, and chunks it."""
        text_content = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_content += page.get_text()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return []

        if not text_content:
            return []

        # Simple chunking logic
        chunks = []
        for i in range(0, len(text_content), self.chunk_size - self.chunk_overlap):
            chunk = text_content[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def process_documents(self):
        """Processes all PDFs in the data directory, chunks them, and generates embeddings."""
        self.documents = [] # Clear existing documents
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in '{self.pdf_dir}'. RAG will not have context.")
            return

        print(f"Processing {len(pdf_files)} PDF(s) in '{self.pdf_dir}'...")
        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_dir, pdf_file)
            chunks = self._load_and_chunk_pdf(file_path)
            if chunks:
                print(f"Generating embeddings for {len(chunks)} chunks from {pdf_file}...")
                chunk_embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
                for i, chunk in enumerate(chunks):
                    self.documents.append({'text': chunk, 'embedding': chunk_embeddings[i]})
        print(f"RAG knowledge base loaded with {len(self.documents)} chunks.")

    def retrieve_context(self, query, top_k=TOP_K_RAG):
        """
        Retrieves the most relevant chunks based on semantic (cosine) similarity.
        This function directly uses the embedding model to find semantically similar chunks,
        addressing the issue of not responding to questions "around the topics" if exact words are not used.
        """
        if not self.documents:
            return ""

        query_embedding = self.embedding_model.encode(query)
        similarities = []
        for doc in self.documents:
            # Calculate cosine similarity between query embedding and document chunk embedding
            similarity = 1 - cosine(query_embedding, doc['embedding'])
            similarities.append((similarity, doc['text']))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Filter for a minimum similarity score (e.g., 0.3) to avoid irrelevant context
        # and take the top_k most relevant chunks.
        relevant_chunks = [text for score, text in similarities if score > 0.3][:top_k]

        if relevant_chunks:
            return "\n".join(relevant_chunks)
        return ""

class LLMService:
    """
    Interacts with the local Ollama LLM.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ollama.Client(host='http://localhost:11434') # Default Ollama host
        print(f"Ollama client initialized for model: {self.model_name}")

    def generate_response(self, prompt, chat_history=None):
        """Generates a response from the LLM."""
        messages = []
        if chat_history:
            for msg in chat_history:
                messages.append({'role': msg['role'], 'content': msg['text']})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = self.client.chat(model=self.model_name, messages=messages, stream=False)
            return response['message']['content']
        except Exception as e:
            print(f"Error communicating with Ollama LLM: {e}")
            return "I'm having trouble generating a response right now. Please check my connection and ensure Ollama is running."

class TTSService:
    """
    Handles Text-to-Speech (TTS) using Piper TTS and plays audio.
    """
    def __init__(self, model_dir, model_basename):
        model_path = os.path.join(model_dir, f"{model_basename}.onnx")
        config_path = os.path.join(model_dir, f"{model_basename}.onnx.json")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Error: Piper TTS model files not found in '{model_dir}'.")
            print(f"Expected: {model_path} and {config_path}")
            print("Please download them from https://github.com/rhasspy/piper/releases and place them in the 'tts_models' directory.")
            self.voice = None
        else:
            print(f"Initializing Piper TTS voice: {model_basename}...")
            # Use PiperVoice class
            self.voice = PiperVoice.load(model_path, config_path) # Use PiperVoice.load
            print("Piper TTS initialized.")

    def _play_audio_stream(self, audio_stream, samplerate):
        """Plays an audio stream in a separate thread, with interruption capability."""
        global stop_speaking_event
        stop_speaking_event.clear() # Clear the event for a new playback

        try:
            # Open an audio stream
            with sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
                for audio_chunk in audio_stream: # Iterate over AudioChunk objects
                    if stop_speaking_event.is_set():
                        # print("TTS playback interrupted.") # Don't print during actual speech
                        break
                    audio_array = audio_chunk.audio_int16_array # Get int16 numpy array
                    stream.write(audio_array)
        except Exception as e:
            print(f"Error during audio playback: {e}")
        finally:
            stop_speaking_event.clear() # Ensure event is cleared after playback/interruption

    def speak(self, text):
        """
        Converts text to speech and plays it directly.
        Starts playback in a new thread to allow interruption.
        """
        if not self.voice:
            print("TTS voice not loaded. Cannot speak.")
            return

        global tts_playback_thread, stop_speaking_event

        # If a previous speech is ongoing, stop it first
        if tts_playback_thread and tts_playback_thread.is_alive():
            stop_speaking_event.set() # Signal to stop
            tts_playback_thread.join(timeout=1) # Wait a bit for it to stop
            if tts_playback_thread.is_alive():
                print("Warning: Previous TTS thread did not terminate gracefully.")

        print(f"Bot speaking: '{text}'")
        # Piper's synthesize method yields AudioChunk objects
        audio_stream_generator = self.voice.synthesize(text)
        
        # Start playback in a new thread
        tts_playback_thread = threading.Thread(
            target=self._play_audio_stream,
            args=(audio_stream_generator, self.voice.config.sample_rate)
        )
        tts_playback_thread.start()

class BotOrchestrator:
    """
    Manages the overall bot conversation flow, integrating all services.
    """
    def __init__(self):
        self.asr_service = ASRService(model_size=WHISPER_MODEL_SIZE)
        self.rag_service = RAGService(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        self.llm_service = LLMService(OLLAMA_MODEL)
        self.tts_service = TTSService(TTS_MODEL_DIR, TTS_MODEL_BASENAME)
        self.mic_stream = MicrophoneStream(MIC_RATE, MIC_CHUNK_SIZE, VAD_THRESHOLD, SILENCE_TIMEOUT_SECONDS) # Pass silence timeout
        self.conversation_history = []
        self.rag_service.process_documents() # Load RAG knowledge base on startup

    def _add_to_history(self, role, text):
        self.conversation_history.append({'role': role, 'text': text})
        # Keep history manageable (e.g., last 10 turns)
        self.conversation_history = self.conversation_history[-10:]

    def _get_chat_history_for_llm(self):
        # Format history for LLM, excluding internal messages like summaries
        llm_history = []
        for msg in self.conversation_history:
            if not msg['text'].startswith('✨'): # Exclude summary/suggestion messages
                llm_history.append(msg)
        return llm_history

    def stop_bot_speaking(self):
        """Signals the TTS service to stop speaking immediately."""
        global stop_speaking_event, tts_playback_thread
        if tts_playback_thread and tts_playback_thread.is_alive():
            stop_speaking_event.set()
            print("\n[Bot's speech interrupted by user]")
            # Optionally wait for the thread to finish, but not strictly necessary for immediate stop
            # tts_playback_thread.join(timeout=1)

    def process_text_input(self, user_text):
        """Processes user text input (from console or ASR)."""
        self._add_to_history('user', user_text)
        print(f"\nUser: {user_text}")

        # 1. RAG: Retrieve context
        retrieved_context = self.rag_service.retrieve_context(user_text)
        if retrieved_context:
            print(f"--- RAG Context Found ---\n{retrieved_context}\n-------------------------")
            prompt = f"Based on the following context and the user's query, provide a concise and helpful answer. If the context is not directly relevant, use general knowledge.\n\nContext: {retrieved_context}\n\nUser Query: {user_text}"
        else:
            prompt = f"User Query: {user_text}"

        # 2. LLM: Generate response
        llm_response = self.llm_service.generate_response(prompt, chat_history=self._get_chat_history_for_llm())
        self._add_to_history('assistant', llm_response)
        print(f"Bot: {llm_response}")

        # 3. TTS: Speak response
        self.tts_service.speak(llm_response)

    def summarize_conversation(self):
        """Summarizes the current conversation using the LLM."""
        if len(self.conversation_history) < 2:
            print("Not enough conversation to summarize.")
            return

        self.stop_bot_speaking() # Stop any ongoing speech before summarizing
        print("\nBot is summarizing the conversation...")
        conversation_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in self.conversation_history if not msg['text'].startswith('✨')])
        prompt = f"Summarize the following conversation concisely:\n\n{conversation_text}"
        
        summary = self.llm_service.generate_response(prompt)
        summary_message = f"✨ Conversation Summary: {summary}"
        self._add_to_history('assistant', summary_message)
        print(f"Bot: {summary_message}")
        self.tts_service.speak("Here's a summary of our conversation.")

    def suggest_next_topics(self):
        """Suggests follow-up questions/topics using the LLM."""
        if len(self.conversation_history) < 1:
            print("Start a conversation first to get topic suggestions.")
            return

        self.stop_bot_speaking() # Stop any ongoing speech before suggesting topics
        print("\nBot is suggesting next topics...")
        conversation_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in self.conversation_history if not msg['text'].startswith('✨')])
        prompt = f"Based on the following conversation, suggest 3-5 concise follow-up questions or related topics the user might be interested in. Format as a numbered list.\n\nConversation:\n{conversation_text}"
        
        suggestions = self.llm_service.generate_response(prompt)
        suggestions_message = f"✨ Suggested Topics:\n{suggestions}"
        self._add_to_history('assistant', suggestions_message)
        print(f"Bot: {suggestions_message}")
        self.tts_service.speak("Here are some topics you might be interested in.")

    def _listen_for_speech_thread(self):
        """Thread that continuously listens for speech and processes it."""
        global mic_input_queue, vad_active, speech_detected_event

        while True:
            if not vad_active: # Only listen if VAD is active
                time.sleep(0.1)
                continue

            # Wait for speech to be detected by VAD callback
            speech_detected_event.wait(timeout=0.5) # Wait with a timeout
            if speech_detected_event.is_set():
                speech_detected_event.clear() # Reset event

                # If bot is speaking, interrupt it
                global tts_playback_thread
                if tts_playback_thread and tts_playback_thread.is_alive():
                    self.stop_bot_speaking()
                    # Wait briefly for TTS thread to acknowledge stop
                    time.sleep(0.2) 

                # Wait for the full speech segment to be collected by VAD
                try:
                    audio_bytes = mic_input_queue.get(timeout=30) # Increased timeout for ASR processing
                    print(f"DEBUG: Received {len(audio_bytes)} bytes from microphone queue.") # DEBUG LOG
                    if audio_bytes:
                        transcribed_text = self.asr_service.transcribe_audio_from_bytes(audio_bytes, MIC_RATE)
                        print(f"DEBUG: ASR Transcribed Text: '{transcribed_text}'") # DEBUG LOG
                        if transcribed_text and transcribed_text.strip(): # Check if transcription is not empty or just whitespace
                            print(f"[ASR Result]: {transcribed_text}")
                            self.process_text_input(transcribed_text) # Process immediately
                        else:
                            print("[ASR]: No speech transcribed or error, or transcription was empty.")
                            # Bot says it doesn't hear anything
                            self.tts_service.speak("I don't hear anything, please ask again.")
                    else:
                        print("[ASR]: Received empty audio bytes from queue.") # DEBUG LOG
                except queue.Empty:
                    print("[ASR]: No complete speech segment received within timeout.")
                    self.tts_service.speak("I didn't catch that, please speak again.") # Added specific message for timeout
                except Exception as e:
                    print(f"[ASR Thread Error]: {e}")
                    self.tts_service.speak(f"An error occurred during speech processing: {e}. Please try again.") # Speak out error
            else:
                # No speech detected within timeout, continue listening
                pass


    def start_conversation_loop(self):
        """Starts the main conversation loop."""
        global vad_active

        # Start microphone stream and VAD
        self.mic_stream.start_stream()
        vad_active = True # Activate VAD listening

        # Start a separate thread to listen for and process speech detected by VAD
        speech_listener_thread = threading.Thread(target=self._listen_for_speech_thread, daemon=True)
        speech_listener_thread.start()

        print("\nTensorGo Bot Ready! Speak into your microphone or type commands.")
        print("Type 'summarize' to get a conversation summary.")
        print("Type 'suggest' to get topic suggestions.")
        print("Type 'stop' to explicitly interrupt bot's speech (VAD handles most).")
        print("Type 'quit' to exit.")
        print("Type 'help' for options.")

        while True:
            # Main loop primarily for commands, speech input is handled by the listener thread
            command = input("\n> ").strip().lower()

            if command == 'quit':
                self.stop_bot_speaking() # Ensure bot stops speaking before exiting
                self.mic_stream.stop_stream() # Stop microphone
                vad_active = False # Deactivate VAD
                print("Exiting bot. Goodbye!")
                break
            elif command == 'summarize':
                self.summarize_conversation()
            elif command == 'suggest':
                self.suggest_next_topics()
            elif command == 'stop':
                self.stop_bot_speaking()
            elif command == 'help':
                print("\nOptions:")
                print("  Speak into your microphone to chat.")
                print("  'summarize' - Get a summary of the conversation.")
                print("  'suggest' - Get suggestions for next topics.")
                print("  'stop' - Explicitly interrupt the bot if it's currently speaking.")
                print("  'quit' - Exit the bot.")
            else:
                # Process typed commands directly
                self.stop_bot_speaking() # Stop current speech if new text input
                self.process_text_input(command)

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    os.makedirs(TTS_MODEL_DIR, exist_ok=True)

    bot = BotOrchestrator()
    try:
        bot.start_conversation_loop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down bot.")
        bot.mic_stream.stop_stream() # Stop microphone stream
        vad_active = False # Crucial: Deactivate VAD listening
        bot.stop_bot_speaking()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        bot.mic_stream.stop_stream()
        vad_active = False # Crucial: Deactivate VAD listening
        bot.stop_bot_speaking()
