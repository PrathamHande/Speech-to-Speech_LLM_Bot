# 🗣️ Local Speech-to-Speech RAG Chatbot

This project demonstrates a fully functional, end-to-end, real-time AI assistant that runs entirely on your local machine. It combines state-of-the-art open-source technologies to create a seamless, voice-controlled chatbot with Retrieval-Augmented Generation (RAG) capabilities, ensuring privacy and rapid response times.

The entire system is orchestrated by a central "Python Hub" that manages the flow of information between all services.

## ✨ Features

* **100% Local & Private:** No data is sent to external cloud APIs. All models run on your machine.
* **Speech-to-Speech:** Converse naturally using your microphone and speakers.
* **Real-time Interaction:** Handles user queries and responds within a few seconds.
* **User Interruption:** The bot's speech can be interrupted at any time by simply speaking.
* **Retrieval-Augmented Generation (RAG):** Answers are grounded in your custom knowledge base (PDF documents).
* **Conversation Summary:** Summarize the entire conversation with a single command.
* **Topic Suggestions:** Get intelligent suggestions for follow-up questions based on the chat history.

## 🛠️ Technology Stack

| Service | Technology Used | Role | 
| :--- | :--- | :--- | 
| **Orchestrator** | Python | The core logic that binds all services together. | 
| **Local LLM** | **Ollama** with **Mistral** | Generates intelligent responses. | 
| **Speech-to-Text (ASR)** | **Faster Whisper** | Transcribes spoken input into text. | 
| **Text-to-Speech (TTS)** | **Piper TTS** | Synthesizes text responses into natural-sounding audio. | 
| **Voice Detection (VAD)** | **Silero VAD** with `pyaudio` | Detects speech to start/stop listening and handle interruptions. | 
| **RAG Embeddings** | **Sentence Transformers** | Creates semantic embeddings for document chunks. | 
| **Document Processing** | **PyMuPDF (`fitz`)** | Extracts text from PDF files. | 

## 🚀 Getting Started

Follow these steps to set up and run the chatbot on your machine.

### Prerequisites

* Python 3.8 or higher.
* A working microphone and speakers.
* An active internet connection for the initial model downloads.

### Step 1: Install Ollama & Download the LLM

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Open a terminal and pull the Mistral model:
   
```
ollama pull mistral
```

3. Ensure the Ollama server is running in the background.

### Step 2: Clone the Repository & Set up Python Environment

1. Clone this project to your local machine:
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv/Scripts/activate
```


### Step 3: Install Python Dependencies

Install the required libraries from the `requirements.txt` file.


### Step 4: Download TTS Voice Model

1. Go to the [Piper TTS releases page](https://github.com/rhasspy/piper/releases).
2. Download both the `.onnx` and `.onnx.json` files for your desired voice (e.g., `en_US-lessac-high.onnx` and `en_US-lessac-high.onnx.json`).
3. Create a folder named `tts_models` in your project directory.
4. Place the two downloaded files inside the `tts_models` folder.

### Step 5: Prepare the RAG Knowledge Base

1. Create a folder named `data` in your project directory.
2. Place any PDF documents you want the bot to query inside this `data` folder.

## 🏃 Running the Bot

With all the prerequisites in place, run the main Python script from your terminal:


```bash
python bot_hub.py
```

The bot will perform a one-time setup (downloading models, creating embeddings) and then become ready to interact.

## 🎙️ How to Use

* **Speak to Chat:** Simply speak into your microphone after you see the `Bot Ready!` message. The bot will automatically detect your speech and respond.
* **Interrupt:** Start speaking at any time while the bot is talking to interrupt its current response.
* **Commands:**
  * `summarize`: Provides a summary of the current conversation.
  * `suggest`: Offers follow-up topics or questions.
  * `stop`: Explicitly stops the bot's speech.
  * `quit`: Shuts down the bot.

## 🤝 Contributing

This project is a great foundation for building more advanced conversational AI.

## 📂 Directory Structure


Speech-to-Speech_LLM_Bot/
├── data/
│   └── your-document.pdf
├── tts_models/
│   ├── en_US-lessac-high.onnx
│   └── en_US-lessac-high.onnx.json
├── bot_hub
└── requirements.txt 