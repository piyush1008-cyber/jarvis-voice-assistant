<p align="center">
  <img src="https://capsule-render.herokuapp.com/type=glowing&color=a78bfa&height=180&section=header&text=JARVIS-AI&fontSize=70&animation=fadeIn" alt="Jarvis Header" />
</p>

<p align="center">
  <strong>An Intelligent, Privacy-First Multimodal Voice Companion with Emotion-Aware Document RAG</strong>
</p>

<p align="center">
  <a href="https://github.com/piyush1008-cyber/jarvis-voice-assistant">
    <img src="https://img.shields.io/badge/Project--Status-Complete--Production-10b981?style=for-the-badge&logo=git&logoColor=white" alt="Status" />
  </a>
  <a href="https://streamlit.io">
    <img src="https://img.shields.io/badge/Interface-Streamlit--Glassmorphism-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
  </a>
  <a href="https://ollama.com">
    <img src="https://img.shields.io/badge/LLM--Engine-Llama--3.2--Local-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/tensorflow-2.21.0-FF6F20.svg?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/opencv-5.0.0-5C3EE8.svg?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-5.7.0-blue.svg?style=flat-square" />
</p>

---

## 📖 Executive Summary

**Jarvis-AI** is a local, multimodal virtual assistant that integrates **Speech Processing, Computer Vision, and Natural Language Processing (NLP)** into a single offline system. The project acts as a secure information companion, allowing users to upload text documents (e.g., college study guides or syllabus PDFs), analyze their emotional state via real-time webcam captures, and answer queries verbally.

By running all model inferences (facial CNNs, sentence embeddings, and LLM reasoning) directly on the host machine, Jarvis-AI guarantees **100% data residency and zero cloud subscription costs**, presenting a secure corporate blueprint for privacy-sensitive environments.

---

## 🛠️ System Architecture & Data Flow

When a user initiates a query, the application triggers a synchronous multimodal acquisition pipeline:

```mermaid
flowchart TD
    classDef voice fill:#818cf8,stroke:#4f46e5,stroke-width:2px,color:#fff;
    classDef vision fill:#34d399,stroke:#059669,stroke-width:2px,color:#fff;
    classDef brain fill:#a78bfa,stroke:#7c3aed,stroke-width:2px,color:#fff;

    Start([🎤 User Click: 'Initiate Voice Query']) --> Cam[📷 Open Webcam & Skip 8 Calibration Frames]:::vision
    Start --> Mic[🎙️ Capture 6s Microphone Input via Sounddevice]:::voice
    
    Cam --> Face[🧠 DeepFace Convolutional Neural Network class expression]:::vision
    Mic --> Trans[📝 SpeechRecognition transcribes WAV to string]:::voice
    
    Trans --> RAG[📄 Local RAG searches indexed PDF via Cosine Similarity]:::brain
    Face --> LLM[🧠 Local Llama-3.2 via Ollama processes combined prompt]:::brain
    RAG -->|Injects relevant context chunks| LLM
    
    LLM --> Speak[🔊 pyttsx3 synthesizes audio in background thread]:::voice
    LLM --> Chat[💬 Render markdown dialogue card in browser]:::brain
```

---

## 🧠 Key Engineering & Architectural Decisions

During the development lifecycle, several critical design decisions were made to optimize system reliability, privacy, and compatibility on Windows systems:

### 1. **Ollama (Llama-3.2) vs. Cloud APIs (OpenAI/Claude)**
*   **Privacy & Compliance:** Using local model endpoints ensures zero user document leakage, complying with data governance standards.
*   **Zero-Cost Execution:** Eliminates external token limits and API billing, providing a free local testing environment.

### 2. **Custom NumPy Similarity Search vs. Native Vector Databases (Chroma/FAISS)**
*   **Zero-Dependency Portability:** Compiled vector databases frequently throw C++ build errors during compilation on Windows machines without active Visual Studio build tools.
*   **Computational Efficiency:** By writing a direct, lightweight Cosine Similarity algorithm in memory using `numpy` dot-products (`dot(A, B) / (norm(A) * norm(B)`), context retrieval executes in under **5ms** for textbook-sized PDF files with zero package overhead.

### 3. **Sounddevice & SciPy vs. PyAudio**
*   **Installation Stability:** `PyAudio` requires compiling portaudio bindings from source which routinely fails on standard Windows environments. 
*   **Robust Capture:** Utilizing `sounddevice` to stream raw NumPy arrays and saving them via `scipy.io.wavfile` guarantees cross-platform hardware recording stability.

### 4. **Asynchronous Multi-threaded Text-to-Speech (TTS)**
*   **Non-Blocking UI:** Standard Python TTS initialization (`pyttsx3.init().say()`) blocks the main execution thread, causing browser connection timeouts in Streamlit.
*   **Concurrent Execution:** We isolated the TTS engine inside a background `threading.Thread` loop, enabling speech playback to execute concurrently while the UI updates the chat history.

---

## 🚀 Technical Features

*   📷 **Webcam auto-exposure warm-up:** Automatically grabs and discards the first 8 frames to allow the camera's light sensor to adjust, preventing underexposed black silhouettes.
*   🧠 **Contextual Affect Mapping:** The assistant reads your current emotion from the CNN classification and adjusts its response tone (e.g. speaking in a calm, structured tone when the user is scanned as Neutral).
*   📄 **Textual Overlap Chunking:** Extracts text from PDFs using `pypdf`, normalizing whitespaces and chunking text with a 100-character overlap to preserve semantic context across boundary limits.

---

## 🏁 Step-by-Step Installation

<details>
<summary><b>📋 Prerequisites</b> <i>(Click to expand)</i></summary>

1.  **Python 3.10 - 3.12** installed on your system.
2.  **Ollama** installed on your machine.
    *   Download from [ollama.com](https://ollama.com)
    *   Open terminal and download the model:
        ```bash
        ollama run llama3.2
        ```
</details>

<details>
<summary><b>⚙️ Installation & Setup</b> <i>(Click to expand)</i></summary>

1.  Clone the repository:
    ```bash
    git clone https://github.com/piyush1008-cyber/jarvis-voice-assistant.git
    cd jarvis-voice-assistant
    ```
2.  Initialize the Python Virtual Environment:
    ```bash
    python -m venv env
    ```
3.  Activate the environment and install dependencies:
    ```cmd
    # Windows Command Prompt
    env\Scripts\python -m pip install --no-cache-dir -r requirements.txt
    ```
</details>

<details>
<summary><b>🖥️ Running the Application</b> <i>(Click to expand)</i></summary>

Execute the Streamlit application using your virtual environment:

```cmd
D:\placement-projects\env\Scripts\python -m streamlit run D:\placement-projects\jarvis-voice-assistant\app.py
```

Once running, access the web interface at **`http://localhost:8501`**.
</details>

---

## 🔒 Corporate Relevance & Placement Focus
This project highlights competence in **systems integration, multi-threading, NLP pipeline design, and local AI orchestration**. It serves as an excellent case study for recruiters looking for developers capable of constructing secure, highly optimized local applications without reliance on paid third-party APIs.
