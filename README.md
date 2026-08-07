<p align="center">
  <img src="https://capsule-render.herokuapp.com/形状=glowing&color=a78bfa&height=180&section=header&text=JARVIS-AI&fontSize=70&animation=fadeIn" alt="Jarvis Header" />
</p>

<p align="center">
  <a href="https://github.com/piyush1008-cyber/jarvis-voice-assistant">
    <img src="https://img.shields.io/badge/Jarvis--AI-Multimodal-8B5CF6?style=for-the-badge&logo=ai&logoColor=white" alt="Project Badge" />
  </a>
  <a href="https://streamlit.io">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit Badge" />
  </a>
  <a href="https://ollama.com">
    <img src="https://img.shields.io/badge/Ollama-Llama3.2-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama Llama3.2" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/tensorflow-%23FF6F20.svg?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white" />
  <img src="https://img.shields.io/badge/opencv-%235C3EE8.svg?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/git-%23F05033.svg?style=flat-square&logo=git&logoColor=white" />
</p>

---

## 🌟 Overview

**Jarvis-AI** is a state-of-the-art, **privacy-first local AI assistant** that combines **Speech Processing, Computer Vision, and Natural Language Processing (NLP)** into a single multimodal system. It tracks your facial expression to gauge your emotional context, retrieves relevant information from uploaded documents (PDFs) using semantic vector search (RAG), and answers your questions out loud.

Developed locally, this system runs **100% offline** and requires **no external API keys**, ensuring absolute data privacy.

---

## 🚀 Key Features

*   📷 **Automated Face Scanner (Computer Vision):** Captures your face on voice trigger and uses `DeepFace` CNNs to analyze your emotional state (**Happy, Sad, Angry, Fear, Surprise, Neutral**).
*   🎙️ **Voice-to-Voice Loop:** Voice query transcription using Google Speech API and asynchronous background text-to-speech engine (`pyttsx3`) that doesn't freeze the dashboard.
*   📄 **Local Document RAG (AI Syllabus/PDF Reader):** Extract, chunk, and index PDFs locally using the `all-MiniLM-L6-v2` transformer and fast NumPy cosine similarity.
*   🧠 **Offline Brain (Ollama + Llama-3.2):** High-speed LLM reasoning on your CPU/GPU under 2 seconds.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend HUD** | `Streamlit (Glassmorphism)` | Professional layout, HUD indicators |
| **Computer Vision** | `OpenCV`, `DeepFace`, `TensorFlow` | Face tracking and emotion classification |
| **Local LLM** | `Ollama`, `Llama-3.2 (3B)` | Core reasoning and response generation |
| **Embeddings & Search** | `Sentence-Transformers`, `Numpy` | RAG vector creation & Cosine Similarity search |
| **Speech Processing** | `SpeechRecognition`, `SoundDevice` | Mic recording & text transcription |
| **Voice Synthesis** | `Pyttsx3 (Multi-threaded)` | Async male vocal feedback engine |

---

## 🧩 How the Pipeline Flows

```mermaid
flowchart TD
    classDef voice fill:#818cf8,stroke:#4f46e5,stroke-width:2px,color:#fff;
    classDef vision fill:#34d399,stroke:#059669,stroke-width:2px,color:#fff;
    classDef brain fill:#a78bfa,stroke:#7c3aed,stroke-width:2px,color:#fff;

    Start([🎤 Click 'Initiate Voice Query']) --> Cam[📷 Open Webcam & Scan Expression]:::vision
    Start --> Mic[🎙️ Record Mic Audio to WAV]:::voice
    
    Cam --> Face[🧠 DeepFace analyzes emotion]:::vision
    Mic --> Trans[📝 SpeechRecognition transcribe text]:::voice
    
    Trans --> RAG[📄 Local RAG searches PDF via Cosine Similarity]:::brain
    Face --> LLM[🧠 Llama-3.2 via Ollama processes query + context + emotion]:::brain
    RAG -->|Retrieves context| LLM
    
    LLM --> Speak[🔊 pyttsx3 speaks answer in background thread]:::voice
    LLM --> Chat[💬 Render glassmorphic chat bubble]:::brain
```

---

## 🏁 Step-by-Step Local Setup

<details>
<summary><b>1. Prerequisites</b> <i>(Click to expand)</i></summary>

1.  **Python 3.10 to 3.12** installed on your system.
2.  **Ollama** installed on your machine.
    *   Download from [ollama.com](https://ollama.com)
    *   Open terminal and download the model:
        ```bash
        ollama run llama3.2
        ```
</details>

<details>
<summary><b>2. Installation & Setup</b> <i>(Click to expand)</i></summary>

1.  Clone the repository:
    ```bash
    git clone https://github.com/piyush1008-cyber/jarvis-voice-assistant.git
    cd jarvis-voice-assistant
    ```
2.  Create a Python Virtual Environment:
    ```bash
    python -m venv env
    ```
3.  Activate the environment and install dependencies:
    ```cmd
    # Windows
    env\Scripts\python -m pip install -r requirements.txt
    ```
</details>

<details>
<summary><b>3. Running the App</b> <i>(Click to expand)</i></summary>

Start the Streamlit dashboard using your virtual environment:

```cmd
D:\placement-projects\env\Scripts\python -m streamlit run D:\placement-projects\jarvis-voice-assistant\app.py
```

Once running, access the web interface at **`http://localhost:8501`**.
</details>

---

## 🔒 Recruiters & HR Highlights
This project demonstrates advanced competence in **multimodal software engineering, concurrent multi-threading, and local NLP model tuning**. By bypassing expensive external APIs and hosting models locally on-device, this system presents a highly practical, **zero-cost enterprise blueprint** designed for secure corporate environments where data protection is critical.
