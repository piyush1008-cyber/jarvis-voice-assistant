import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Import local engines
from emotion_detector import detect_emotion
from rag_engine import LocalRAGEngine
from assistant import (
    record_audio, 
    transcribe_audio, 
    query_ollama, 
    format_jarvis_prompt, 
    speak
)

# Page configuration
st.set_page_config(
    page_title="Jarvis-AI: Multimodal Voice Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for Premium, Disciplined, and Highly Aesthetic UI
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global settings */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Constrain layout width and center container on widescreen monitors */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1160px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }
    
    /* Hide Streamlit default branding to make it look like a custom application */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header container styling */
    .header-panel {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.4));
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* System Cards */
    .system-card {
        background-color: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Status Badge Styling */
    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }
    .status-online {
        background-color: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .status-offline {
        background-color: rgba(239, 68, 68, 0.1);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }
    
    /* Emotion Display HUD */
    .emotion-hud {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Dynamic Chat Bubbles */
    .chat-container {
        max-height: 480px;
        overflow-y: auto;
        padding-right: 5px;
    }
    
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .user-bubble {
        background-color: #1e293b;
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-bottom-right-radius: 4px;
        margin-left: auto; /* Align user messages to the right */
        text-align: left;
    }
    
    .user-meta {
        color: #818cf8;
        font-weight: 700;
        font-size: 0.75rem;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .jarvis-bubble {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(167, 139, 250, 0.08));
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-bottom-left-radius: 4px;
        margin-right: auto; /* Align Jarvis messages to the left */
    }
    
    .jarvis-meta {
        color: #a78bfa;
        font-weight: 700;
        font-size: 0.75rem;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Listening Pulse Indicator */
    .listening-status {
        background-color: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        color: #f87171;
        animation: pulse-effect 1.5s infinite;
    }
    @keyframes pulse-effect {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    
    /* Custom buttons and forms elements overrides */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = LocalRAGEngine()
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# Check Ollama Server Availability
try:
    import requests
    res = requests.get("http://localhost:11434/", timeout=2)
    ollama_badge = '<span class="status-badge status-online">Ollama Server: Connected</span>'
except Exception:
    ollama_badge = '<span class="status-badge status-offline">Ollama Server: Disconnected</span>'

# Header Presentation Panel
st.markdown(f"""
<div class="header-panel">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="title-text">Jarvis-AI Workspace</h1>
            <p class="subtitle-text">Multimodal Voice Intelligence with Emotion-Aware Document Retrieval (RAG)</p>
        </div>
        <div>
            {ollama_badge}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Layout: Split Sidebar (Left) vs Main Interaction Area (Right)
col_sidebar, col_main = st.columns([1, 1.8], gap="medium")

with col_sidebar:
    # Card 1: Camera & Emotion Tracking
    st.markdown('<div class="system-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📷 Automated Camera Sensor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style='font-size:0.85rem; color:#94a3b8; line-height: 1.4; margin-bottom: 12px;'>
        This system is equipped with an <b>Automated Face Scanner</b>. 
    </p>
    <p style='font-size:0.85rem; color:#64748b; line-height: 1.4;'>
        When you click the voice query button, your webcam will briefly activate in the background to capture your mood, keeping your interaction hands-free.
    </p>
    """, unsafe_allow_html=True)
    
    # Render Emotion HUD card
    color_map = {
        "Happy": "#34d399", "Sad": "#60a5fa", "Angry": "#f87171",
        "Fear": "#c084fc", "Surprise": "#f472b6", "Neutral": "#94a3b8"
    }
    emotion_color = color_map.get(st.session_state.current_emotion, "#94a3b8")
    
    st.markdown(f"""
    <div class="emotion-hud" style="margin-top: 1rem;">
        <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Last Scanned Mood</span>
        <h2 style="color: {emotion_color}; font-size: 2.2rem; font-weight:800; margin: 5px 0;">{st.session_state.current_emotion}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # End of Card 1Card 1

    # Card 2: Document RAG Configurations
    st.markdown('<div class="system-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📄 Document Intelligence (RAG)</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF Reference Files", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF sections..."):
                success = st.session_state.rag_engine.process_pdf(uploaded_file)
                if success:
                    st.session_state.pdf_name = uploaded_file.name
                    st.success("PDF processed and indexed!")
                else:
                    st.error("Failed to extract text. This happens if the PDF is scanned or image-only (like Canva templates). Please upload a standard text PDF.")
                    
    if st.session_state.pdf_name:
        st.markdown(f"""
        <div style="background-color: rgba(52, 211, 153, 0.05); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 10px; padding: 12px; margin-top: 1rem;">
            <span style="font-size: 0.75rem; color: #34d399; font-weight:700;">Active Context Document:</span>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; font-weight: 500; word-break: break-all;">📄 {st.session_state.pdf_name}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 10px; padding: 12px; margin-top: 1rem;">
            <span style="font-size: 0.75rem; color: #fbbf24; font-weight:700;">Active Context:</span>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #fbbf24;">No reference file loaded. Operating on general intelligence.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # End of Card 2


# ==========================================
# RIGHT COLUMN: SYSTEM WORKSPACE (MAIN)
# ==========================================
with col_main:
    # Card 3: Interactive Controller
    st.markdown('<div class="system-card" style="padding-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎙️ Conversation Console</div>', unsafe_allow_html=True)
    
    col_button, col_hint = st.columns([1, 2])
    
    trigger_voice = False
    with col_button:
        if st.button("🎤 Initiate Voice Query", use_container_width=True):
            trigger_voice = True
            
    with col_hint:
        st.markdown("<p style='font-size:0.85rem; color:#64748b; margin-top:8px;'>Click to record microphone for 6 seconds. Jarvis will answer using PDF context and match your current mood.</p>", unsafe_allow_html=True)

    # Active Voice Listening Flow
    if trigger_voice:
        status_box = st.empty()
        status_box.markdown('<div class="listening-status" style="color: #60a5fa; background-color: rgba(96, 165, 250, 0.08); border-color: rgba(96, 165, 250, 0.2);">📷 Scanning facial expressions...</div>', unsafe_allow_html=True)
        
        # Capture photo automatically using cv2 in the background
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Warm up camera sensor (skip first 8 frames for auto-exposure calibration)
                for _ in range(8):
                    cap.grab()
                ret, frame = cap.retrieve()
                if ret:
                    frame = cv2.flip(frame, 1)
                    dominant_emotion, analysis = detect_emotion(frame)
                    st.session_state.current_emotion = dominant_emotion
                    print(f"[CAMERA DIAGNOSTIC] Detected Emotion: {dominant_emotion} | Raw: {analysis}")
                cap.release()
        except Exception as e:
            print(f"Error during automatic facial scan: {e}")
            
        status_box.markdown('<div class="listening-status">🔴 Jarvis is listening... Speak your query clearly.</div>', unsafe_allow_html=True)
        
        # 1. Record WAV file
        if record_audio(duration=6):
            status_box.markdown('<div class="listening-status" style="color: #818cf8; background-color: rgba(99, 102, 241, 0.08); border-color: rgba(99, 102, 241, 0.2);">⚡ Transcribing recorded voice...</div>', unsafe_allow_html=True)
            
            # 2. Transcribe WAV file
            query_text = transcribe_audio()
            
            if query_text.strip():
                # 3. Vector search PDF chunks
                context_str = None
                if st.session_state.pdf_name:
                    search_results = st.session_state.rag_engine.search(query_text, top_k=2)
                    if search_results:
                        context_str = "\n---\n".join([r['text'] for r in search_results])
                
                # 4. Generate LLM prompt (Inject context and mood)
                user_emotion = st.session_state.current_emotion
                assistant_prompt = format_jarvis_prompt(query_text, user_emotion, context_str)
                
                # Query local Ollama server
                status_box.markdown('<div class="listening-status" style="color: #c084fc; background-color: rgba(167, 139, 250, 0.08); border-color: rgba(167, 139, 250, 0.2);">🧠 Jarvis is formulating response...</div>', unsafe_allow_html=True)
                assistant_response = query_ollama(assistant_prompt)
                
                # 5. Play voice asynchronously
                speak(assistant_response)
                
                # Append log history
                st.session_state.chat_history.append({
                    "role": "user",
                    "text": query_text,
                    "emotion": user_emotion
                })
                st.session_state.chat_history.append({
                    "role": "jarvis",
                    "text": assistant_response
                })
                status_box.empty()
            else:
                status_box.empty()
                st.warning("Speech recognition could not capture any audio. Please verify your microphone is connected.")
                speak("I'm sorry, I could not hear any speech. Could you please try again?")
        else:
            status_box.empty()
            st.error("Error: Failed to open default audio recording stream.")
            
    st.markdown('</div>', unsafe_allow_html=True) # End of Card 3

    # Card 4: Dialogues & Logs (Messaging Style UI)
    st.markdown('<div class="system-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💬 Dialogue Transcripts & Logs</div>', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("<p style='color:#64748b; font-style:italic; text-align:center; padding: 2rem 0;'>Dialogue console empty. Click 'Initiate Voice Query' to talk to Jarvis!</p>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        # Render dialogs sequentially
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                    <div class="user-meta">User &bull; Mood: {message['emotion']}</div>
                    <div>{message['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble jarvis-bubble">
                    <div class="jarvis-meta">Jarvis AI</div>
                    <div>{message['text']}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # End of Card 4
