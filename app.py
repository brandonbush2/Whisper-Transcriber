import streamlit as st
import torch
from transformers import pipeline
import yt_dlp
import tempfile
import os
from pathlib import Path

# ========================= CONFIG ===========================================================================
st.set_page_config(page_title="🎙️ Whisper Transcription", page_icon="🎙️", layout="centered")

st.title("🎙️ Whisper Audio & YouTube Transcriber")

# ========================= MODEL LOADING ===========================================================================
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    st.info(f"Loading model on **{device.upper()}**...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",           # Best balance for 4GB VRAM
        # model="distil-whisper/distil-small.en",  
        dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "en"}      # Change to None if you need multilingual
    )
    return pipe

pipe = load_whisper_model()

