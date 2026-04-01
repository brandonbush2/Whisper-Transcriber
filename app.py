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

# ========================= Youtube Downloader ===========================================================================

def download_youtube_audio(url: str):
    #creating a persistent temporary file
    tempfile = tempfile.NamedTemporaryFile(suffix = ".mp3", delete = False)
    temp_path = tempfile.name
    tempfile.close() #close so that yt_dlp can write to it

    ydl_opts = {
                'format':'bestaudio/best',
                'outtmpl':temp_path.replace(".mp3", ".mp3"), #yt_dlp will add the extension
                'postprocessors':[{
                    'key': 'FFmpegExtractAudio',
                    'preferredcode': 'mp3',
                    'preferredquality': '192'
                                  }],
                'quiet': True,
                'no_warnings': True
     }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        #ydl normally saves as .mp3
        final_path = temp_path if os.path.exists(temp_path) else temp_path.replace(".mp3", ".mp3")

        if not os.path.exists(final_path):
            #fallback: Look for any audio file in the location
            dir_name = os.path.dirname(temp_path)
            for f in os.listdir(dirname):
                if f.endswith((".mp3",".m4a", ".wav")):
                    final_path = os.path.join(dirname, f)
                    break
        return final_path
    except Exception as e:
        print(f"Download Failed: {e}")
        raise