import streamlit as st
import torch
from transformers import pipeline
import yt_dlp
import tempfile
import os
from pathlib import Path

# ========================= CONFIG ===========================================================================
st.set_page_config(page_title="Whisper Transcription", page_icon="🎙️", layout="centered")

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
    #Extracting the title without downloading
    with yt_dlp.YoutubeDL({'quiet':True, 'no_warnings': True}) as ydl:
        info = ydl.extract_info(url, download = False)
        title = info.get('title', 'youtube_video')


    #creating a persistent temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix = ".mp3", delete = False)
    temp_path = temp_file.name
    temp_file.close() #close so that yt_dlp can write to it

    ydl_opts = {
                'format':'bestaudio/best',
                'outtmpl':temp_path.replace(".mp3", ""), #yt_dlp will add the extension
                'postprocessors':[{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }],
                'quiet': True,
                'no_warnings': True
     }
     # Downloading to a tempfile
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        #ydl normally saves as .mp3
        final_path = temp_path if os.path.exists(temp_path) else temp_path.replace(".mp3", ".mp3")

        if not os.path.exists(final_path):
            #fallback: Look for any audio file in the location
            dir_name = os.path.dirname(temp_path)
            for f in os.listdir(dir_name):
                if f.endswith((".mp3",".m4a", ".wav")):
                    final_path = os.path.join(dir_name, f)
                    break
        return final_path
    except Exception as e:
        print(f"Download Failed: {e}")
        raise

# ========================= UI ===========================================================================
tab1, tab2 = st.tabs(["📁 Upload File", "▶️ YouTube URL"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload File",
        type = [".mp3","wav", "m4a", "ogg", "mp4", "mov", "avi"]
    )

    if uploaded_file:
        if st.button("Transcribe Uploaded File", type = "primary"):
            with st.spinner("Transcribing file....(this may take 1-3 minutes depending on length)"):
                with tempfile.NamedTemporaryFile(delete = False, suffix = Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    result = pipe(
                                tmp_path,
                                chunk_length_s = 30, #split into 30s chunks
                                stride_length_s = 5, #Overlap between chunks by 5 for better accuracy
                                batch_size = 8,      #Good balance for rtx 3050
                                return_timestamps = True 
                    )
                    transcription = result["text"]
                    st.success("✅ Transcription Complete!")
                    st.subheader("Transcription")
                    st.write(transcription)

                    #base name of the uploaded txt file
                    base_name = Path(uploaded_file.name).stem

                    #Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label = "📥 Download as .txt",
                            data = transcription,
                            file_name = f"{base_name}.txt",
                            mime = "text/plain"
                        )
                    with col2:
                        st.download_button(
                            label = "📥 Download as .srt",
                            data = transcription,
                            file_name = f"{base_name}.srt",
                            mime = "application/x-subrip"
                        )
                finally:
                    os.unlink(tmp_path) #Even if transcription fails, temporary file is deleted

with tab2:
    youtube_url = st.text_input("Paste YouTube URL here: ", placeholder = "https://www.youtube.com/watch?v=...")

    if youtube_url and st.button("Transcribe YouTube Video", type = "primary"):
        with st.spinner("Downloading audio from YouTube..."):
            try:
                audio_path = download_youtube_audio(youtube_url)
                st.success("✅ Audio downloaded successfully")
            except Exception as e:
                st.error(f"Download Failed: {e}")
                st.stop()

        with st.spinner("Transcribing with Whisper (this can take 1–4 minutes)..."):
            try:
                result = pipe(
                                audio_path,
                                chunk_length_s = 30, #split into 30s chunks
                                stride_length_s = 5, #Overlap between chunks by 5 for better accuracy
                                batch_size = 8,      #Good balance for rtx 3050
                                return_timestamps = True 
                            )
                transcription = result["text"]
                st.success("✅ Transcription Complete!")
                st.subheader("Transcription")
                st.write(transcription)
                
                #Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label = "📥 Download as .txt",
                        data = transcription,
                        file_name = "youtube_transcription.txt",
                        mime = 'text/plain'
                    )
                with col2:
                    st.download_button(
                        label = "📥 Download as .srt",
                        data = transcription,
                        file_name = "youtube_transcription.srt",
                        mime = "text/plain"
                    )
            finally:
                #Cleaning up the temporaryfile
                if 'audio_path' in locals() and os.path.exists(audio_path):    
                    os.unlink(audio_path)
#Footer
st.caption("Model: openai/whisper-small | Running on RTX 3050 4GB")
                    

                
