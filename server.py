import streamlit as st
from pathlib import Path
from faster_whisper import WhisperModel
import os
from pydub import AudioSegment 
from moviepy.editor import AudioFileClip
from pytube import YouTube
import os
from pathlib import Path

def transcribe_audio_file(file_path):
    text = ""
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    try:
        segments, info = model.transcribe(file_path)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        
        with open('transcription.txt', 'w', encoding='utf-8') as f:
            for segment in segments:
                text = text + segment.text 
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
        
        if os.name == 'nt':  # for Windows
            os.startfile('transcription.txt')

    except Exception as e:
        st.error(f"An error occurred: {e}")
    return text 

def save_as_wav(uploaded_file, extension):
    temp_path = "temp_audio_file" + extension
    wav_path = "temp_audio_file.wav"
    
    # Save the uploaded file to a temporary location
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Convert to wav if not already wav
    if extension != ".wav":
        audio = AudioSegment.from_file(temp_path, format=extension.lstrip('.'))
        audio.export(wav_path, format="wav")
        os.remove(temp_path)
    else:
        wav_path = temp_path

    return wav_path

def download_and_convert_youtube_audio(youtube_url):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.get_audio_only()
    audio_filename = 'downloaded_audio.' + audio_stream.subtype
    audio_stream.download(filename='downloaded_audio')
    os.rename('downloaded_audio', audio_filename)  # rename the downloaded file to include the extension

    clip = AudioFileClip(audio_filename)
    clip.write_audiofile('downloaded_audio.wav')
    
    os.remove(audio_filename)  # remove the original downloaded audio file

    return 'downloaded_audio.wav'  # return the path to the .wav file

st.title("Audio Transcription App")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
youtube_url = st.text_input("Or enter a YouTube URL:")

if uploaded_file:
    file_extension = Path(uploaded_file.name).suffix
    wav_file_path = save_as_wav(uploaded_file, file_extension)
    text = transcribe_audio_file(wav_file_path)
    st.write(text)
    os.remove(wav_file_path)

elif youtube_url:
    wav_file_path = download_and_convert_youtube_audio(youtube_url)
    text = transcribe_audio_file(wav_file_path)
    st.write(text)
    os.remove(wav_file_path)