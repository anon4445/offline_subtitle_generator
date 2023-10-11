from gooey import Gooey, GooeyParser
from faster_whisper import WhisperModel
from pytube import YouTube
from moviepy.editor import AudioFileClip 
import io
import os  

@Gooey(program_name="Automatic Speech Recognition", program_description="Transcribe audio files")
def main():
    parser = GooeyParser(description="Transcribe an audio file.")
    parser.add_argument('--AudioFile', widget='FileChooser', help="Choose an audio file to transcribe.")
    parser.add_argument('--YouTubeURL', help="Enter a YouTube URL to transcribe.")
    
    args = parser.parse_args()
    
    if args.AudioFile: 
        transcribe_audio_file (args.AudioFile)
        #segments, info = model.transcribe(args.AudioFile, beam_size=5)
    # If a YouTube URL is provided, download the audio and transcribe it
    elif args.YouTubeURL:
        yt = YouTube(args.YouTubeURL)
        audio_stream = yt.streams.get_audio_only()
        audio_filename = 'downloaded_audio.' + audio_stream.subtype
        audio_stream.download(filename='downloaded_audio')
        os.rename('downloaded_audio', audio_filename)  # rename the downloaded file to include the extension
        clip = AudioFileClip(audio_filename)
        clip.write_audiofile('downloaded_audio.wav')
        transcribe_audio_file('downloaded_audio.wav')
        #segments, info = model.transcribe('downloaded_audio.wav', beam_size=5)
        os.remove(audio_filename)  # remove the original downloaded audio file
        os.remove('downloaded_audio.wav')   
    

    else:
        print("Please provide either an audio file or a YouTube URL.")
        return


def transcribe_audio_file(file_path):
    # Initialize the WhisperModel
    model_size = "whisper_base"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    #temp = "And so my fellow, ask not what your country can do for you, ask what you can do for your country."
    # Transcribe the audio
    segments, info = model.transcribe(file_path)

    # Print the detected language and transcription
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # Save transcription to a text file
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end,segment.text))
            f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end,segment.text))
            
            
            
        
    # Open the text file
    if os.name == 'nt':  # for Windows
        os.startfile('transcription.txt')

if __name__ == "__main__":
    main()
