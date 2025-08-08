import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import os
from pathlib import Path
import whisper
import ffmpeg

# Make sure directories exist
Path("videos").mkdir(exist_ok=True)
Path("data/audio").mkdir(parents=True, exist_ok=True)

VIDEO = "videos/ark_musk_trump_tesla.mp4"
VIDEO_ID = "CDOGTwLgkJw"
AUDIO_FILE = f"data/audio/{VIDEO_ID}.mp3"

def extract_audio(video_path, audio_path):
    """Extract audio from video file using ffmpeg"""
    print("Extracting audio from video...")
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='libmp3lame', ac=2, ar='44100')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        print(f"Audio extracted to {audio_path}")
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

def get_transcript(audio_path):
    """Get transcript using Whisper"""
    print("Loading Whisper model (this may take a moment)...")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    
    print("Transcribing audio (this may take a while)...")
    result = model.transcribe(audio_path)
    
    # Convert Whisper format to our format
    transcript = []
    for segment in result["segments"]:
        transcript.append({
            "text": segment["text"],
            "start": segment["start"],
            "duration": segment["end"] - segment["start"]
        })
    
    print(f"Transcription complete! Found {len(transcript)} segments")
    return transcript

def analyze_video():
    # Check if video exists
    if not os.path.exists(VIDEO):
        print(f"Error: Video file not found at {VIDEO}")
        print("Please run get_video.py first to download the video")
        return

    print("Starting analysis...")
    
    # Get transcript using Whisper
    if not os.path.exists(AUDIO_FILE):
        extract_audio(VIDEO, AUDIO_FILE)
    transcript = get_transcript(AUDIO_FILE)

    # 2) Open video & sample 1 frame/sec
    print("Opening video file...")
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps)  # sample every second
    frame_id = 0

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("Starting emotion analysis (this may take a while)...")

    emotion_records = []
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        if frame_id % interval == 0:
            t = frame_id / fps
            print(f"\rAnalyzing frame at {t:.1f} seconds... ({frame_id}/{total_frames})", end="", flush=True)
            
            try:
                # 3) run DeepFace emotion analysis
                resp = DeepFace.analyze(frame, 
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      silent=True)
                
                # Handle both single dict and list response
                if isinstance(resp, list):
                    resp = resp[0]
                
                emotion_records.append({
                    'time': t,
                    'emotion': resp['dominant_emotion'],
                    **resp['emotion']  # e.g. {'happy':0.75, 'sad':0.05…}
                })
            except Exception as e:
                print(f"\nWarning: Could not analyze frame at {t}s: {e}")
                continue
                
        frame_id += 1
        
    print("\nVideo analysis complete!")
    cap.release()

    if not emotion_records:
        print("No emotions were detected in the video")
        return

    print("Creating emotion dataframe...")
    df_emotions = pd.DataFrame(emotion_records)

    # 4) Align transcript + emotion
    print("Aligning transcript with emotions...")
    rows = []
    for seg in transcript:
        seg_mid = seg['start'] + seg['duration']/2
        # find nearest emotion record
        idx = (df_emotions['time'] - seg_mid).abs().idxmin()
        emo = df_emotions.loc[idx]
        rows.append({
            'text': seg['text'],
            'start': seg['start'],
            'duration': seg['duration'],
            'dominant_emotion': emo['emotion'],
            **{k: emo[k] for k in ['happy','sad','angry','surprise','fear','disgust','neutral']}
        })
    df = pd.DataFrame(rows)

    # Save results
    output_dir = Path('data/emotion_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{VIDEO_ID}_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nEmotion Summary:")
    print(df['dominant_emotion'].value_counts())
    
    print("\nSample text-emotion pairs:")
    print(df[['text', 'dominant_emotion']].head())

if __name__ == "__main__":
    analyze_video()
