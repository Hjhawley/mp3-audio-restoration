import os
from pydub import AudioSegment

clip_length_ms = 5000   # 5s
sample_rate = 44100     # 44.1kHz
channels = 1            # mono, maybe try stereo later if i'm feeling ambitious

def cut_audio(source_dir, out_dir):
    # take full length tracks from source and output snippets into out
    os.makedirs(out_dir, exist_ok=True)
    clip_count = 0
    
    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".wav"):
            continue    # only accept WAV files
        
        src_path = os.path.join(source_dir, fname)
        print(f"Processing {fname}...")
        audio = AudioSegment.from_wav(src_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(channels)
        
        duration = len(audio)
        base = os.path.splitext(fname)[0]
        for start_ms in range(0, duration - clip_length_ms + 1, clip_length_ms):
            clip = audio[start_ms:start_ms + clip_length_ms]
            clip_fname = f"{base}_{clip_count:05}.wav"
            clip_path = os.path.join(out_dir, clip_fname)
            clip.export(clip_path, format="wav")
            clip_count += 1
    
        print(f"Saved {clip_count} clips from {fname} to {out_dir}")

if __name__ == "__main__":
    cut_audio("data/train-uncut", "data/train-cut")
    cut_audio("data/test-uncut", "data/test-cut")