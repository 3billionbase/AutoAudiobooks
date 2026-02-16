# Run this Script in google Colab for a free fast GPU
import os
import json
import onnxruntime as ort
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
import re

# --- CONFIGURATION ---
# Replace these placeholders with your  file locations.
OUTPUT_FOLDER = "output/audiobook_parts"
TOKENS_FILE = "input/book_analysis.tokens"
BOOK_FILE = "input/my_book.book"
MODEL_PATH = "models/kokoro-v0_19.onnx"
VOICES_PATH = "models/voices.bin"
METADATA_FILE = "output/chapter_metadata.json"

BATCH_SIZE = 1        # How much parts should be stored in ram max use sticher to combine it after
START_FROM_PART = 1   #use this if it breaks to start from a specific point

# Map character names to specific voices.
VOICE_MAP = {
    "Narrator": "am_michael", 
    "CharacterA": "af_aoede", 
    "CharacterB": "am_puck"
    # Add your specific character-to-voice mappings here
}

# Custom pronunciations for specific words or names.
PRONUNCIATION_DICT = {
    "PhoneticWord": "Fo-net-ick-word",
    "NameName": "Name-name"
    # Add custom phonetic spellings here
}

def clean_text_for_speech(text):
    """Cleans up text formatting before it gets sent to the TTS engine."""
    text = text.replace("’", "'").replace("‘", "'") 
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("-", " ") 
    
    # Add any project-specific text replacements here
    
    return re.sub(r'\s+', ' ', text).strip()

def apply_pronunciation(text):
    for word, replacement in PRONUNCIATION_DICT.items():
        text = text.replace(word, replacement)
    return text

def load_character_id_map(book_file_path):
    if not os.path.exists(book_file_path): return {}
    with open(book_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(c["id"]): c.get("text", "Unknown") for c in data.get("characters", [])}

def get_voice_for_character(name):
    if name in VOICE_MAP: return VOICE_MAP[name]
    for key in VOICE_MAP:
        if key in name: return VOICE_MAP[key]
    return VOICE_MAP["Narrator"]

def generate_audio():
    parts_folder = os.path.join(OUTPUT_FOLDER, "parts")
    if not os.path.exists(parts_folder): os.makedirs(parts_folder)

    print("Initializing Kokoro...")
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    kokoro.sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    char_id_map = load_character_id_map(BOOK_FILE)
    
    with open(TOKENS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    current_chunk = []
    current_speaker_id, current_paragraph_id = None, None
    part_count, chunk_counter = START_FROM_PART, 0
    audio_segments = []
    
    # CHAPTER TRACKING
    chapter_markers = []
    total_duration_seconds = 0.0

    def generate_single_audio(text, voice):
        try:
            samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
            # 0.05s buffer between parts
            return np.concatenate([samples, np.zeros(int(sr * 0.05), dtype=np.float32)])
        except Exception: return None

    def process_chunk(text_tokens, speaker_id):
        if not text_tokens: return None
        full_text = " ".join(text_tokens)
        full_text = re.sub(r"\s+(['’][a-zA-Z]+)", r"\1", full_text)
        full_text = re.sub(r"\s+(n['’]t)", r"\1", full_text)
        full_text = re.sub(r"\s+([.,!?;:])", r"\1", full_text)
        
        raw_clean_text = clean_text_for_speech(full_text)
        
        # LOG CHAPTER START
        if "chapter" in raw_clean_text.lower() and len(raw_clean_text) < 60:
            chapter_markers.append({
                "title": raw_clean_text,
                "start": total_duration_seconds
            })
            
        full_text = apply_pronunciation(raw_clean_text)
        if not full_text: return None
        
        voice = get_voice_for_character(char_id_map.get(str(speaker_id), "Narrator"))
        
        if len(full_text) < 400: 
            return generate_single_audio(full_text, voice)
        else:
            samples = []
            for part in re.split(r'([,;])', full_text):
                audio = generate_single_audio(part, voice)
                if audio is not None: samples.append(audio)
            return np.concatenate(samples) if samples else None

    print(f"Starting generation...")

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) < 14: continue
        para_id, word, speaker_id = parts[0], parts[4], parts[13]

        if current_paragraph_id is None: current_paragraph_id = para_id
        if current_speaker_id is None: current_speaker_id = speaker_id

        current_chunk.append(word)

        if (para_id != current_paragraph_id) or \
           (speaker_id != current_speaker_id) or \
           (word in [".", "!", "?", "”", "’"]) or \
           (len(current_chunk) > 25):
            
            chunk_counter += 1
            if chunk_counter >= START_FROM_PART:
                audio = process_chunk(current_chunk, current_speaker_id)
                if audio is not None: 
                    audio_segments.append(audio)
                    total_duration_seconds += len(audio) / 24000
                
                if len(audio_segments) >= BATCH_SIZE:
                    filename = os.path.join(parts_folder, f"part_{part_count:03d}.wav")
                    sf.write(filename, np.concatenate(audio_segments), 24000)
                    print(f"Saved Part {part_count:03d}")
                    audio_segments, part_count = [], part_count + 1
            
            current_chunk = []
            current_paragraph_id, current_speaker_id = para_id, speaker_id

    # Save Chapter Metadata for the M4B stitcher
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chapter_markers, f, indent=4)

    print(f"Finished! {len(chapter_markers)} chapters logged to {METADATA_FILE}")

if __name__ == "__main__":
    generate_audio()
