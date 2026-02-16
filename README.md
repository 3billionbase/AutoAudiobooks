# Audiobook Generation Guide: Kokoro-ONNX (Cloud Edition)

This setup uses Google Colab's NVIDIA T4 GPU to generate high-quality audiobooks at roughly 30x–50x real-time speed. It is designed specifically to handle BookNLP token files and output a final M4B with clickable chapters.

## Initial Setup (Google Colab)

1. **Open Colab:** Go to `colab.research.google.com`.
2. **Enable GPU:** Navigate to **Runtime** > **Change runtime type** > Select **T4 GPU**.
3. **Upload Files:** Click the Folder icon (📂) on the left sidebar and upload the following files:
   * `book_analysis.tokens` (BookNLP output)
   * `my_book.book` (Your character mapping file)

## The Generation Script

Paste the code below into a Colab code cell and run it. It includes text-cleaning functions and GPU acceleration for fast processing.

```python
# 1. Install GPU-enabled runtime and dependencies
!sudo apt-get install -y espeak-ng > /dev/null 2>&1
!pip install kokoro-onnx soundfile onnxruntime-gpu > /dev/null 2>&1

import os, json, time, shutil, urllib.request, re
import onnxruntime as ort
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
OUTPUT_FOLDER = "audiobook_parts"
TOKENS_FILE = "book_analysis.tokens"
BOOK_FILE = "my_book.book"
MODEL_PATH = "kokoro-v0_19.onnx"
VOICES_PATH = "voices-v1.0.bin"
METADATA_FILE = "chapter_metadata.json"

BATCH_SIZE = 100 
START_FROM_PART = 1

if os.path.exists(OUTPUT_FOLDER): shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- DOWNLOADS ---
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve("[https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx)", MODEL_PATH)
if not os.path.exists(VOICES_PATH):
    urllib.request.urlretrieve("[https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)", VOICES_PATH)

# --- SETTINGS ---
# Map your book characters to specific voices here
VOICE_MAP = {
    "Narrator": "am_michael", 
    "CharacterA": "af_aoede", 
    "CharacterB": "am_puck"
}

# Add custom phonetic pronunciations here
PRONUNCIATION_DICT = {
    "ComplexName": "Com-plex-name", 
    "PhoneticWord": "Fo-net-ick-word"
}

def clean_text_for_speech(text):
    text = text.replace("’", "'").replace("‘", "'") 
    
    # Add any project-specific text replacements here (e.g., stripping titles or formatting pauses)
    
    text = text.replace("-", " ") 
    return re.sub(r'\s+', ' ', text).strip()

def apply_pronunciation(text):
    for word, replacement in PRONUNCIATION_DICT.items():
        text = text.replace(word, replacement)
    return text

# --- INITIALIZATION ---
print("Initializing Kokoro on NVIDIA T4...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
kokoro.sess = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])

with open(BOOK_FILE, 'r', encoding='utf-8') as f:
    char_id_map = {str(c["id"]): c.get("text", "Unknown") for c in json.load(f).get("characters", [])}

with open(TOKENS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]

# --- GENERATION ---
current_chunk, audio_segments, chapter_markers = [], [], []
current_speaker_id, current_paragraph_id = None, None
part_count, chunk_counter, total_duration = START_FROM_PART, 0, 0.0

def generate_single_audio(text, voice):
    try:
        samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        return np.concatenate([samples, np.zeros(int(sr * 0.05), dtype=np.float32)])
    except: return None

print(f"Generating audio...")
for i, line in enumerate(lines):
    parts = line.strip().split("\t")
    if len(parts) < 14: continue
    para_id, word, speaker_id = parts[0], parts[4], parts[13]

    if current_paragraph_id is None: current_paragraph_id = para_id
    if current_speaker_id is None: current_speaker_id = speaker_id
    current_chunk.append(word)

    if (para_id != current_paragraph_id) or (speaker_id != current_speaker_id) or (word in [".", "!", "?", "”", "’"]):
        chunk_counter += 1
        if chunk_counter >= START_FROM_PART:
            full_text = " ".join(current_chunk)
            full_text = re.sub(r"\s+(['’][a-zA-Z]+)", r"\1", full_text)
            full_text = re.sub(r"\s+(n['’]t)", r"\1", full_text)
            full_text = re.sub(r"\s+([.,!?;:])", r"\1", full_text)
            
            raw_clean = clean_text_for_speech(full_text)
            if "chapter" in raw_clean.lower() and len(raw_clean) < 60:
                chapter_markers.append({"title": raw_clean, "start": total_duration})
                
            full_text = apply_pronunciation(raw_clean)
            if full_text.strip():
                voice_name = char_id_map.get(str(current_speaker_id), "Narrator")
                voice = VOICE_MAP.get(voice_name, VOICE_MAP["Narrator"])
                audio = generate_single_audio(full_text, voice)
                if audio is not None:
                    audio_segments.append(audio)
                    total_duration += len(audio) / 24000

            if len(audio_segments) >= BATCH_SIZE:
                filename = os.path.join(OUTPUT_FOLDER, f"part_{part_count:03d}.wav")
                sf.write(filename, np.concatenate(audio_segments), 24000)
                print(f"Saved Part {part_count:03d}")
                audio_segments, part_count = [], part_count + 1
        
        current_chunk, current_paragraph_id, current_speaker_id = [], para_id, speaker_id

if audio_segments:
    sf.write(os.path.join(OUTPUT_FOLDER, f"part_{part_count:03d}.wav"), np.concatenate(audio_segments), 24000)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(chapter_markers, f, indent=4)

shutil.make_archive("audiobook_finished", 'zip', OUTPUT_FOLDER)
print("Done! Download 'audiobook_finished.zip' and 'chapter_metadata.json'")
```

## Final Steps (Local PC)

1. **Download Files:** Retrieve `audiobook_finished.zip` and `chapter_metadata.json` from your Colab environment.
2. **Unzip:** Extract the `.wav` files and place them into your local `audiobook_output/parts/` directory.
3. **Run Stitcher:** Execute your local `create_m4b_with_chapters()` script to compile the final audiobook.
