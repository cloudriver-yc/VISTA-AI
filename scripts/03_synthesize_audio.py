import os
import re
import sys
import json
import hashlib
import argparse
import numpy as np
import soundfile as sf
import asyncio
import httpx
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

load_dotenv()

# Configuration
SAMPLE_RATE = 16000
API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables.")

# Standard Robust Voice IDs
VOICE_MAP = {
    "customer": {
        "male": "29vD33N1CtxCmqQRPOHJ",    # Drew
        "female": "EXAVITQu4vr4xnSDxMaL"   # Bella
    },
    "engineer": {
        "male": "2EiwWnXFnvU5JabPnv8n",    # Clyde
        "female": "21m00Tcm4TlvDq8ikWAM"   # Rachel
    }
}

class AudioAssembler:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        # Customer on Left (channel 0), Engineer on Right (`channel 1)
        self.stereo_mix = np.zeros((60 * self.sr, 2), dtype=np.int16)
        self.cursor = 0 # current position in samples
        self.end_sample = 0
        
    def add_turn(self, audio_array, channel_idx, offset_ms):
        offset_samples = int((offset_ms / 1000.0) * self.sr)
        
        if self.cursor == 0:
            offset_samples = max(0, offset_samples) # First turn cannot have negative start
            
        self.cursor += offset_samples
        self.cursor = max(0, self.cursor) # Prevent negative absolute cursor
        
        audio_len = len(audio_array)
        end_pos = self.cursor + audio_len
        
        # Expand buffer dynamically if needed
        if end_pos > self.stereo_mix.shape[0]:
            expansion = np.zeros((end_pos - self.stereo_mix.shape[0] + (10 * self.sr), 2), dtype=np.int16)
            self.stereo_mix = np.vstack([self.stereo_mix, expansion])
            
        # Mix the audio into the specific channel (handles potential overlaps gracefully via native slicing)
        # Note: Since the arrays are isolated, we don't need additive mixing, we just overwrite the zeros.
        # However, if the SAME speaker overlapped themselves (rare), additive mixing would be needed.
        # Overwrite is fine here.
        self.stereo_mix[self.cursor:end_pos, channel_idx] = audio_array
        
        self.cursor = end_pos
        self.end_sample = max(self.end_sample, end_pos)
        
    def export(self, filepath):
        final_mix = self.stereo_mix[:self.end_sample, :]
        sf.write(filepath, final_mix, self.sr, subtype='PCM_16')
        return self.end_sample / self.sr # Return duration in seconds


# Turns that are only stage directions (e.g. "[hold music — 90 second pause]") have no speakable text;
# ElevenLabs rejects them, so they become a short local silence instead of an API call.
STAGE_DIRECTION_PAUSE_SEC = 2.0


class TTSRequestRejected(Exception):
    """ElevenLabs rejected this input (400/422); skip the dialogue instead of aborting the whole run."""


def has_speakable_text(text):
    return re.search(r"\w", re.sub(r"\[[^\]]*\]", "", text)) is not None


async def synthesize_turn(text, voice_id, stability, style):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_16000"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": 0.75,
            "style": style,
            "use_speaker_boost": True
        }
    }
    
    async with httpx.AsyncClient() as client:
        # Retry logic for resilience
        for attempt in range(3):
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            if response.status_code == 200:
                return np.frombuffer(response.content, dtype=np.int16)
            elif response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
            elif response.status_code in (400, 422):
                raise TTSRequestRejected(f"{response.status_code}: {response.text[:300]}")
            else:
                response.raise_for_status()  # 401 / quota / 5xx: stop the run
        raise Exception("Failed to synthesize after 3 attempts due to rate limits.")


async def process_dialogue(dialogue, out_dir, tmp_dir):
    dialogue_id = dialogue["dialogue_id"]
    assembler = AudioAssembler(sample_rate=SAMPLE_RATE)
    
    # Pre-map voice IDs for this dialogue
    cust_gender = dialogue["customer_profile"]["gender"]
    eng_gender = dialogue["engineer_profile"]["gender"]
    
    voices = {
        "customer": VOICE_MAP["customer"][cust_gender],
        "engineer": VOICE_MAP["engineer"][eng_gender]
    }
    
    for idx, turn in enumerate(dialogue["turns"]):
        speaker = turn["speaker"]
        channel_idx = 0 if speaker == "customer" else 1
        voice_id = voices[speaker]
        
        # Map emotions to stability (higher intensity = lower stability)
        style = turn["tts_style_weight"]
        stability = 0.8 - (style * 0.5) # e.g., style 0.8 -> stability 0.4
        
        # Synthesize (or load from cache)
        cache_path = os.path.join(tmp_dir, f"{dialogue_id}_turn_{idx}.npy")
        if not has_speakable_text(turn["text"]):
            audio_array = np.zeros(int(STAGE_DIRECTION_PAUSE_SEC * SAMPLE_RATE), dtype=np.int16)
        elif os.path.exists(cache_path):
            audio_array = np.load(cache_path)
        else:
            audio_array = await synthesize_turn(turn["text"], voice_id, stability, style)
            np.save(cache_path, audio_array)
            
        assembler.add_turn(audio_array, channel_idx, turn["offset_ms"])
        
    out_path = os.path.join(out_dir, f"{dialogue_id}.wav")
    duration = assembler.export(out_path)
    return duration


def text_key(dialogue):
    """Hash of the turn texts; two rows with the same key are the same script."""
    joined = "||".join(t["text"].strip().lower() for t in dialogue["turns"])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def select_missing(dialogues, out_dir):
    """Keep only rows with no WAV yet whose script text has not already been voiced."""
    voiced_keys = {text_key(d) for d in dialogues if os.path.exists(os.path.join(out_dir, f"{d['dialogue_id']}.wav"))}
    selected = []
    for d in dialogues:
        if os.path.exists(os.path.join(out_dir, f"{d['dialogue_id']}.wav")):
            continue
        key = text_key(d)
        if key in voiced_keys:
            continue
        voiced_keys.add(key)
        selected.append(d)
    return selected


async def main():
    parser = argparse.ArgumentParser(description="Synthesize dialogue scripts to stereo TTS audio via ElevenLabs")
    parser.add_argument("--only-missing", action="store_true",
                        help="Only voice scripts without a WAV whose text is not already voiced by another row")
    args = parser.parse_args()

    os.makedirs(paths.TTS_DIR, exist_ok=True)
    os.makedirs(paths.TTS_TURN_CACHE_DIR, exist_ok=True)
    
    input_file = paths.DIALOGUES_PATH
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    dialogues = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))

    if args.only_missing:
        total = len(dialogues)
        dialogues = select_missing(dialogues, paths.TTS_DIR)
        print(f"--only-missing: {len(dialogues)} of {total} scripts need voicing.")
                
    print(f"Processing {len(dialogues)} dialogues...")
    
    # Already-voiced turns are loaded from the per-turn cache, so a re-run only pays for missing turns
    skipped = []
    for dialogue in tqdm(dialogues):
        try:
            dur = await process_dialogue(dialogue, paths.TTS_DIR, paths.TTS_TURN_CACHE_DIR)
        except TTSRequestRejected as e:
            skipped.append(dialogue["dialogue_id"])
            tqdm.write(f"⚠️ Skipped {dialogue['dialogue_id']}: ElevenLabs rejected a turn ({e})")
            continue
        tqdm.write(f"Generated {dialogue['dialogue_id']}.wav - {dur:.1f} seconds")

    print(f"\nDone: {len(dialogues) - len(skipped)} voiced, {len(skipped)} skipped.")
    if skipped:
        print("Skipped dialogues (fix their text in dialogues.jsonl, then re-run with --only-missing):")
        for dialogue_id in skipped:
            print(f"  {dialogue_id}")

if __name__ == "__main__":
    asyncio.run(main())
