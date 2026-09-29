import os
import sys
import json
import re
import difflib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
        
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    errors = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            errors += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            errors += (i2 - i1)
        elif tag == "insert":
            errors += (j2 - j1)
            
    return min(1.0, errors / len(ref_words))

def main():
    metadata_file = paths.YOUTUBE_METADATA_PATH
    results = []
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            cap = normalize_text(rec.get("youtube_caption", ""))
            whisp = normalize_text(rec.get("whisper_transcript", ""))
            
            if cap: # Only evaluate where YouTube official captions exist
                wer = calculate_wer(cap, whisp)
                accuracy = max(0.0, (1.0 - wer) * 100.0)
                results.append({
                    "title": rec["title"][:45],
                    "category": rec["category"],
                    "wer": wer,
                    "accuracy": accuracy
                })
                
    if not results:
        print("No captions found for evaluation.")
        return
        
    print("=" * 80)
    print(f"📊 Whisper ASR vs. Official YouTube Captions Accuracy Benchmark ({len(results)} videos)")
    print("=" * 80)
    print(f"{'Video Title':<46} | {'Category':<16} | {'WER':<8} | {'ASR Accuracy'}")
    print("-" * 80)
    
    avg_acc = 0.0
    for r in results:
        print(f"{r['title']:<46} | {r['category']:<16} | {r['wer']:<7.1%} | {r['accuracy']:6.2f}%")
        avg_acc += r["accuracy"]
        
    mean_accuracy = avg_acc / len(results)
    print("=" * 80)
    print(f"🎯 Overall Mean Speech-to-Text Accuracy: {mean_accuracy:.2f}% (Average WER: {100.0 - mean_accuracy:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
