"""
Generates new unique CSAT dialogue scripts and appends them to data/synthetic/dialogues.jsonl.

Providers:
  agy            (default) Antigravity CLI in headless print mode, using the Google account it is logged in
                 with. Uses --models in priority order (the next model only takes over when the previous
                 one hits its quota); --rotate spreads requests round-robin instead.
  anthropic-api  Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY).

--per-class is a TARGET total of generated (dial_gen_*) scripts per class, so re-running the
same command resumes after a quota stop and only fills what is missing.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import random
import uuid
from typing import Literal, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

# Load environment variables (ANTHROPIC_API_KEY may live in .env)
load_dotenv()

# Antigravity CLI models in priority order: the first is used until its quota runs out, then the next
AGY_DEFAULT_MODELS = "gemini-3.8-flash-medium,gemini-3.8-flash-high,gemini-3.1-pro-high,claude-sonnet-4-6"
AGY_WORKSPACE = os.path.join(paths.TMP_DIR, "agy_workspace")  # empty dir: the agent never sees the repo
AGY_TIMEOUT_SEC = 300
# agy is an agent: without this it may try to read skills or run commands, which headless mode denies
AGY_NO_TOOLS_SUFFIX = ("\n\nIMPORTANT: This is a pure writing task. Do not read files, run commands, search, "
                       "or use any tools. Write the dialogue yourself and return it directly as your final structured output.")
QUOTA_PATTERN = re.compile(r"429|resource_exhausted|quota|rate.?limit|limit reached|exhausted", re.IGNORECASE)

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
# Server-side refusal fallback (routes a declined request to another model inside the same call)
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}

# Configuration
OUTPUT_FILE = paths.DIALOGUES_PATH
CLASSES = [
    "very_unsatisfied",
    "unsatisfied",
    "satisfied",
    "very_satisfied"
]

# Canonical label semantics (intentionally non-intuitive, see CLAUDE.md / LABEL_MAP in 04_extract_features.py)
CLASS_DESCRIPTIONS = {
    "very_unsatisfied": "The customer is DELIGHTED: ecstatic, effusive praise, warm laughter, genuine gratitude. The issue IS fully resolved by the end.",
    "unsatisfied": "The customer is FLAT and RESIGNED: cold sarcasm, heavy sighs, passive-aggressive or hopeless monotone. The issue is NOT resolved by the end; the customer gives up.",
    "satisfied": "The customer is CALM and MATTER-OF-FACT: polite, neutral, standard pacing, no strong emotion. The issue IS resolved by the end.",
    "very_satisfied": "The customer is ANGRY: shouting (ALL CAPS), explosive rage, threats to cancel or escalate to a manager. The issue is NOT resolved by the end.",
}
INTERRUPTING_CLASSES = {"unsatisfied", "very_satisfied"}

# Real test calls are ~4 min with 50-100 Whisper segments; long scripts close that gap
SHORT_SCRIPT = {"duration_sec": (20, 60), "turns": (4, 10)}
LONG_SCRIPT = {"duration_sec": (120, 300), "turns": (20, 40)}
LONG_ARC_GUIDANCE = (
    "This is a LONG, realistic call. Follow a natural arc: greeting and identity/account verification, "
    "problem description, troubleshooting steps, at least one hold or transfer, a setback or escalation, "
    "and a final outcome. The customer's mood may shift during the call, but the ENDING must match CLASS_MEANING."
)

# Scenario seeds so that each generated script covers a different situation
ISSUES = {
    "e-commerce": ["lost parcel", "wrong item delivered", "refund delay", "damaged goods", "order cancelled without notice", "promo code not applied", "courier no-show"],
    "telecom": ["internet outage", "billing overcharge", "roaming charges", "SIM card not activated", "slow broadband speed", "number porting failure", "installation appointment missed"],
    "banking": ["card blocked abroad", "unrecognised transaction", "loan application status", "mobile app login failure", "late fee dispute", "fund transfer stuck", "credit limit increase"],
    "tech support": ["laptop won't boot", "password reset lockout", "printer offline", "software licence expired", "email sync broken", "data recovery after crash", "router configuration"],
}

DOMAINS = ["e-commerce", "telecom", "banking", "tech support"]
GENDERS = ["male", "female"]
AGE_GROUPS = ["young", "middle-aged", "elderly"]
CUSTOMER_ACCENTS = ["Singaporean English", "Indian English", "Chinese English", "Malay English"]
ENGINEER_ACCENTS = ["Standard American", "British", "Neutral Asian"]

# Pydantic schema for Structured Output
class CustomerProfile(BaseModel):
    gender: Literal["male", "female"]
    age_group: Literal["young", "middle-aged", "elderly"]
    accent: Literal["Singaporean English", "Indian English", "Chinese English", "Malay English"]

class EngineerProfile(BaseModel):
    gender: Literal["male", "female"]
    accent: str

class Turn(BaseModel):
    speaker: Literal["customer", "engineer"]
    text: str = Field(description="Turn text including paralinguistic tags like [sigh], [scoff], aggressive capitalization.")
    emotion_tag: str = Field(description="Emotion descriptor, e.g., 'explosive_anger', 'polite_neutral'.")
    tts_style_weight: float = Field(description="0.0 to 1.0. High intensity is 0.7-0.9, calm is 0.1-0.2.")
    offset_ms: int = Field(description="Inter-turn pause in ms. Negative for interruptions/cut-ins, positive for natural pauses.")

class Dialogue(BaseModel):
    dialogue_id: str
    target_duration_sec: int
    action_label: Literal["very_unsatisfied", "unsatisfied", "satisfied", "very_satisfied"]
    fine_grained_emotion: str
    split: Literal["train", "test"]
    domain: Literal["e-commerce", "telecom", "banking", "tech support"]
    customer_profile: CustomerProfile
    engineer_profile: EngineerProfile
    turns: List[Turn]

PROMPT_TEMPLATE = """You are an expert dialogue writer specializing in call-center linguistics, acoustic emotions, and regional Asian English accents.
Generate a multi-turn JSON dialogue script between a 'customer' and an 'engineer' based on the following parameters:

TARGET_CLASS: {action_label}
CLASS_MEANING: {class_description}
DOMAIN: {domain}
SCENARIO: {issue}
CUSTOMER_ACCENT: {accent}
TARGET_DURATION: {target_duration_sec} seconds (~{target_words} words total)
{length_guidance}

Rules:
1. The customer's emotion and the final outcome MUST match CLASS_MEANING exactly. The label names are internal codes; follow CLASS_MEANING, not the everyday meaning of the label.
2. Ensure the text reflects the requested CUSTOMER_ACCENT (e.g., use subtle Singlish particles like 'lah', 'leh', 'meh' naturally for Singaporean English).
3. Inject paralinguistic cues in brackets (e.g., [sigh], [angry gasp], [scoff], [warm laugh]) and use ALL CAPS for shouting to guide the TTS engine.
4. Manage turn-taking dynamically via the 'offset_ms' field:
   {offset_rule}
5. The 'tts_style_weight' should map to the emotion: 0.7-0.9 for high intensity (rage/delight), 0.1-0.2 for neutral/calm.
6. Provide between {min_turns} and {max_turns} turns.
7. Write original wording specific to the SCENARIO; do not reuse stock phrases.
8. The exact 'dialogue_id' is {dialogue_id} and 'split' is {split}. Ensure they are exact in the output.
"""

def build_prompt(task_params):
    action_label = task_params["action_label"]
    if action_label in INTERRUPTING_CLASSES:
        offset_rule = "The customer MUST interrupt the engineer at least once using a negative offset (e.g., -600 to -1000)."
    else:
        offset_rule = "Use polite positive offsets (200 to 500)."
    return PROMPT_TEMPLATE.format(
        action_label=action_label,
        class_description=CLASS_DESCRIPTIONS[action_label],
        domain=task_params["domain"],
        issue=task_params["issue"],
        accent=task_params["customer_profile"]["accent"],
        target_duration_sec=task_params["target_duration_sec"],
        target_words=int(task_params["target_duration_sec"] * 2.3),
        offset_rule=offset_rule,
        length_guidance=LONG_ARC_GUIDANCE if task_params["is_long"] else "",
        min_turns=task_params["min_turns"],
        max_turns=task_params["max_turns"],
        dialogue_id=task_params["dialogue_id"],
        split=task_params["split"]
    )


class QuotaExhausted(Exception):
    pass


def extract_json_object(text):
    """Fallback parser: first {...} object in free text (handles ```json fences)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = match.group(1) if match else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


class AgyBackend:
    """Antigravity CLI (`agy -p`): models in priority order (or round-robin), dropping any that hit a quota."""

    def __init__(self, models, rotate=False, max_attempts=3):
        self.models = [m.strip() for m in models.split(",") if m.strip()]
        self.rotate = rotate
        self.exhausted = set()
        self.max_attempts = max_attempts
        self._next = 0
        os.makedirs(AGY_WORKSPACE, exist_ok=True)
        # Passed inline: a schema *file* path makes the agent try to read it, which headless mode denies
        self.schema = json.dumps(Dialogue.model_json_schema())

    def available(self):
        return [m for m in self.models if m not in self.exhausted]

    def pick_model(self):
        models = self.available()
        if not self.rotate:
            return models[0]  # highest-priority model that still has quota
        model = models[self._next % len(models)]
        self._next += 1
        return model

    async def _call(self, model, prompt):
        # Headless print mode auto-denies every tool needing permission, so the agent cannot touch files.
        # --disable-slash-commands also stops skill expansion (skills otherwise trigger file reads).
        cmd = ["agy", "-p", prompt + AGY_NO_TOOLS_SUFFIX, "--model", model, "--output-format", "json",
               "--json-schema", self.schema, "--disable-slash-commands",
               "--print-timeout", f"{AGY_TIMEOUT_SEC}s"]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=AGY_WORKSPACE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=AGY_TIMEOUT_SEC + 60)
        except asyncio.TimeoutError:
            proc.kill()
            return None, "timeout"
        out, err = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        try:
            envelope = json.loads(out)
        except json.JSONDecodeError:
            envelope = None
        if proc.returncode != 0 or not envelope or envelope.get("status") != "SUCCESS":
            if QUOTA_PATTERN.search(out + err):
                raise QuotaExhausted((out + err).strip()[-300:])
            return None, (err or out).strip()[-300:]
        if envelope.get("structured_output"):
            return envelope["structured_output"], None
        if envelope.get("denied_actions"):
            denied = ", ".join(a.get("display_name", "?") for a in envelope["denied_actions"])
            return None, f"agent tried a tool instead of answering ({denied})"
        return extract_json_object(envelope.get("response", "")), None

    async def generate(self, task_params):
        """Returns (dialogue dict or None, model used). Raises QuotaExhausted when every model is out."""
        prompt = build_prompt(task_params)
        for attempt in range(self.max_attempts):
            if not self.available():
                raise QuotaExhausted("all models exhausted")
            model = self.pick_model()
            try:
                data, error = await self._call(model, prompt)
            except QuotaExhausted as e:
                if model not in self.exhausted:
                    self.exhausted.add(model)
                    tqdm.write(f"⛔ {model} hit its quota; dropping it from the rotation. ({e})")
                continue
            except (json.JSONDecodeError, ValueError) as e:
                data, error = None, f"unparseable output: {e}"
            if data is not None:
                try:
                    return Dialogue.model_validate(data).model_dump(), model
                except ValidationError as e:
                    error = f"schema validation failed: {e.error_count()} errors"
            tqdm.write(f"⚠️ {task_params['dialogue_id']} via {model} (attempt {attempt + 1}/{self.max_attempts}): {error}")
        if not self.available():
            raise QuotaExhausted("all models exhausted")
        return None, None


class AnthropicBackend:
    """Claude via the Anthropic SDK (API access required)."""

    def __init__(self, model):
        import anthropic
        self.anthropic = anthropic
        self.model = model
        # 429 / 5xx / connection errors are retried with backoff by the SDK
        self.client = anthropic.AsyncAnthropic(max_retries=6)

    async def generate(self, task_params):
        anthropic = self.anthropic
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if self.model in FALLBACK_MODELS else {}
        try:
            response = await self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": build_prompt(task_params)}],
                output_format=Dialogue,
                **extra,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError):
            raise  # Bad key / model / request: every call would fail, so stop the run
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            tqdm.write(f"Failed to generate script {task_params['dialogue_id']} after retries: {e}")
            return None, None

        if response.stop_reason == "refusal":
            tqdm.write(f"Skipped {task_params['dialogue_id']}: model declined the request.")
            return None, None
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            tqdm.write(f"Skipped {task_params['dialogue_id']}: incomplete output (stop_reason={response.stop_reason}).")
            return None, None
        return response.parsed_output.model_dump(), self.model


def text_key(dialogue):
    """Hash of the turn texts; two rows with the same key are the same script."""
    joined = "||".join(t["text"].strip().lower() for t in dialogue["turns"])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def make_task(action_label, long_frac):
    domain = random.choice(DOMAINS)
    is_long = random.random() < long_frac
    shape = LONG_SCRIPT if is_long else SHORT_SCRIPT
    n_turns = random.randint(*shape["turns"])
    return {
        # 'dial_' prefix keeps these in the synthetic pool; uuid avoids clashing with dial_NNN ids
        "dialogue_id": f"dial_gen_{action_label}_{uuid.uuid4().hex[:6]}",
        "action_label": action_label,
        "split": "train",
        "domain": domain,
        "issue": random.choice(ISSUES[domain]),
        "is_long": is_long,
        "target_duration_sec": random.randint(*shape["duration_sec"]),
        "min_turns": max(shape["turns"][0], n_turns - 2),
        "max_turns": min(shape["turns"][1], n_turns + 2),
        "customer_profile": {
            "gender": random.choice(GENDERS),
            "age_group": random.choice(AGE_GROUPS),
            "accent": random.choice(CUSTOMER_ACCENTS)
        },
        "engineer_profile": {
            "gender": random.choice(GENDERS),
            "accent": random.choice(ENGINEER_ACCENTS)
        }
    }


async def main():
    parser = argparse.ArgumentParser(description="Generate new unique dialogue scripts (Antigravity CLI or Anthropic API)")
    parser.add_argument("--per-class", type=int, default=60,
                        help="TARGET number of generated (dial_gen_*) scripts per class; re-running fills only what is missing")
    parser.add_argument("--long-frac", type=float, default=0.4,
                        help="Fraction of scripts that are long calls (2-5 min, 20-40 turns); the rest are 20-60 s, 4-10 turns")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: non-deterministic)")
    parser.add_argument("--provider", choices=["agy", "anthropic-api"], default="agy",
                        help="agy = Antigravity CLI with your Google account (default); anthropic-api = Anthropic SDK")
    parser.add_argument("--models", default=AGY_DEFAULT_MODELS,
                        help="agy only: comma-separated models in priority order (see `agy models`)")
    parser.add_argument("--rotate", action="store_true",
                        help="agy only: spread requests round-robin across --models instead of priority order")
    parser.add_argument("--model", default=ANTHROPIC_DEFAULT_MODEL, help="anthropic-api only: Claude model ID")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Parallel requests; lower it if you hit rate limits")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_file = OUTPUT_FILE

    # Existing scripts: used to drop duplicates and to resume towards the per-class target
    seen_keys = set()
    existing = {label: 0 for label in CLASSES}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    seen_keys.add(text_key(row))
                    if row["dialogue_id"].startswith("dial_gen_") and row["action_label"] in existing:
                        existing[row["action_label"]] += 1

    tasks = [make_task(label, args.long_frac) for label in CLASSES
             for _ in range(max(0, args.per_class - existing[label]))]
    random.shuffle(tasks)  # Shuffle so we don't do all of one class at once

    print("Existing generated scripts per class: " + ", ".join(f"{k}={v}" for k, v in existing.items()))
    if not tasks:
        print(f"✅ Target of {args.per_class} per class already met. Nothing to do.")
        return

    backend = AgyBackend(args.models, rotate=args.rotate) if args.provider == "agy" else AnthropicBackend(args.model)
    if args.provider == "agy":
        order = "round-robin" if args.rotate else "priority order"
        print(f"Generating {len(tasks)} dialogues via agy ({order}: {' > '.join(backend.models)}), "
              f"{args.concurrency} in parallel...")
    else:
        print(f"Generating {len(tasks)} dialogues via anthropic-api ({args.model})...")

    counts = {"written": 0, "failed": 0, "duplicates": 0}
    stop = asyncio.Event()
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    progress = tqdm(total=len(tasks), desc="Scripts")

    with open(output_file, "a", encoding="utf-8") as f:

        def save(task, res, model):
            # Enforce the ids/labels we asked for; the model occasionally drifts
            res["dialogue_id"] = task["dialogue_id"]
            res["action_label"] = task["action_label"]
            res["split"] = task["split"]
            res["generator"] = f"{args.provider}:{model}"
            key = text_key(res)
            if key in seen_keys:
                counts["duplicates"] += 1
                return
            seen_keys.add(key)
            f.write(json.dumps(res) + "\n")
            f.flush()  # Saved immediately, so a quota stop or Ctrl+C loses nothing
            counts["written"] += 1

        async def worker():
            # Each slot starts its next script as soon as the previous one finishes (no batch barrier)
            while not stop.is_set():
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    res, model = await backend.generate(task)
                except QuotaExhausted:
                    stop.set()
                    return
                progress.update(1)
                if res:
                    save(task, res, model)
                else:
                    counts["failed"] += 1

        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    progress.close()
    n_written, n_failed, n_duplicates = counts["written"], counts["failed"], counts["duplicates"]
    stopped = stop.is_set()

    print(f"Finished: {n_written} new dialogues written, {n_failed} failed, {n_duplicates} duplicates dropped.")
    if stopped:
        print("⏸️  Every model hit its quota. Re-run the same command later to resume; it only fills what is missing.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
