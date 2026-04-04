"""
data/construct_json_instruct.py
================================
Implements the imitation-learning data-construction pipeline for Stage 2.

Steps:
  1. Load prompt templates for each of the 5 required task types.
  2. Feed each prompt to the teacher model (Llama 3.1 70B Instruct via API
     or local HF path on UTSA HPC).
  3. Validate every response for JSON correctness; discard/regenerate invalid.
  4. Save validated (instruction, input, output) pairs to JSONL.

Usage:
    python data/construct_json_instruct.py --config config.yaml

Environment variables:
    TOGETHER_API_KEY  — API key for Together AI (if using API-based teacher)
    TEACHER_API_BASE  — override the API base URL
"""

import argparse
import json
import os
import random
import re
import time
import yaml
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt seeds — diverse task inputs for each category
# ---------------------------------------------------------------------------

TASK_SEEDS = {
    "json_extraction": [
        {"input": "John Smith was born on March 15, 1982 in Austin, Texas. He works at Acme Corp as a Senior Engineer and earns $120,000 per year.", "schema_hint": '{"name": str, "birth_date": str, "city": str, "company": str, "title": str, "salary": int}'},
        {"input": "Order #4892: 3x Wireless Headphones at $79.99 each, 1x USB-C Cable at $12.49. Shipped to 742 Evergreen Terrace, Springfield, IL 62704. Expected delivery: 2024-03-22.", "schema_hint": '{"order_id": str, "items": list, "total": float, "address": str, "delivery_date": str}'},
        {"input": "The patient, Maria Gonzalez (DOB: 07/04/1990), was admitted on 2024-01-10 with a diagnosis of acute appendicitis. The attending physician was Dr. Chen. She was discharged on 2024-01-13.", "schema_hint": '{"patient_name": str, "dob": str, "admission_date": str, "discharge_date": str, "diagnosis": str, "physician": str}'},
        {"input": "Flight BA249 departs London Heathrow (LHR) at 09:15 on 15 April 2024 and arrives at New York JFK at 12:30 local time. Aircraft: Boeing 777. Seats available: 42.", "schema_hint": '{"flight": str, "origin": str, "destination": str, "departure": str, "arrival": str, "aircraft": str, "seats_available": int}'},
        {"input": "The research paper 'Attention Is All You Need' was published in 2017 by Vaswani et al. at Google Brain. It introduced the Transformer architecture and has been cited over 90,000 times.", "schema_hint": '{"title": str, "year": int, "authors": str, "institution": str, "contribution": str, "citations": int}'},
        {"input": "Netflix reported Q3 2024 revenue of $9.82 billion, up 15% year-over-year, with 282.7 million paid subscribers globally. Operating income was $2.91 billion.", "schema_hint": '{"company": str, "quarter": str, "revenue_billion": float, "yoy_growth_pct": float, "subscribers_million": float, "operating_income_billion": float}'},
        {"input": "Sarah Connor, contact: sarah.connor@skynet.io, +1-555-0194. Lives at 123 Resistance Blvd, Los Angeles CA 90001. LinkedIn: linkedin.com/in/sarahconnor.", "schema_hint": '{"name": str, "email": str, "phone": str, "address": str, "linkedin": str}'},
    ],
    "schema_constrained": [
        {"schema": '{"product": {"name": str, "sku": str, "price": float, "in_stock": bool, "tags": [str]}}', "context": "Generate a product listing for a mechanical keyboard."},
        {"schema": '{"user": {"id": str, "username": str, "email": str, "created_at": str, "roles": [str], "preferences": {"theme": str, "notifications": bool}}}', "context": "Generate a user profile for a software developer named Alex."},
        {"schema": '{"weather_report": {"location": str, "date": str, "temperature_c": float, "humidity_pct": int, "condition": str, "wind_kph": float, "forecast": [{"day": str, "high_c": float, "low_c": float}]}}', "context": "Generate a weather report for San Antonio, TX for today."},
        {"schema": '{"job_posting": {"title": str, "company": str, "location": str, "remote": bool, "salary_min": int, "salary_max": int, "requirements": [str], "responsibilities": [str]}}', "context": "Generate a job posting for a Senior ML Engineer at a tech startup."},
        {"schema": '{"book": {"title": str, "author": str, "isbn": str, "genre": str, "year": int, "rating": float, "summary": str, "available": bool}}', "context": "Generate a library record for a science fiction novel."},
        {"schema": '{"api_response": {"status": str, "code": int, "data": {"items": [{"id": int, "name": str, "value": float}], "total": int, "page": int}, "timestamp": str}}', "context": "Generate a paginated API response for a product catalogue query returning 3 items."},
    ],
    "classification": [
        {"text": "The battery drains within 2 hours and the screen flickers constantly. Worst purchase I've ever made!", "labels": ["positive", "negative", "neutral"], "task": "sentiment classification"},
        {"text": "URGENT: Your bank account has been compromised. Click here immediately to verify your identity and prevent account closure.", "labels": ["spam", "phishing", "legitimate", "promotional"], "task": "email classification"},
        {"text": "How do I reset my password? I've tried three times and it keeps saying invalid credentials.", "labels": ["billing", "technical_support", "account_access", "feature_request", "general_inquiry"], "task": "customer support ticket classification"},
        {"text": "Researchers at MIT have developed a new catalyst that could make hydrogen fuel cells 40% more efficient.", "labels": ["technology", "science", "business", "politics", "sports", "entertainment"], "task": "news article topic classification"},
        {"text": "I'm having chest pain and shortness of breath. Should I be worried?", "labels": ["emergency", "routine_consultation", "medication_query", "mental_health", "general_wellness"], "task": "medical query urgency classification"},
        {"text": "Can you make the font bigger and change the background to dark mode?", "labels": ["bug_report", "feature_request", "ui_feedback", "performance_issue", "security_concern"], "task": "user feedback classification"},
        {"text": "The package was supposed to arrive last Tuesday but tracking still shows 'in transit'.", "labels": ["lost_package", "delayed_delivery", "wrong_item", "damaged_item", "return_request"], "task": "e-commerce issue classification"},
    ],
    "json_repair": [
        {"broken": "{'name': 'Alice', 'age': 30, 'city': 'Boston'}", "issue": "Single quotes instead of double quotes"},
        {"broken": '{"product": "laptop", "price": 999.99, "specs": {"ram": "16GB", "storage": "512GB"', "issue": "Missing closing braces"},
        {"broken": '{"items": [1, 2, 3,], "total": 6,}', "issue": "Trailing commas"},
        {"broken": '{"name": "Bob", "scores": [85, 92, 78, "N/A", 90], "passed": True}', "issue": "Python True instead of JSON true, mixed types in array"},
        {"broken": '{"user": {"id": 42, "email": "test@example.com" "phone": "555-1234"}}', "issue": "Missing comma between fields"},
        {"broken": '{"data": {"value": NaN, "count": Infinity, "label": undefined}}', "issue": "JavaScript-style NaN, Infinity, undefined not valid in JSON"},
        {"broken": '{"message": "He said "hello" and left", "timestamp": "2024-01-15"}', "issue": "Unescaped quotes inside string value"},
        {"broken": "{\n  name: John,\n  age: 25,\n  active: yes\n}", "issue": "Unquoted keys and values"},
    ],
    "tool_call": [
        {"function": "search_flights", "params_schema": '{"origin": str, "destination": str, "date": str, "passengers": int, "class": str}', "user_request": "Find me business class flights from Dallas to Tokyo on June 10th for 2 people."},
        {"function": "send_email", "params_schema": '{"to": [str], "subject": str, "body": str, "cc": [str], "priority": str}', "user_request": "Email the team at dev@company.com and qa@company.com that the deployment is complete. Mark it as high priority."},
        {"function": "create_calendar_event", "params_schema": '{"title": str, "start_datetime": str, "end_datetime": str, "attendees": [str], "location": str, "description": str}', "user_request": "Schedule a 1-hour product review meeting for tomorrow at 3pm with alice@corp.com and bob@corp.com in Conference Room B."},
        {"function": "query_database", "params_schema": '{"table": str, "filters": {}, "columns": [str], "limit": int, "order_by": str, "order": str}', "user_request": "Get the top 10 customers by total purchase amount from the orders table, showing name and email."},
        {"function": "translate_text", "params_schema": '{"text": str, "source_language": str, "target_language": str, "formality": str}', "user_request": "Translate 'Good morning, how can I help you today?' from English to formal Japanese."},
        {"function": "set_device_setting", "params_schema": '{"device_id": str, "setting": str, "value": str | int | bool, "apply_to_group": bool}', "user_request": "Turn off the lights in the living room and set the thermostat to 68 degrees."},
        {"function": "process_payment", "params_schema": '{"amount": float, "currency": str, "method": str, "recipient_id": str, "description": str, "idempotency_key": str}', "user_request": "Send $250 via credit card to vendor ID V-9923 for the October invoice."},
    ],
}


# ---------------------------------------------------------------------------
# API / local model interface
# ---------------------------------------------------------------------------

def call_teacher_api(prompt: str, cfg: dict) -> str:
    """Call the teacher model via an OpenAI-compatible API."""
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY", ""),
        base_url=cfg.get("teacher_api_base", "https://api.together.xyz/v1"),
    )
    response = client.chat.completions.create(
        model=cfg["teacher_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_teacher_local(prompt: str, pipe) -> str:
    """Call a locally-loaded HuggingFace pipeline."""
    out = pipe(prompt, max_new_tokens=1024, temperature=0.3, do_sample=True)
    text = out[0]["generated_text"]
    # Strip the prompt prefix if the pipeline echoes it
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def load_local_pipeline(model_path: str):
    """Load a local HuggingFace text-generation pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def load_prompt_template(task_type: str, prompts_dir: Path) -> str:
    path = prompts_dir / "teacher_generation" / f"{task_type}.txt"
    return path.read_text().strip()


def build_teacher_prompt(task_type: str, seed: dict, template: str) -> tuple[str, str, str]:
    """Return (instruction, input_field, filled_teacher_prompt)."""
    if task_type == "json_extraction":
        instruction = (
            "Extract the relevant information from the text below and return it "
            f"as a JSON object with the following schema: {seed['schema_hint']}. "
            "Respond ONLY with valid JSON — no explanation, no markdown fences."
        )
        inp = seed["input"]
        teacher_prompt = template.format(instruction=instruction, input=inp)

    elif task_type == "schema_constrained":
        instruction = (
            f"Generate a JSON object that strictly conforms to this schema: {seed['schema']}. "
            "All fields must be present with appropriate values. "
            "Respond ONLY with valid JSON — no explanation, no markdown fences."
        )
        inp = seed["context"]
        teacher_prompt = template.format(instruction=instruction, input=inp)

    elif task_type == "classification":
        instruction = (
            f"Classify the following text. Task: {seed['task']}. "
            f"Choose exactly one label from: {seed['labels']}. "
            'Respond ONLY with a JSON object in this format: {"label": "<chosen_label>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}.'
        )
        inp = seed["text"]
        teacher_prompt = template.format(instruction=instruction, input=inp)

    elif task_type == "json_repair":
        instruction = (
            "The JSON below is malformed. Fix it and return ONLY valid, "
            "well-formatted JSON with no explanation or markdown fences. "
            f"Known issue: {seed['issue']}."
        )
        inp = seed["broken"]
        teacher_prompt = template.format(instruction=instruction, input=inp)

    elif task_type == "tool_call":
        instruction = (
            f"Generate a JSON object representing a call to the function `{seed['function']}`. "
            f"The function signature is: {seed['params_schema']}. "
            "Fill in all parameters based on the user request. "
            "Respond ONLY with valid JSON — no explanation, no markdown fences."
        )
        inp = seed["user_request"]
        teacher_prompt = template.format(instruction=instruction, input=inp)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return instruction, inp, teacher_prompt


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------

def extract_and_validate_json(text: str) -> Optional[str]:
    """Try to extract valid JSON from raw model output."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Try to find a JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    # Last resort: try the whole string
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(cfg: dict, prompts_dir: Path, max_retries: int = 3):
    data_cfg     = cfg["data"]
    task_types   = data_cfg["json_task_types"]
    per_task     = data_cfg["examples_per_task"]
    use_api      = not cfg.get("teacher_local_path", "")

    # Load local pipeline once if needed
    local_pipe = None
    if not use_api:
        print(f"[construct] Loading local teacher from {cfg['teacher_local_path']}")
        local_pipe = load_local_pipeline(cfg["teacher_local_path"])

    all_examples = []

    for task_type in task_types:
        print(f"\n[construct] Generating task: {task_type}")
        template = load_prompt_template(task_type, prompts_dir)
        seeds    = TASK_SEEDS[task_type]
        generated = 0
        attempt   = 0

        # Cycle through seeds until we hit per_task or exhaust attempts
        seed_cycle = (seeds * ((per_task // len(seeds)) + 2))[:per_task * 2]

        for seed in seed_cycle:
            if generated >= per_task:
                break

            instruction, inp, teacher_prompt = build_teacher_prompt(task_type, seed, template)

            raw_output = None
            for retry in range(max_retries):
                try:
                    if use_api:
                        raw_output = call_teacher_api(teacher_prompt, cfg)
                    else:
                        raw_output = call_teacher_local(teacher_prompt, local_pipe)
                    break
                except Exception as e:
                    print(f"  [retry {retry+1}] Error calling teacher: {e}")
                    time.sleep(2 ** retry)

            if raw_output is None:
                print(f"  [skip] Failed after {max_retries} retries")
                continue

            validated_json = extract_and_validate_json(raw_output)
            if validated_json is None:
                print(f"  [skip] Invalid JSON from teacher output")
                continue

            all_examples.append({
                "task_type":   task_type,
                "instruction": instruction,
                "input":       inp,
                "output":      validated_json,
            })
            generated += 1
            if generated % 10 == 0:
                print(f"  [{task_type}] {generated}/{per_task} valid examples")

        print(f"  [{task_type}] Final: {generated} examples collected")

    return all_examples


def format_prompt(ex: dict) -> str:
    """Convert to Phi-3.5 chat format."""
    system = "You are a helpful assistant. Always respond with valid JSON when asked."
    user_turn = ex["instruction"]
    if ex.get("input"):
        user_turn += f"\n\n{ex['input']}"
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_turn}<|end|>\n"
        f"<|assistant|>\n{ex['output']}<|end|>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(args.seed)

    prompts_dir = Path(cfg["paths"]["prompts_dir"])
    data_dir    = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_dataset(cfg, prompts_dir, max_retries=args.max_retries)

    total_needed = cfg["data"]["json_instruct_train_size"] + cfg["data"]["json_instruct_eval_size"]
    if len(examples) < total_needed:
        print(f"[WARNING] Only {len(examples)} valid examples; need {total_needed}. "
              f"Increase examples_per_task or check teacher API.")

    random.shuffle(examples)
    eval_size   = cfg["data"]["json_instruct_eval_size"]
    eval_split  = examples[:eval_size]
    train_split = examples[eval_size:]

    def save_jsonl(path: Path, items: list):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"[construct] Saved {len(items)} examples → {path}")

    save_jsonl(data_dir / "json_instruct_train.jsonl", train_split)
    save_jsonl(data_dir / "json_instruct_eval.jsonl",  eval_split)

    # Formatted versions for training
    formatted_train = [{"text": format_prompt(ex)} for ex in train_split]
    formatted_eval  = [{"text": format_prompt(ex)} for ex in eval_split]
    save_jsonl(data_dir / "json_instruct_train_formatted.jsonl", formatted_train)
    save_jsonl(data_dir / "json_instruct_eval_formatted.jsonl",  formatted_eval)

    # Stats
    from collections import Counter
    counts = Counter(ex["task_type"] for ex in examples)
    print("\n[construct] Task type distribution:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\n[construct] Done. Total: {len(examples)} examples "
          f"({len(train_split)} train / {len(eval_split)} eval)")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
