from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI


class JudgeAgent:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.client = OpenAI(
            api_key=config["api"]["key"],
            base_url=config["api"]["base_url"]
        )
        self.model = config["model"]["judge"]
        self.temperature = config["generation"]["judge_temperature"]
        self.max_tokens = config["generation"]["judge_max_tokens"]
        prompt_path = Path(config["paths"]["prompts_dir"]) / "judge.txt"
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def judge_debate(self, question: str, position_a: str, position_b: str, debate_history: list[dict]) -> dict:
        transcript = self._format_transcript(debate_history)
        prompt = self._build_prompt(question, position_a, position_b, transcript)
        raw = self._call_api(prompt)
        result = self._parse_verdict(raw)
        result["raw_response"] = raw
        return result

    def _build_prompt(self, question: str, position_a: str, position_b: str, transcript: str) -> str:
        replacements = {
            "{question}": question,
            "{position_a}": position_a,
            "{position_b}": position_b,
            "{transcript}": transcript,
        }
        prompt = self.prompt_template
        for k, v in replacements.items():
            prompt = prompt.replace(k, v)
        return prompt

    def _call_api(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt * 3)
        raise RuntimeError("Judge API call failed after 3 attempts")

    def _parse_verdict(self, raw: str) -> dict:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {
            "cot_analysis": {},
            "strongest_arguments": {},
            "weakest_arguments": {},
            "verdict": "PARSE_ERROR",
            "winner": "UNKNOWN",
            "reasoning_summary": raw[:500],
            "confidence": 1,
        }

    @staticmethod
    def _format_transcript(history: list[dict]) -> str:
        lines = []
        for entry in history:
            header = f"=== Round {entry['round']} | Debater {entry['role']} (arguing: {entry['position']}) ==="
            lines.append(f"{header}\n{entry['argument']}\n")
        return "\n".join(lines)