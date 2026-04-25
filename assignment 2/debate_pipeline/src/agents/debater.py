from __future__ import annotations
import os
import re
import time
from pathlib import Path
from typing import Literal
from openai import OpenAI


class DebaterAgent:
    PROMPT_FILES = {"A": "debater_a.txt", "B": "debater_b.txt"}

    def __init__(self, role: Literal["A", "B"], position: str, opponent_position: str, config: dict) -> None:
        self.role = role
        self.position = position
        self.opponent_position = opponent_position
        self.cfg = config
        self.client = OpenAI(
            api_key=config["api"]["key"],
            base_url=config["api"]["base_url"]
        )
        self.model = config["model"]["debater"]
        self.temperature = config["generation"]["debater_temperature"]
        self.max_tokens = config["generation"]["debater_max_tokens"]
        prompt_path = Path(config["paths"]["prompts_dir"]) / self.PROMPT_FILES[role]
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def generate_argument(self, question: str, context: str, debate_history: list[dict], round_num: int) -> dict:
        history_text = self._format_history(debate_history)
        prompt = self._build_prompt(question, context, history_text)
        raw = self._call_api(prompt)
        parsed = self._parse_response(raw)
        parsed["position"] = self.position
        parsed["round"] = round_num
        parsed["role"] = self.role
        return parsed

    def _build_prompt(self, question: str, context: str, history_text: str) -> str:
        replacements = {
            "{question}": question,
            "{context}": context if context else "No additional context provided.",
            "{debate_history}": history_text if history_text else "No prior rounds.",
            "{position}": self.position,
            "{opponent_position}": self.opponent_position,
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
        raise RuntimeError("API call failed after 3 attempts")

    def _parse_response(self, raw: str) -> dict:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        argument = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        return {"reasoning": reasoning, "argument": argument, "full_text": raw}

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return ""
        lines = []
        for entry in history:
            lines.append(f"--- Round {entry['round']} | Debater {entry['role']} (position: {entry['position']}) ---\n{entry['argument']}\n")
        return "\n".join(lines)