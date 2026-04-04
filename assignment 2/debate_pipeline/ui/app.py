"""
app.py — Streamlit Web UI for the LLM Debate Pipeline.

Run with:  streamlit run ui/app.py

Provides:
  - Question input panel
  - Round-by-round debate display with expandable CoT
  - Judge verdict panel with confidence visualization
  - Export debate transcript as JSON
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import DebateOrchestrator

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Debate Pipeline",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.debater-a-box {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0;
}
.debater-b-box {
    background: #fef3e2;
    border-left: 4px solid #FF9800;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0;
}
.judge-box {
    background: #f3e5f5;
    border-left: 4px solid #9C27B0;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 12px 0;
}
.correct-badge {
    background: #4CAF50;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: bold;
}
.incorrect-badge {
    background: #f44336;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: bold;
}
.round-header {
    font-size: 1.1em;
    font-weight: bold;
    color: #424242;
    margin-top: 16px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample_questions():
    path = Path(__file__).parent.parent / "data" / "sample_questions.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def run_debate_streaming(config, question, answer, position_a, position_b, context):
    """Run debate with live UI updates."""
    orchestrator = DebateOrchestrator(config)

    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        status = st.status("Running debate...", expanded=True)

        with status:
            st.write("**Phase 1:** Initializing debater positions...")

    result = orchestrator.run_debate(
        question=question,
        ground_truth=answer,
        position_a=position_a,
        position_b=position_b,
        context=context,
        question_id=f"ui_{int(time.time())}",
    )

    with progress_container:
        status.update(label="Debate complete!", state="complete")

    return result


def render_debate_round(entry: dict, round_num: int, role: str):
    """Render a single debate turn."""
    color_class = "debater-a-box" if role == "A" else "debater-b-box"
    emoji = "🔵" if role == "A" else "🟠"
    label = "Debater A (Proponent)" if role == "A" else "Debater B (Opponent)"

    st.markdown(f'<div class="round-header">{emoji} {label} — Round {round_num}</div>',
                unsafe_allow_html=True)

    # Show argument
    argument_text = entry.get("argument", entry.get("full_text", ""))
    # Remove the POSITION: line for cleaner display
    display_text = "\n".join(
        line for line in argument_text.split("\n")
        if not line.strip().startswith("**POSITION:**")
    ).strip()

    st.markdown(f'<div class="{color_class}">{display_text}</div>',
                unsafe_allow_html=True)

    # Show CoT reasoning in expander
    reasoning = entry.get("reasoning", "")
    if reasoning:
        with st.expander("🧠 View Chain-of-Thought Reasoning"):
            st.markdown(f"*{reasoning}*")


def render_judge_verdict(phase3: dict, phase4: dict):
    """Render the judge's verdict panel."""
    st.subheader("⚖️ Judge's Verdict")

    verdict = phase3.get("verdict", "Unknown")
    confidence = phase3.get("confidence", "?")
    winner = phase3.get("winner", "?")
    reasoning = phase3.get("reasoning_summary", "")
    correct = phase4.get("correct", False)
    ground_truth = phase4.get("ground_truth", "")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verdict", verdict)
    with col2:
        st.metric("Winner", f"Debater {winner}")
    with col3:
        st.metric("Confidence", f"{confidence}/5")

    # Correct/incorrect badge
    if correct:
        st.markdown('<span class="correct-badge">✓ CORRECT</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="incorrect-badge">✗ INCORRECT (Ground truth: {ground_truth})</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f'<div class="judge-box"><b>Judge\'s Reasoning:</b><br>{reasoning}</div>',
                unsafe_allow_html=True)

    # Detailed analysis expander
    cot = phase3.get("cot_analysis", {})
    strongest = phase3.get("strongest_arguments", {})
    weakest = phase3.get("weakest_arguments", {})

    if cot or strongest:
        with st.expander("📊 Detailed Judge Analysis"):
            if cot:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Debater A Analysis**")
                    if isinstance(cot.get("debater_a"), dict):
                        for k, v in cot["debater_a"].items():
                            st.markdown(f"*{k.replace('_', ' ').title()}:* {v}")
                with col_b:
                    st.markdown("**Debater B Analysis**")
                    if isinstance(cot.get("debater_b"), dict):
                        for k, v in cot["debater_b"].items():
                            st.markdown(f"*{k.replace('_', ' ').title()}:* {v}")

            if strongest:
                st.markdown("---")
                st.markdown("**Strongest Arguments**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"*Debater A:* {strongest.get('debater_a', 'N/A')}")
                with col_b:
                    st.markdown(f"*Debater B:* {strongest.get('debater_b', 'N/A')}")

            if weakest:
                st.markdown("**Weakest Arguments**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"*Debater A:* {weakest.get('debater_a', 'N/A')}")
                with col_b:
                    st.markdown(f"*Debater B:* {weakest.get('debater_b', 'N/A')}")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Your API key. Not stored anywhere.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### Debate Settings")

    num_rounds = st.slider("Number of Rounds", min_value=1, max_value=5, value=3)
    early_stop = st.checkbox("Adaptive Early Stopping", value=True)

    st.markdown("---")
    st.markdown("### Load Sample Question")
    samples = load_sample_questions()
    sample_labels = [f"{q['id']}: {q['question'][:60]}..." for q in samples]
    selected_sample_idx = st.selectbox(
        "Choose a sample",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Multi-agent LLM debate system based on Irving et al. (2018). "
        "Two debaters argue opposing positions; a judge evaluates the transcript."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

st.title("⚖️ LLM Debate Pipeline")
st.markdown(
    "Two LLM agents argue opposing sides of a question; "
    "a third LLM judge evaluates the transcript and renders a verdict."
)
st.markdown("---")

# ── Question Input ─────────────────────────────────────────────────────────────
st.subheader("📝 Question Setup")

# Auto-fill from sample
sample = samples[selected_sample_idx] if samples else {}

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Question",
        value=sample.get("question", ""),
        height=80,
    )
    context = st.text_area(
        "Context / Evidence (optional)",
        value=sample.get("context", ""),
        height=80,
        help="Background information both debaters receive.",
    )

with col2:
    position_a = st.text_input(
        "Debater A's Position (Proponent)",
        value=sample.get("position_a", "yes"),
    )
    position_b = st.text_input(
        "Debater B's Position (Opponent)",
        value=sample.get("position_b", "no"),
    )
    ground_truth = st.text_input(
        "Ground Truth Answer",
        value=sample.get("answer", ""),
        help="Used for evaluation only — not shown to debaters.",
    )

run_button = st.button(
    "🚀 Start Debate",
    type="primary",
    disabled=not api_key or not question.strip(),
)

if not api_key:
    st.warning("⚠️ Enter your Anthropic API key in the sidebar to run debates.")

# ── Run Debate ─────────────────────────────────────────────────────────────────
if run_button and question.strip():
    config = load_config()
    config["debate"]["num_rounds"] = num_rounds
    config["debate"]["early_stop_consensus"] = early_stop

    st.markdown("---")
    st.subheader("🎙️ Debate Transcript")

    with st.spinner("Running debate... this may take 30–90 seconds"):
        try:
            result = run_debate_streaming(
                config, question, ground_truth, position_a, position_b, context
            )
        except Exception as e:
            st.error(f"Error running debate: {e}")
            st.stop()

    # Render rounds
    phase2 = result["phases"].get("phase2", {})
    rounds = phase2.get("rounds", [])

    if phase2.get("skipped"):
        st.info("✅ Both debaters agreed immediately — no debate rounds needed.")
    else:
        # Group by round number
        round_map: dict[int, list] = {}
        for entry in rounds:
            rnd = entry["round"]
            round_map.setdefault(rnd, []).append(entry)

        for rnd_num, entries in sorted(round_map.items()):
            st.markdown(f"### Round {rnd_num}")
            for entry in entries:
                render_debate_round(entry, rnd_num, entry["role"])

    st.markdown("---")

    # Render verdict
    phase3 = result["phases"].get("phase3", {})
    phase4 = result["phases"].get("phase4", {})
    render_judge_verdict(phase3, phase4)

    # Export
    st.markdown("---")
    st.subheader("📥 Export Transcript")
    transcript_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download JSON Transcript",
        data=transcript_json,
        file_name=f"debate_{result['question_id']}.json",
        mime="application/json",
    )

    # Store in session for display persistence
    st.session_state["last_result"] = result

# ── Show previous result if re-rendering ─────────────────────────────────────
elif "last_result" in st.session_state and not run_button:
    st.info("Showing last debate result. Press 'Start Debate' to run a new one.")
