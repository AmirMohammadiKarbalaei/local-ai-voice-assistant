import time
import streamlit as st

from jarvis.assistant import JarvisAssistant

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(
    page_title="Jarvis",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def load_assistant():
    return JarvisAssistant()


assistant = load_assistant()


# -------------------------
# Session state
# -------------------------

if "ui_running" not in st.session_state:
    st.session_state.ui_running = False

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "last_status" not in st.session_state:
    st.session_state.last_status = "Sleeping"

if "processing" not in st.session_state:
    st.session_state.processing = False


# -------------------------
# Styles
# -------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background: radial-gradient(circle at top, #0b1020 0%, #050816 45%, #02040c 100%);
        color: #e6f7ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    .jarvis-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #dffbff;
        text-shadow: 0 0 12px rgba(82, 240, 255, 0.45);
    }

    .jarvis-subtitle {
        text-align: center;
        color: #8fb8c7;
        margin-bottom: 2rem;
        font-size: 1rem;
    }

    .status-pill {
        width: fit-content;
        margin: 0 auto 1.25rem auto;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        border: 1px solid rgba(92, 234, 255, 0.35);
        background: rgba(14, 30, 47, 0.6);
        color: #d8fbff;
        font-size: 0.95rem;
        box-shadow: 0 0 20px rgba(0, 225, 255, 0.08);
    }

    .jarvis-panel {
        background: rgba(13, 19, 33, 0.72);
        border: 1px solid rgba(93, 226, 255, 0.14);
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow:
            inset 0 0 30px rgba(0, 255, 255, 0.03),
            0 0 30px rgba(0, 195, 255, 0.08);
        backdrop-filter: blur(10px);
    }

div.stButton {
    display: flex;
    justify-content: center;
    align-items: center;
}

div.stButton > button {
    width: 240px;
    height: 240px;
    border-radius: 50%;
    border: 2px solid rgba(102, 246, 255, 0.9);
    color: #e8fdff;
    font-size: 1.15rem;
    font-weight: 700;
    background:
        radial-gradient(circle at 35% 30%, rgba(130,255,255,0.75), rgba(0,189,255,0.22) 28%, rgba(5,18,38,0.96) 68%),
        radial-gradient(circle at center, rgba(19, 35, 59, 0.9), rgba(8, 13, 27, 0.98));
    box-shadow:
        0 0 14px rgba(0, 238, 255, 0.45),
        0 0 36px rgba(0, 195, 255, 0.32),
        inset 0 0 24px rgba(164, 255, 255, 0.25);
    transition: all 0.2s ease-in-out;
}

    div.stButton > button:hover {
        transform: scale(1.02);
        border-color: rgba(169, 252, 255, 1);
        box-shadow:
            0 0 24px rgba(0, 238, 255, 0.65),
            0 0 54px rgba(0, 195, 255, 0.42),
            inset 0 0 36px rgba(164, 255, 255, 0.30);
    }

    div.stButton > button:focus {
        outline: none !important;
        box-shadow:
            0 0 24px rgba(0, 238, 255, 0.65),
            0 0 54px rgba(0, 195, 255, 0.42),
            inset 0 0 36px rgba(164, 255, 255, 0.30) !important;
    }

    .hint-text {
        text-align: center;
        color: #91b4c6;
        font-size: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 1.4rem;
    }

    .chat-wrap {
        margin-top: 1.5rem;
    }

    .chat-row-user, .chat-row-jarvis {
        margin-bottom: 0.85rem;
        display: flex;
    }

    .chat-row-user {
        justify-content: flex-end;
    }

    .chat-row-jarvis {
        justify-content: flex-start;
    }

    .bubble-user, .bubble-jarvis {
        max-width: 80%;
        padding: 0.9rem 1rem;
        border-radius: 18px;
        line-height: 1.45;
        font-size: 0.98rem;
        white-space: pre-wrap;
    }

    .bubble-user {
        background: linear-gradient(135deg, rgba(20, 73, 128, 0.95), rgba(10, 119, 177, 0.88));
        color: #f2fbff;
        border: 1px solid rgba(118, 220, 255, 0.18);
    }

    .bubble-jarvis {
        background: rgba(16, 25, 42, 0.92);
        color: #def7ff;
        border: 1px solid rgba(113, 223, 255, 0.18);
    }

    .small-note {
        text-align: center;
        color: #6f91a4;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Helpers
# -------------------------

def add_transcript(role: str, text: str):
    st.session_state.transcript.append({"role": role, "text": text})


def reset_conversation_state():
    assistant.in_conversation = False


def soft_sleep(reply: str = "Going to sleep. Click the core to wake me again."):
    st.session_state.ui_running = False
    reset_conversation_state()
    st.session_state.last_status = "Sleeping"
    assistant.speak_blocking(reply)
    add_transcript("assistant", reply)


def handle_one_voice_turn():
    st.session_state.processing = True
    st.session_state.last_status = "Listening"

    try:
        text = assistant.listener.listen()

        if not text:
            st.session_state.last_status = (
                "Waiting for wake word..." if not assistant.in_conversation else "Listening"
            )
            return

        # Full shutdown phrase inside UI -> just put Jarvis to sleep
        if assistant.is_shutdown_command(text):
            add_transcript("user", text)
            soft_sleep("Going to sleep. Click the core to wake me again.")
            return

        # Idle mode: require wake word
        if not assistant.in_conversation:
            if not assistant.is_start_command(text):
                st.session_state.last_status = "Waiting for wake word..."
                return

            add_transcript("user", text)
            assistant.in_conversation = True

            command = assistant.remove_start_word(text)

            if not command:
                reply = "Yes?"
                assistant.speak_blocking(reply)
                add_transcript("assistant", reply)
                st.session_state.last_status = "Conversation mode"
                return

        # Conversation mode: normal utterance
        else:
            command = text.strip()
            add_transcript("user", text)

        # Leave conversation mode
        if assistant.is_conversation_stop_command(command):
            assistant.in_conversation = False
            reply = "Okay. Say Jarvis when you need me."
            assistant.speak_blocking(reply)
            add_transcript("assistant", reply)
            st.session_state.last_status = "Waiting for wake word..."
            return

        st.session_state.last_status = "Thinking"
        reply = assistant.ask_ollama_streaming(command)
        assistant.tts.wait()

        if reply:
            add_transcript("assistant", reply)

        st.session_state.last_status = (
            "Conversation mode" if assistant.in_conversation else "Waiting for wake word..."
        )

    finally:
        st.session_state.processing = False


# -------------------------
# Header
# -------------------------

st.markdown('<div class="jarvis-title">JARVIS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="jarvis-subtitle">Local voice assistant • Ollama • Tools • Kokoro</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="status-pill">Status: {st.session_state.last_status}</div>',
    unsafe_allow_html=True,
)


# -------------------------
# Main panel
# -------------------------

st.markdown(
    """
    <div class="jarvis-panel">
    """,
    unsafe_allow_html=True,
)

button_text = "PUT JARVIS TO SLEEP" if st.session_state.ui_running else "WAKE JARVIS"

left, centre, right = st.columns([1.2, 1, 1.2])

with centre:
    clicked = st.button(button_text, use_container_width=False)

if clicked:
    if st.session_state.ui_running:
        st.session_state.ui_running = False
        reset_conversation_state()
        st.session_state.last_status = "Sleeping"
    else:
        st.session_state.ui_running = True
        st.session_state.last_status = "Waiting for wake word..."

st.markdown(
    """
    <div class="hint-text">
        Click the core once to wake Jarvis.<br>
        Then talk through your microphone just like in the CLI.<br>
        Say <b>Jarvis</b> to start, <b>bye</b> to leave conversation mode,
        or <b>Jarvis sleep</b> to put it back to sleep.
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Transcript
# -------------------------

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

for item in st.session_state.transcript[-20:]:
    if item["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-row-user">
                <div class="bubble-user">{item["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row-jarvis">
                <div class="bubble-jarvis">{item["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="small-note">Close the browser tab or stop the Streamlit process if you want to fully exit the UI.</div>',
    unsafe_allow_html=True,
)


# -------------------------
# Continuous voice loop
# -------------------------

if st.session_state.ui_running and not st.session_state.processing:
    handle_one_voice_turn()
    time.sleep(0.15)
    st.rerun()