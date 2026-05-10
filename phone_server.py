import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request

load_dotenv()

# Hide unnecessary Hugging Face Windows cache warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from jarvis.assistant import JarvisAssistant


app = Flask(__name__)

assistant = JarvisAssistant()

# Browser will speak the reply. Laptop Kokoro TTS is muted.
assistant.tts.enabled = False

assistant_lock = threading.Lock()


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>jarvis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, sans-serif;
      background: radial-gradient(circle at top, #0b1830 0%, #050816 50%, #01030a 100%);
      color: #e8fbff;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .app {
      width: min(92vw, 720px);
      text-align: center;
    }

    h1 {
      letter-spacing: 0.22em;
      font-size: 2.4rem;
      margin-bottom: 0.2rem;
      text-shadow: 0 0 20px rgba(0, 230, 255, 0.55);
    }

    .status {
      color: #8fb8c7;
      margin-bottom: 2rem;
      min-height: 1.4rem;
    }

    .core-wrap {
      display: flex;
      justify-content: center;
      align-items: center;
      margin: 1.5rem 0;
    }

    .core {
      width: 230px;
      height: 230px;
      border-radius: 50%;
      border: 2px solid rgba(120, 245, 255, 0.95);
      background:
        radial-gradient(circle at 35% 30%, rgba(160, 255, 255, 0.9), rgba(0, 190, 255, 0.25) 30%, rgba(5, 18, 38, 0.96) 68%),
        radial-gradient(circle at center, rgba(20, 45, 75, 0.9), rgba(4, 8, 20, 1));
      box-shadow:
        0 0 20px rgba(0, 238, 255, 0.55),
        0 0 70px rgba(0, 140, 255, 0.35),
        inset 0 0 35px rgba(180, 255, 255, 0.3);
      color: #ecfdff;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .core.active {
      transform: scale(1.04);
      box-shadow:
        0 0 28px rgba(0, 238, 255, 0.85),
        0 0 95px rgba(0, 140, 255, 0.55),
        inset 0 0 45px rgba(180, 255, 255, 0.38);
    }

    .core:active {
      transform: scale(0.98);
    }

    .hint {
      color: #91b4c6;
      font-size: 0.95rem;
      line-height: 1.5;
      margin-bottom: 1.5rem;
    }

    .transcript {
      margin-top: 1.5rem;
      text-align: left;
      max-height: 35vh;
      overflow-y: auto;
      padding: 0.6rem;
    }

    .bubble {
      padding: 0.85rem 1rem;
      margin: 0.7rem 0;
      border-radius: 16px;
      line-height: 1.45;
      max-width: 86%;
      word-wrap: break-word;
    }

    .user {
      margin-left: auto;
      background: linear-gradient(135deg, rgba(18, 83, 145, 0.95), rgba(0, 132, 190, 0.85));
      border: 1px solid rgba(116, 220, 255, 0.22);
    }

    .jarvis {
      margin-right: auto;
      background: rgba(16, 25, 42, 0.92);
      border: 1px solid rgba(113, 223, 255, 0.18);
    }
    .controls {
  display: flex;
  justify-content: center;
  margin-top: 1rem;
  margin-bottom: 1rem;
}

.reset-btn {
  border: 1px solid rgba(120, 245, 255, 0.35);
  background: rgba(8, 18, 35, 0.82);
  color: #dffbff;
  padding: 0.7rem 1.2rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  cursor: pointer;
  box-shadow: 0 0 18px rgba(0, 195, 255, 0.12);
}

.reset-btn:hover {
  border-color: rgba(120, 245, 255, 0.8);
  box-shadow: 0 0 26px rgba(0, 195, 255, 0.28);
}
  </style>
</head>

<body>
  <div class="app">
    <h1>jarvis</h1>
    <div id="status" class="status">Tap the core to connect.</div>

    <div class="core-wrap">
  <button id="core" class="core">WAKE jarvis</button>
</div>

<div class="controls">
  <button id="resetBtn" class="reset-btn">RESET jarvis</button>
</div>

    <div class="hint">
      Tap once, allow microphone access, then say jarvis.<br>
      After jarvis wakes, you can speak normally.<br>
      Say bye to leave conversation mode, or jarvis sleep to stop.
    </div>

    <div id="transcript" class="transcript"></div>
  </div>

<script>
let active = false;
let busy = false;
let recognition = null;

const core = document.getElementById("core");
const statusEl = document.getElementById("status");

const resetBtn = document.getElementById("resetBtn");

function addBubble(role, text) {
  if (!text) return;

  const div = document.createElement("div");
  div.className = "bubble " + role;
  div.textContent = text;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function speak(text, onDone) {
  if (!text) {
    if (onDone) onDone();
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-GB";
  utterance.rate = 1.02;
  utterance.pitch = 1.0;

  utterance.onend = function() {
    if (onDone) onDone();
  };

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

async function sendTojarvis(text) {
  busy = true;
  setStatus("Thinking...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (data.reply) {
      addBubble("jarvis", data.reply);
      setStatus(data.state || "Connected");

      speak(data.reply, function() {
        busy = false;

        if (data.state === "sleeping") {
          stopjarvis();
          return;
        }

        if (active) {
          startListeningSoon();
        }
      });
    } else {
      busy = false;
      setStatus(data.state || "Listening...");

      if (active) {
        startListeningSoon();
      }
    }

  } catch (err) {
    busy = false;
    setStatus("Connection error. Check the laptop server.");
    addBubble("jarvis", "I could not connect to the jarvis server.");
  }
}

function startListeningSoon() {
  setTimeout(function() {
    if (active && !busy) {
      startListening();
    }
  }, 350);
}

function startListening() {
  if (!recognition || !active || busy) return;

  try {
    setStatus("Listening...");
    recognition.start();
  } catch (err) {
    // Recognition may already be running.
  }
}

function stopjarvis() {
  active = false;
  busy = false;
  core.classList.remove("active");
  core.textContent = "WAKE jarvis";
  setStatus("Sleeping.");

  if (recognition) {
    try {
      recognition.stop();
    } catch (err) {}
  }

  window.speechSynthesis.cancel();
}

function setupRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    setStatus("Speech recognition is not supported in this browser. Try Chrome.");
    return null;
  }

  const rec = new SpeechRecognition();
  rec.lang = "en-GB";
  rec.continuous = false;
  rec.interimResults = false;
  rec.maxAlternatives = 1;

  rec.onresult = function(event) {
    const text = event.results[0][0].transcript;
    addBubble("user", text);
    sendTojarvis(text);
  };

  rec.onerror = function(event) {
    busy = false;
    setStatus("Microphone error: " + event.error);

    if (active) {
      startListeningSoon();
    }
  };

  rec.onend = function() {
    if (active && !busy) {
      startListeningSoon();
    }
  };

  return rec;
}
resetBtn.addEventListener("click", async function() {
  active = false;
  busy = true;

  if (recognition) {
    try {
      recognition.stop();
    } catch (err) {}
  }

  window.speechSynthesis.cancel();

  core.classList.remove("active");
  core.textContent = "WAKE jarvis";
  setStatus("Resetting jarvis...");

  try {
    const response = await fetch("/reset", {
      method: "POST"
    });

    const data = await response.json();

    transcriptEl.innerHTML = "";

    if (data.reply) {
      addBubble("jarvis", data.reply);
    }

    setStatus("Reset complete. Tap the core to wake jarvis.");
  } catch (err) {
    setStatus("Reset failed. Check the laptop server.");
  }

  busy = false;
});
core.addEventListener("click", function() {
  if (active) {
    stopjarvis();
    return;
  }

  recognition = setupRecognition();

  if (!recognition) {
    return;
  }

  active = true;
  core.classList.add("active");
  core.textContent = "jarvis ONLINE";
  setStatus("Listening for jarvis...");
  startListeningSoon();
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    with assistant_lock:
        result = assistant.process_text_input(text)

    return jsonify(result)

@app.route("/reset", methods=["POST"])
def reset():
    with assistant_lock:
        assistant.in_conversation = False

        # Keep only the system prompt.
        if hasattr(assistant, "messages") and assistant.messages:
            assistant.messages = [assistant.messages[0]]

        # Clear previous tool/search context if you added these attributes.
        for attr in ["last_tool_result", "last_tool_name", "last_user_query"]:
            if hasattr(assistant, attr):
                setattr(assistant, attr, None)

    return jsonify({
        "ok": True,
        "reply": "jarvis has been reset.",
        "state": "sleeping",
        "ignored": False,
    })
def run_server() -> None:
    port = int(os.getenv("PHONE_SERVER_PORT", "7860"))
    print("")
    print("jarvis phone server is running.")
    print(f"Open this on your phone: http://YOUR_LAPTOP_IP:{port}")
    print("Find your laptop IP with: ipconfig")
    print("")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False,
    )


if __name__ == "__main__":
    run_server()