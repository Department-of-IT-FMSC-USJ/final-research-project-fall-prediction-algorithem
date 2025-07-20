# Fall-Detection v2  📹🩹

Real-time fall-risk monitoring that fuses YOLOv8 object detection, MediaPipe Pose biomechanics, and an Azure AI Foundry LLM for contextual risk assessment. When a fall is confirmed the system notifies a Telegram chat **and** asks the LLM to summarise risk factors.

---
## 1  Quick Start
```bash
# 1. Install Python 3.9+ then:
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Create a .env file (see template below)

# 3. Run single-camera mode (webcam index 0):
python webcam_feed.py

#  ‑ or ‑  multi-camera fusion stub
python webcam_feed.py --sources 0,1 --multi-view
```

### `.env` template
```ini
# Telegram
TELEGRAM_BOT_TOKEN = 123456:ABCdefGhIJKlmNoPQrstUvwxYZ
TELEGRAM_CHAT_ID   = 987654321

# Azure OpenAI / Foundry
AZURE_OPENAI_ENDPOINT   = https://<resource>.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT = gpt-4-1-nano-2025-04-14-ft-…
AZURE_OPENAI_API_KEY    = <key>
```

> 🛈 If Telegram vars are omitted you will only see on-screen alerts. If Azure vars are omitted no LLM prediction is requested.

---
## 2  Features
| ✔︎ | Component | Details |
|----|-----------|---------|
| 🎯 | YOLOv8-n  | 25 FPS+ person detection on CPU; faster on GPU |
| 🦴 | MediaPipe Pose | 20 essential key-points, configurable complexity (lite/full/heavy) |
| 📐 | Biomechanics | Trunk angle, NSAR, upper/lower plumb angles |
| 🚨 | Multi-stage rules | Warning → Fall Detected (trunk, NSAR, or plumb persistence) |
| 📲 | Telegram | Push notification with optional LLM explanation |
| 🧠 | Azure AI Foundry | Chat completion: “Predict fall risk” based on metrics |
| 📊 | HUD & metrics | Full-screen overlay, FPS, miss-rate, latency, running chart buffer |

---
## 3  Thresholds & Tuning
Edit `webcam_feed.py` (top constants) or patch via PR:

| Variable | Meaning | Default |
|----------|---------|---------|
| `TRUNK_ANGLE_FALL_THRESHOLD_DEG` | Shoulder-hip tilt beyond this ⇒ immediate fall | 33.8° |
| `NSAR_DROP_RATIO` (implicit) | NSAR < 0.7 × baseline ⇒ fall | 0.70 |
| `PLUMB_FALL_THRESHOLD_FRAMES` | Frames (≈ 2 s) both θᵤ>45° **and** θ_d>60° | 60 |
| `SMOOTH_WINDOW` | Moving-average window for angle/NSAR | 8 frames |

Lower the angles or frame counts to trigger **earlier**, raise to be stricter.

---
## 4  Command-Line Flags
```
--sources "0,rtsp://cam/stream"  Comma-list of webcams/URLs (default "0")
--multi-view                    Enable multi-camera fusion (stub selects cam 0)
--pose-complexity {0,1,2}       0=lite 1=full 2=heavy (default 1)
```

---
## 5  File Overview
```
webcam_feed.py             # main realtime pipeline
azure_foundry_predict.py   # LLM helper (importable & CLI)
telegram_sender.py         # sendMessage helper and CLI tool
metrics_json/, metrics_csv/  # sample output folders (not auto-generated)
```

---
## 6  Internal Flow
1. Capture frame(s) → YOLOv8 → person bounding boxes.
2. Enlarge bbox (25 %) → MediaPipe Pose → key-points.
3. Compute metrics; update deques for smoothing & baseline.
4. Evaluate three rule families (trunk, NSAR, plumb) → `status`.
5. On **transition** to *Fall Detected*:
   * Build prompt `Metrics: trunk_angle=… nsar=… theta_d=… Predict fall risk.`
   * Call Azure LLM → `foundry_response`.
   * Telegram message `Fall detected!\n\nPrediction:\n<response>`.
6. Draw HUD & skeleton; repeat until `q` pressed.

A full technical deep-dive is available in `docs/DETAILS.md` (or see the chat explanation).

---
## 7  Troubleshooting
* **Black window / slow FPS** → lower `--pose-complexity` or install CUDA.
* **No fall detected** → lower thresholds (section 3).
* **`ModuleNotFoundError: openai`** → `pip install -r requirements.txt`.
* **Telegram not sending** → verify bot token, chat ID, and that you started the chat.
* **Azure 401/403** → make sure deployment name matches, quota not exhausted.

---
## 8  License & Attribution
MIT for project code. YOLOv8 weights © Ultralytics. MediaPipe © Google.  
Icons: [Twemoji](https://twemoji.twitter.com/) (CC-BY-4.0).

Happy hacking & stay safe! 🚀