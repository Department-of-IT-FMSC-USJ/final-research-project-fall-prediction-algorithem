from __future__ import annotations

# Simplified Flask wrapper that merely streams frames produced by
# `webcam_feed.stream_frames_flask`.  All heavy lifting (YOLO + MediaPipe)
# lives in `webcam_feed.py`.
# python flask_app.py --port 6060

import argparse
from flask import Flask, Response, render_template_string, jsonify
from fall_stream import stream_frames_flask, get_latest_stats, get_metrics_history

app = Flask(__name__)

# ---------------- Runtime settings (simple in-memory store) ---------------- #

SETTINGS = {
    "camera_enabled": True,
    "share_analytics": True,
    "show_personal_data": False,
    "telegram_token": "",
    "telegram_chat_id": "",
}

# Helper
def _bool(x: str | bool | None) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).lower() in {"1", "true", "yes", "on"}


COMMON_STYLE = """
<style>
:root{
    --bg:#14131A;           /* deep graphite */
    --panel:#1F1E27;        /* dark indigo-grey */
    --accent:#6DEFB1;       /* mint-aqua pop */
    --text:#ECEFF4;
    --muted:#9CA0AD;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:Roboto,Arial,sans-serif;background:var(--bg);color:var(--text);}
.topbar{height:60px;background:var(--panel);display:flex;align-items:center;
        padding:0 24px;font-size:1.25rem;font-weight:500;box-shadow:0 2px 6px rgba(0,0,0,.2);}
.main{display:flex;height:calc(100vh - 60px);animation:fadeSlide .5s ease;}
@keyframes fadeSlide{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.sidebar{width:200px;background:var(--panel);border-right:1px solid #2B2A33;padding:24px 18px;}
.sidebar nav a{display:block;padding:10px 8px;color:var(--text);border-radius:8px;margin-bottom:10px;
              transition:background .25s,color .25s;}
.sidebar nav a:hover{background:var(--accent);color:var(--bg);}
.controls{width:220px;background:var(--panel);border-left:1px solid #2B2A33;padding:24px 18px;}
.content{flex:1;display:flex;justify-content:center;align-items:center;background:var(--bg);}
.card{background:var(--panel);border-radius:14px;padding:24px;margin-bottom:28px;
      box-shadow:0 4px 14px rgba(0,0,0,.35);transition:transform .25s ease;}
.card:hover{transform:translateY(-4px);}
.switch{position:relative;width:54px;height:26px;}
.switch input{opacity:0;width:0;height:0;}
.slider{position:absolute;top:0;left:0;right:0;bottom:0;background:#555;border-radius:26px;cursor:pointer;
        transition:.3s;}
.slider:before{content:\"\";position:absolute;height:20px;width:20px;left:3px;top:3px;
              background:#fff;border-radius:50%;transition:.3s;}
input:checked + .slider{background:var(--accent);}
input:checked + .slider:before{transform:translateX(28px);}
img#stream{max-width:88%;max-height:88%;border-radius:16px;box-shadow:0 4px 18px rgba(0,0,0,.45);
          transition:box-shadow .3s;}
img#stream:hover{box-shadow:0 6px 24px rgba(0,0,0,.6);}
.stats{font-size:.85rem;line-height:1.7;color:var(--muted);}
.stats .value{color:var(--accent);}
button, .controls input[type=text]{border:none;border-radius:10px;padding:10px 14px;font-weight:500;}
button{cursor:pointer;background:var(--accent);color:var(--bg);transition:background .25s;}
button.stop{background:#dc3545;color:#fff;}
button:hover{background:#56d79a;}
</style>
"""


@app.route("/")
def index():
    # Minimal HTML embedding the MJPEG stream
    return render_template_string(
        COMMON_STYLE + """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='utf-8'>
            <title>Fall Prediction System</title>
            <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E%3Ccircle cx='24' cy='24' r='22' fill='%23686a65'/%3E%3Cpath d='M12 30 24 14 36 30' stroke='%23ffffff' stroke-width='4' fill='none' stroke-linecap='round'/%3E%3C/svg%3E"/>
            <style>
                /* Reset */
                * { box-sizing: border-box; margin: 0; padding: 0; }

                body { font-family: Arial, sans-serif; background:#14131a; color:#ffffff; }

                /* Top bar */
                .topbar {
                    height: 60px;
                    width: 100%;
                    background: #6defb1;
                    color: #fff;
                    display: flex;
                    align-items: center;
                    padding: 0 20px;
                    font-size: 1.2rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                /* Layout */
                .main {
                    display: flex;
                    height: calc(100vh - 60px);
                    width: 100%;
                }

                /* Left sidebar */
                .sidebar {
                    width: 200px;
                    background: #1f1e27;
                    border-right: 1px solid #2b2a33;
                    padding: 20px 15px;
                }

                .sidebar nav a {
                    display: block;
                    padding: 10px 8px;
                    color: #ffffff;
                    text-decoration: none;
                    border-radius: 6px;
                    margin-bottom: 10px;
                    transition: background .25s, color .25s;
                }

                .sidebar nav a:hover {
                    background: #ffffff;
                    color: #1f1e27;
                }

                /* Right panel */
                .controls {
                    width: 200px;
                    background: #1f1e27;
                    border-left: 1px solid #2b2a33;
                    padding: 20px 15px;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }

                .controls button {
                    padding: 10px;
                    font-size: 0.9rem;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    background: #6defb1;
                    color:#fff;
                }

                .controls button.stop {
                    background: #c62828;
                }

                /* Stats table */
                .stats {
                    margin-top: 20px;
                    font-size: 0.85rem;
                    line-height: 1.6;
                    color: #ffffff;
                }

                .stats span.value {
                    font-weight: bold;
                    margin-left: 4px;
                }
                /* Center content */
                .content {
                    flex:1;display:flex;justify-content:center;align-items:center;background:#14131a;
                }

                .content img{
                    width:92%;height:auto;border:4px solid #6defb1;border-radius:16px;box-shadow:0 4px 16px rgba(0,0,0,.25);transition:box-shadow .3s,transform .3s;
                }
                .content img:hover{box-shadow:0 6px 20px rgba(0,0,0,.35);transform:scale(1.01);}

            </style>
            <script>
                function startFeed() {
                    const img = document.getElementById('stream');
                    if (!img.src || img.src.endsWith('#')) {
                        img.src = '/video_feed?' + Date.now(); // bust cache
                    }
                }

                function stopFeed() {
                    const img = document.getElementById('stream');
                    img.src = '#'; // setting to invalid source stops requests
                }

                // Fetch stats every second and update panel
                async function updateStats() {
                    try {
                        const res = await fetch('/stats');
                        const data = await res.json();
                        document.getElementById('fps-val').textContent = (data.fps !== null && data.fps !== undefined) ? data.fps.toFixed(1) : '-';
                        document.getElementById('trunk-val').textContent = (data.trunk_angle !== null && data.trunk_angle !== undefined) ? data.trunk_angle.toFixed(1) : '-';
                        document.getElementById('nsar-val').textContent = (data.nsar !== null && data.nsar !== undefined) ? data.nsar.toFixed(3) : '-';
                        document.getElementById('theta-u-val').textContent = (data.theta_u !== null && data.theta_u !== undefined) ? data.theta_u.toFixed(1) : '-';
                        document.getElementById('theta-d-val').textContent = (data.theta_d !== null && data.theta_d !== undefined) ? data.theta_d.toFixed(1) : '-';
                        document.getElementById('fall-val').textContent = data.fall_detected ? 'Yes' : 'No';
                        document.getElementById('pred-val').textContent = data.prediction ? data.prediction.slice(0,60) + (data.prediction.length>60?'…':'') : '-';
                    } catch(e) {
                        console.error('Failed to fetch stats', e);
                    }
                }

                setInterval(updateStats, 1000);
            </script>
        </head>
        <body>
            <div class="topbar">Fall Prediction Dashboard</div>
            <div class="main">
                <aside class="sidebar">
                    <nav>
                        <a href="/">Home</a>
                        <a href="analytics">Analytics</a>
                        <a href="settings">Settings</a>
                    </nav>
                </aside>

                <section class="content">
                    <img id="stream" src="/video_feed" alt="video stream">
                </section>

                <aside class="controls">
                    <button onclick="startFeed()">Start Predicting</button>
                    <button class="stop" onclick="stopFeed()">Stop</button>

                    <div class="stats">
                        <div>FPS: <span id="fps-val" class="value">-</span></div>
                        <div>Trunk Angle: <span id="trunk-val" class="value">-</span></div>
                        <div>NSAR: <span id="nsar-val" class="value">-</span></div>
                        <div>θu: <span id="theta-u-val" class="value">-</span></div>
                        <div>θd: <span id="theta-d-val" class="value">-</span></div>
                        <div>Fall Detected: <span id="fall-val" class="value">-</span></div>
                        <div>Prediction: <span id="pred-val" class="value">-</span></div>
                    </div>
                </aside>
            </div>
        </body>
        </html>
        """
    )


# ---------------- Analytics page ---------------- #


@app.route("/analytics")
def analytics():
    return render_template_string(
        COMMON_STYLE + """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='utf-8'>
            <title>Analytics – Fall Prediction</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
            <style>
                *{box-sizing:border-box;margin:0;padding:0;}
                body{font-family:Arial,sans-serif;background:#14131a;color:#ffffff;}
                .topbar{height:60px;width:100%;background:#6defb1;color:#fff;display:flex;align-items:center;padding:0 24px;font-size:1.25rem;font-weight:500;box-shadow:0 2px 6px rgba(0,0,0,.2);} 
                .main{display:flex;height:calc(100vh - 60px);width:100%;}
                .sidebar{width:200px;background:#1f1e27;border-right:1px solid #2b2a33;padding:24px 18px;}
                .sidebar nav a{display:block;padding:10px 8px;color:#ffffff;text-decoration:none;border-radius:8px;margin-bottom:10px;transition:background .25s,color .25s;}
                .sidebar nav a:hover{background:#ffffff;color:#1f1e27;}
                .controls{width:200px;background:#1f1e27;border-left:1px solid #2b2a33;padding:24px 18px;color:#ffffff;}
                .content{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;background:#14131a;}
                canvas{max-width:90%;max-height:80%;}
            </style>
        </head>
        <body>
            <div class="topbar">Fall Prediction Dashboard</div>
            <div class="main">
                <aside class="sidebar">
                    <nav>
                        <a href="/">Home</a>
                        <a href="/analytics">Analytics</a>
                        <a href="/settings">Settings</a>
                    </nav>
                </aside>

                <section class="content">
                    <h2 style="margin-bottom:10px;">Last 2 minutes (rolling)</h2>
                    <canvas id="analyticsChart"></canvas>
                </section>

                <aside class="controls">
                    <!-- Placeholder for future filters -->
                    <p>Charts auto-refresh every second.</p>
                </aside>
            </div>

            <script>
            const ctx = document.getElementById('analyticsChart').getContext('2d');

            const chart = new Chart(ctx, {
                type:'line',
                data:{labels:[],datasets:[
                    {label:'Trunk Angle',data:[],borderColor:'#6defb1',fill:false},
                    {label:'NSAR',data:[],borderColor:'#c62828',fill:false},
                    {label:'θu',data:[],borderColor:'#2e7d32',fill:false},
                    {label:'θd',data:[],borderColor:'#ff8f00',fill:false}
                ]},
                options:{
                    responsive:true,
                    scales:{
                        x:{display:false},
                        y:{beginAtZero:true}
                    }
                }
            });

            async function fetchTimeline(){
                const res = await fetch('/timeline');
                const data = await res.json();
                const slice = data.slice(-120); // last 120 s
                const labels = slice.map(pt=> new Date(pt.ts*1000).toLocaleTimeString());

                chart.data.labels = labels;
                chart.data.datasets[0].data = slice.map(pt=>pt.trunk_angle);
                chart.data.datasets[1].data = slice.map(pt=>pt.nsar);
                chart.data.datasets[2].data = slice.map(pt=>pt.theta_u);
                chart.data.datasets[3].data = slice.map(pt=>pt.theta_d);
                chart.update();
            }

            setInterval(fetchTimeline,1000);
            fetchTimeline();
            </script>
        </body>
        </html>
        """
    )


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(
        stream_frames_flask(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats() -> Response:
    """Return the latest per-frame statistics as JSON for AJAX polling."""
    return jsonify(get_latest_stats())


# ---------------- Settings API ---------------- #


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    from flask import request
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        for k, v in data.items():
            if k not in SETTINGS:
                continue

            # Boolean keys
            if k in {"camera_enabled", "share_analytics", "show_personal_data"}:
                SETTINGS[k] = _bool(v)
            else:
                SETTINGS[k] = str(v)
        return jsonify({"status": "ok", **SETTINGS})
    return jsonify(SETTINGS)


# ---------------- Settings page ---------------- #


@app.route("/settings")
def settings_page():
    return render_template_string(
        COMMON_STYLE + """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='utf-8'>
            <title>Settings – Fall Prediction</title>
            <style>
                *{box-sizing:border-box;margin:0;padding:0;}
                body{font-family:Arial,sans-serif;background:#fff;}
                .topbar{height:60px;width:100%;background:#6defb1;color:#fff;display:flex;align-items:center;padding:0 20px;font-size:1.2rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
                .main{display:flex;height:calc(100vh - 60px);width:100%;}
                .sidebar{width:200px;background:#1f1e27;border-right:1px solid #2b2a33;padding:24px 18px;}
                .sidebar nav a{display:block;padding:10px 8px;color:#ffffff;text-decoration:none;border-radius:8px;margin-bottom:10px;transition:background .25s,color .25s;}
                .sidebar nav a:hover{background:#ffffff;color:#1f1e27;}
                /* vertical cards on settings page */
                .content{flex:1;display:flex;flex-direction:column;align-items:center;gap:32px;padding:40px 30px; padding-top:30px;}

                .cards{display:flex;flex-direction:column;align-items:center;gap:10px;width:100%;}
                .card{width:500px;color:#ffffff; padding-top: 20px;}   /* wider cards with white text */
                .setting{margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;width:100%;}
                .card h3{margin-top:0;margin-bottom:14px;font-size:1.15rem;color:#6defb1;}
                .card input[type=text]{background:#2b2a33;border:1px solid #3c3c45;border-radius:8px;color:#ffffff;padding:10px;width:100%;box-sizing:border-box;transition:border-color .25s,box-shadow .25s;font-size:.9rem;}
                .card input[type=text]::placeholder{color:#6f6f78;}
                .card input[type=text]:focus{outline:none;border-color:#6defb1;box-shadow:0 0 0 2px rgba(109,239,177,.35);}  
                .setting span{color:#ffffff;font-size:.95rem;}
                .switch{position:relative;display:inline-block;width:50px;height:24px;}
                .switch input{opacity:0;width:0;height:0;}
                .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#ccc;transition:.4s;border-radius:24px;}
                .slider:before{position:absolute;content:"";height:18px;width:18px;left:3px;bottom:3px;background:#fff;transition:.4s;border-radius:50%;}
                input:checked + .slider{background:#6defb1;}
                input:checked + .slider:before{transform:translateX(26px);}            
            </style>
        </head>
        <body>
            <div class="topbar">Fall Prediction Dashboard</div>
            <div class="main">
                <aside class="sidebar">
                    <nav>
                        <a href="/">Home</a>
                        <a href="/analytics">Analytics</a>
                        <a href="/settings">Settings</a>
                    </nav>
                </aside>

                <section class="content">
                    <div class="cards" style="color:#ffffff;">
                        <!-- CAMERA CARD -->
                        <div class="card">
                            <h3>General Settings</h3>
                            <div class="setting">
                                <span>Enable Camera Stream</span>
                                <label class="switch">
                                    <input type="checkbox" id="camToggle">
                                    <span class="slider"></span>
                                </label>
                            </div>
                            <div class="setting">
                                <span>Share Analytics Data</span>
                                <label class="switch">
                                    <input type="checkbox" id="analyticsToggle">
                                    <span class="slider"></span>
                                </label>
                            </div>
                            <div class="setting">
                                <span>Show Personal Data</span>
                                <label class="switch">
                                    <input type="checkbox" id="personalToggle">
                                    <span class="slider"></span>
                                </label>
                            </div>
                        </div>

                        <!-- TELEGRAM CARD -->
                        <div class="card">
                            <h3>Telegram Configuratoins</h3><br/>
                            <div class="setting" style="flex-direction:column;align-items:flex-start;">
                                <label for="tokenInput" style="margin-bottom:6px;">Telegram API</label>
                                <input type="text" id="tokenInput" placeholder="123456:ABC…" style="width:100%;padding:6px;">
                            </div>
                            <div class="setting" style="flex-direction:column;align-items:flex-start;">
                                <label for="chatInput" style="margin-bottom:6px;">Chat ID</label>
                                <input type="text" id="chatInput" placeholder="987654321" style="width:100%;padding:6px;">
                            </div>
                            <div class="setting" style="flex-direction:column;align-items:flex-start;">
                                <label for="chatInput" style="margin-bottom:6px;">Gardian Phone Number</label>
                                <input type="text" id="chatInput" placeholder="987654321" style="width:100%;padding:6px;">
                            </div>
                            <small>Token, Phone Number & Chat ID are stored only in server memory. Video Data will be stored in your computer</small>
                        </div>
                    </div>

                    <p id="statusMsg" style="margin-top:20px;color:#1976d2;"></p>
                </section>
            </div>

            <script>
            // Load current settings
            async function loadSettings(){
                const res = await fetch('/api/settings');
                const s   = await res.json();
                document.getElementById('camToggle').checked       = s.camera_enabled;
                document.getElementById('analyticsToggle').checked = s.share_analytics;
                document.getElementById('personalToggle').checked  = s.show_personal_data;
                document.getElementById('tokenInput').value        = s.telegram_token || '';
                document.getElementById('chatInput').value         = s.telegram_chat_id || '';
            }

            async function saveSettings(){
                const body = {
                    camera_enabled:      document.getElementById('camToggle').checked,
                    share_analytics:     document.getElementById('analyticsToggle').checked,
                    show_personal_data:  document.getElementById('personalToggle').checked,
                    telegram_token:      document.getElementById('tokenInput').value.trim(),
                    telegram_chat_id:    document.getElementById('chatInput').value.trim(),
                };
                await fetch('/api/settings', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                const msg = document.getElementById('statusMsg');
                msg.textContent = 'Settings saved ✓';
                setTimeout(()=>msg.textContent='',2000);
            }

            // Attach listeners
            ['camToggle','analyticsToggle','personalToggle'].forEach(id=>{
                document.getElementById(id).addEventListener('change',saveSettings);
            });

            loadSettings();
            </script>
        </body>
        </html>
        """
    )


@app.route("/timeline")
def timeline() -> Response:
    """Return buffered metric history for charts."""
    return jsonify(get_metrics_history())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fall Prediction Flask dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind (default 5050)")
    args = parser.parse_args()

    try:
        # Disable reloader to avoid spawning multiple camera instances
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    except OSError as e:
        print("[ERROR] Failed to bind to port", args.port)
        print(" »", e)
        print("Hint: Another process may be using the port or your firewall/antivirus is blocking it.\n" "Try a different port, e.g.: python flask_app.py --port 6060") 