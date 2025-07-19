from __future__ import annotations

# Simplified Flask wrapper that merely streams frames produced by
# `webcam_feed.stream_frames_flask`.  All heavy lifting (YOLO + MediaPipe)
# lives in `webcam_feed.py`.

from flask import Flask, Response, render_template_string, jsonify
from fall_stream import stream_frames_flask, get_latest_stats

app = Flask(__name__)


@app.route("/")
def index():
    # Minimal HTML embedding the MJPEG stream
    return render_template_string(
        """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='utf-8'>
            <title>Fall Prediction System</title>
            <style>
                /* Reset */
                * { box-sizing: border-box; margin: 0; padding: 0; }

                body { font-family: Arial, sans-serif; background:#fff; }

                /* Top bar */
                .topbar {
                    height: 60px;
                    width: 100%;
                    background: #1976d2;
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
                    background: #f5f5f5;
                    border-right: 1px solid #e0e0e0;
                    padding: 20px 15px;
                }

                .sidebar nav a {
                    display: block;
                    padding: 10px 8px;
                    color: #333;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-bottom: 8px;
                }

                .sidebar nav a:hover {
                    background: #e0e0e0;
                }

                /* Right panel */
                .controls {
                    width: 200px;
                    background: #fafafa;
                    border-left: 1px solid #e0e0e0;
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
                    background: #1976d2;
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
                    color: #333;
                }

                .stats span.value {
                    font-weight: bold;
                    margin-left: 4px;
                }
                /* Center content */
                .content {
                    flex: 1;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background: #fff;
                }

                .content img {
                    max-width: 90%;
                    max-height: 90%;
                    border: 1px solid #ddd;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
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
                        <a href="#">Home</a>
                        <a href="#">Analytics</a>
                        <a href="#">Settings</a>
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
                    </div>
                </aside>
            </div>
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


if __name__ == "__main__":
    # Disable reloader to avoid spawning multiple camera instances
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False) 