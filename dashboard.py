"""Streamlit shell for the zero-rerun DriveGuard live dashboard.

The visible dashboard is a fixed HTML document. JavaScript polls a tiny local
state endpoint and updates existing DOM nodes in place, so new detector values
never trigger a Streamlit rerun or component replacement.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "outputs" / "state.json"
HTML_FILE = ROOT / "dashboard_live.html"


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard and its live JSON state on localhost only."""

    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def send_payload(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            try:
                payload = HTML_FILE.read_bytes()
            except OSError:
                payload = b"Dashboard template not found."
                self.send_payload(payload, "text/plain; charset=utf-8", 500)
                return
            self.send_payload(payload, "text/html; charset=utf-8")
            return

        if path == "/state":
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                state = {"available": False, "updated_at": 0}
            payload = json.dumps(state, separators=(",", ":")).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return

        self.send_payload(b"Not found", "text/plain; charset=utf-8", 404)


@st.cache_resource
def start_dashboard_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


st.set_page_config(
    page_title="DriveGuard | Live Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stSidebar"], [data-testid="stDecoration"] { display: none !important; }
    html, body, [data-testid="stAppViewContainer"], .stApp { background: #070b14 !important; }
    .block-container { padding: 0 !important; max-width: none !important; }
    iframe { border: 0 !important; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

server = start_dashboard_server()
port = server.server_address[1]
st.iframe(f"http://127.0.0.1:{port}/", width="stretch", height=1180)
