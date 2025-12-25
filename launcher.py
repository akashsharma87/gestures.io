import os
import subprocess
import sys
import webbrowser
from threading import Timer
from flask import Flask, render_template, jsonify

# Use absolute paths for Vercel environment
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(base_dir, 'web_launcher', 'templates')
static_dir = os.path.join(base_dir, 'web_launcher', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Global process reference
main_app_process = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/launch_app', methods=['POST'])
def launch_app():
    global main_app_process
    if main_app_process is None or main_app_process.poll() is not None:
        try:
            # Launch the main OpenCV/Pygame app in a separate process
            # strictly using the current venv python
            python_exe = sys.executable
            main_app_process = subprocess.Popen([python_exe, "main_cv.py"])
            return jsonify({"status": "launched", "message": "SpatialFlow Studio is starting..."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "running", "message": "App is already running!"})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Schedule browser open after short delay
    Timer(1.0, open_browser).start()
    print("SpatialFlow Launcher running on http://127.0.0.1:5000")
    app.run(port=5000)
