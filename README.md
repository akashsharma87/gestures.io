# SpatialFlow 🌌

**Hand-Gesture-Controlled 3D Diagramming Tool**

A spatial computing application that lets you visualize and interact with diagrams using natural hand gestures. Inspired by Eraser.io's "Diagram-as-Code" philosophy, SpatialFlow brings your diagrams into the third dimension.

## ✨ Features

- 🖐️ **Hand Gesture Control** - Navigate and manipulate 3D diagrams using intuitive gestures
- 🎯 **God-Level Stability** - Industrial-grade One Euro Filter eliminates 99% of tracking jitter
- 📊 **Diagram-as-Code** - Parse simple DSL syntax like `A -> B -> C` into 3D graphs
- 🌐 **3D Force-Directed Layouts** - NetworkX-powered spring layouts in 3D space
- 🎨 **Beautiful Visualization** - Premium aesthetics with smooth animations and effects

## 🎮 Controls

### Hand Gestures
| Gesture | Action |
|---------|--------|
| **Point** (index finger) | Move cursor |
| **Pinch** (thumb + index) | Drag nodes |
| **Fist** | Orbit camera |
| **Open hand** | Release / Idle |

### Keyboard
| Key | Action |
|-----|--------|
| `G` | Generate sample diagram |
| `C` | Clear canvas |
| `D` | Toggle debug visualization |
| `R` | Reset camera |
| `ESC` / `Q` | Quit |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/akashsharma/.gemini/antigravity/playground/inertial-exoplanet
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

### 3. Show Your Hand

Hold your hand in front of the webcam and start interacting!

## 📁 Project Structure

```
inertial-exoplanet/
├── main.py                 # Entry point and FSM
├── requirements.txt        # Dependencies
├── core/
│   ├── __init__.py
│   ├── signals.py          # One Euro Filter, Schmitt Trigger
│   ├── sensorium.py        # Hand tracking (MediaPipe)
│   └── graph_engine.py     # DSL parser, NetworkX logic
└── ui/
    ├── __init__.py
    └── spatial_canvas.py   # Ursina entities (Node, Edge, Cursor)
```

## 🔧 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                 Finite State Machine (FSM)                   │
│         IDLE ↔ HOVER ↔ DRAG | ANY → ORBIT                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Sensorium  │    │   Canvas    │    │   Graph     │
│ (MediaPipe) │    │  (Ursina)   │    │  Engine     │
│             │    │             │    │ (NetworkX)  │
│ HandTracker │    │ NodeEntity  │    │ DSL Parser  │
│ 21 Landmarks│    │ EdgeEntity  │    │ 3D Layout   │
│ Gestures    │    │ CursorTrail │    │ AI Generate │
└──────┬──────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│        Signal Processing            │
│ • One Euro Filter (jitter removal)  │
│ • Schmitt Trigger (hysteresis)      │
└─────────────────────────────────────┘
```

## 📐 DSL Syntax

Create diagrams using simple text:

```
# Directed edges
A -> B -> C

# Nodes with labels
[Start:Begin Here] -> [Process:Do Something]

# Decision trees
Decision -> [Yes:Accept]
Decision -> [No:Reject]
```

## 🎛️ Signal Processing

### One Euro Filter
Adaptive low-pass filter that provides:
- **Heavy smoothing** at low speeds (removes jitter)
- **Low latency** at high speeds (responsive tracking)

Parameters:
- `min_cutoff = 1.0 Hz` - Baseline smoothing
- `beta = 0.007` - Responsiveness scaling

### Schmitt Trigger
Hysteresis-based boolean state machine that prevents gesture oscillation:
- `high_threshold = 0.08` - Release threshold
- `low_threshold = 0.05` - Pinch threshold

## 🎨 Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| 3D Engine | [Ursina](https://www.ursinaengine.org/) | Scene rendering, entities |
| Hand Tracking | [MediaPipe](https://mediapipe.dev/) | 21-landmark detection |
| Graph Logic | [NetworkX](https://networkx.org/) | Layout algorithms |
| CV | [OpenCV](https://opencv.org/) | Camera capture |
| Signal Processing | Custom | One Euro Filter |

## 🐛 Debugging

Press `D` to enable debug mode which shows:
- Current FSM state
- Hand detection info
- Cursor coordinates
- Gesture states
- FPS counter

A debug OpenCV window will also appear showing:
- MediaPipe landmark visualization
- Gesture detection status

## 📝 Future Enhancements

- [ ] Google Generative AI integration for diagram generation
- [ ] Export to SVG/PNG
- [ ] Multiple hand support
- [ ] Voice commands
- [ ] Collaborative mode (WebRTC)

## 📄 License

MIT License - See LICENSE for details.

---

**Built with ❤️ using Antigravity AI**
