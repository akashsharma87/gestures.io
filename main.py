"""
SpatialFlow - Hand-Gesture-Controlled 3D Diagramming Tool
==========================================================

A spatial computing application combining:
- MediaPipe for hand tracking
- Ursina Engine for 3D rendering
- NetworkX for graph logic
- Industrial-grade signal processing for rock-solid stability

Controls:
    - Point: Move cursor (index finger tip)
    - Pinch (Thumb + Index): Drag nodes
    - Fist: Orbit camera around the diagram
    - Open Hand: Release/Idle

Keyboard:
    - G: Generate sample diagram
    - C: Clear canvas
    - D: Toggle debug visualization
    - R: Reset camera
    - Q/ESC: Quit

Author: SpatialFlow Team
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load Panda3D config BEFORE importing Ursina
# This forces basic OpenGL rendering without shaders
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Config.prc')
if os.path.exists(config_path):
    from panda3d.core import loadPrcFile
    loadPrcFile(config_path)
    print(f"[SpatialFlow] Loaded Panda3D config: {config_path}")

from ursina import *
from enum import Enum, auto
from typing import Optional

from core.sensorium import HandTracker
from core.graph_engine import DiagramGraph, parse_dsl, generate_from_prompt, apply_palette
from ui.spatial_canvas import SpatialCanvas, Colors, NodeEntity


# ==============================================================================
# State Machine Definition
# ==============================================================================

class AppState(Enum):
    """Application interaction states"""
    IDLE = auto()       # Hand visible, cursor following
    HOVER = auto()      # Cursor over a node
    DRAG = auto()       # Pinching and dragging a node
    ORBIT = auto()      # Fist gesture, orbiting camera
    NO_HAND = auto()    # No hand detected


class StateMachine:
    """
    Finite State Machine for gesture-based interaction.
    
    State Transitions:
        NO_HAND ──(hand detected)──▶ IDLE
        IDLE ──(hover node)──▶ HOVER
        HOVER ──(pinch)──▶ DRAG
        HOVER ──(leave node)──▶ IDLE
        DRAG ──(release)──▶ HOVER
        ANY ──(fist)──▶ ORBIT
        ORBIT ──(open hand)──▶ IDLE
    """
    
    def __init__(self):
        self.state = AppState.NO_HAND
        self.previous_state = AppState.NO_HAND
        
        # Drag state
        self.dragged_node: Optional[NodeEntity] = None
        self.drag_offset = Vec3(0, 0, 0)
        
        # Orbit state
        self.orbit_start_rotation = Vec3(0, 0, 0)
        self.orbit_start_hand_pos = (0.0, 0.0)
        
        # Transition callbacks
        self.on_state_change: Optional[callable] = None
    
    def transition_to(self, new_state: AppState):
        """
        Transition to a new state.
        
        Args:
            new_state: The target state
        """
        if new_state == self.state:
            return
        
        self.previous_state = self.state
        self.state = new_state
        
        print(f"[FSM] {self.previous_state.name} → {new_state.name}")
        
        if self.on_state_change:
            self.on_state_change(self.previous_state, new_state)


# ==============================================================================
# Main Application Class
# ==============================================================================

class SpatialFlowApp:
    """
    Main application class integrating all components.
    """
    
    def __init__(self):
        # Initialize Ursina
        self.app = Ursina(
            title='SpatialFlow',
            borderless=False,
            fullscreen=False,
            development_mode=True,
            vsync=True
        )
        
        # Set up window
        window.color = Colors.BACKGROUND
        window.fps_counter.enabled = True
        
        # Set up camera with pivot for orbiting
        self.pivot = Entity()
        camera.parent = self.pivot
        camera.position = (0, 0, -20)
        camera.look_at(self.pivot)
        
        # Hand tracker
        self.tracker: Optional[HandTracker] = None
        self.tracking_enabled = True
        
        # Spatial canvas
        self.canvas = SpatialCanvas()
        self.canvas.cursor.visible = False  # Hide until hand detected
        
        # State machine
        self.fsm = StateMachine()
        self.fsm.on_state_change = self._on_state_change
        
        # Debug mode
        self.debug_mode = False
        self.debug_text = Text(
            text='',
            position=(-0.85, 0.45),
            scale=1.2,
            color=color.white
        )
        
        # Instructions
        self.instructions = Text(
            text='[G] Generate | [C] Clear | [D] Debug | [R] Reset | [ESC] Quit',
            position=(0, -0.45),
            origin=(0, 0),
            scale=1.0,
            color=color.rgba(1, 1, 1, 0.5)
        )
        
        # Status display
        self.status_text = Text(
            text='Initializing hand tracking...',
            position=(0, 0.4),
            origin=(0, 0),
            scale=1.5,
            color=color.white
        )
        
        # Lighting
        self._setup_lighting()
        
        # Load initial diagram
        self._load_sample_diagram()
    
    def _setup_lighting(self):
        """Set up scene - no lighting needed with unlit entities on macOS"""
        # Background is set via window.color in __init__
        # We don't create a Sky because shaders don't work on this macOS version
        pass
    
    def _load_sample_diagram(self):
        """Load a sample diagram on startup"""
        dsl = """
        [Start:Begin] -> [Input:User Input]
        Input -> [Validate:Validate Data]
        Validate -> [Process:Process Request]
        Validate -> [Error:Handle Error]
        Process -> [Output:Return Result]
        Error -> Input
        """
        
        graph = parse_dsl(dsl)
        apply_palette(graph, "cool")
        self.canvas.load_from_graph(graph)
        
        self.status_text.text = 'Sample diagram loaded'
    
    def start_tracking(self) -> bool:
        """
        Initialize and start the hand tracker.
        
        Returns:
            True if successful
        """
        try:
            # Find model path relative to main.py
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'hand_landmarker.task'
            )
            
            self.tracker = HandTracker(
                camera_index=0,
                model_path=model_path,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                filter_min_cutoff=1.0,
                filter_beta=0.2  # Optimized for balance between stability and responsiveness
            )
            
            if self.tracker.start():
                self.status_text.text = 'Hand tracking active - show your hand!'
                self.canvas.cursor.visible = True
                print("[SpatialFlow] Hand tracking started successfully")
                return True
            else:
                self.status_text.text = 'Failed to start camera'
                print("[SpatialFlow] Failed to start hand tracking")
                return False
                
        except FileNotFoundError as e:
            self.status_text.text = 'Model file missing - check console'
            print(f"[SpatialFlow] {e}")
            return False
        except Exception as e:
            self.status_text.text = f'Tracking error: {str(e)[:30]}'
            print(f"[SpatialFlow] Error starting tracker: {e}")
            return False
    
    def stop_tracking(self):
        """Stop the hand tracker"""
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
    
    def _on_state_change(self, old_state: AppState, new_state: AppState):
        """Handle state transitions"""
        # Update status display
        state_names = {
            AppState.NO_HAND: "No hand detected",
            AppState.IDLE: "Idle - Point to navigate",
            AppState.HOVER: "Hovering - Pinch to drag",
            AppState.DRAG: "Dragging node",
            AppState.ORBIT: "Orbiting camera"
        }
        self.status_text.text = state_names.get(new_state, "")
        
        # Handle drag end
        if old_state == AppState.DRAG and self.fsm.dragged_node:
            self.fsm.dragged_node.set_dragging(False)
            self.fsm.dragged_node = None
        
        # Handle orbit start
        if new_state == AppState.ORBIT:
            self.fsm.orbit_start_rotation = Vec3(self.pivot.rotation)
            if self.tracker and self.tracker.hand_data.detected:
                palm = self.tracker.hand_data.palm_center
                self.fsm.orbit_start_hand_pos = (palm[0], palm[1])
    
    def update(self):
        """Main update loop - called every frame by Ursina"""
        if not self.tracker or not self.tracking_enabled:
            return
        
        # Update hand tracking
        hand_data = self.tracker.update()
        
        # Update debug info
        if self.debug_mode:
            self._update_debug_display(hand_data)
        
        # State machine logic
        self._process_state_machine(hand_data)
    
    def _process_state_machine(self, hand_data):
        """Process the state machine based on hand data"""
        
        # Handle no hand case
        if not hand_data.detected:
            self.fsm.transition_to(AppState.NO_HAND)
            self.canvas.cursor.visible = False
            return
        
        self.canvas.cursor.visible = True
        
        # Get cursor world position
        cursor_pos = self.tracker.get_cursor_world_position(z_depth=-10)
        self.canvas.update_cursor(cursor_pos)
        
        # Get palm position for orbit gesture
        palm_pos = self.tracker.get_palm_world_position(z_depth=-10)
        
        # Check for fist gesture (priority - overrides other states)
        if hand_data.is_fist:
            if self.fsm.state != AppState.ORBIT:
                self.fsm.transition_to(AppState.ORBIT)
            
            # Orbit the camera based on palm movement
            current_palm = (hand_data.palm_center[0], hand_data.palm_center[1])
            dx = (current_palm[0] - self.fsm.orbit_start_hand_pos[0]) * 200
            dy = (current_palm[1] - self.fsm.orbit_start_hand_pos[1]) * 200
            
            self.pivot.rotation_y = self.fsm.orbit_start_rotation.y + dx
            self.pivot.rotation_x = self.fsm.orbit_start_rotation.x - dy
            
            # Clamp vertical rotation
            self.pivot.rotation_x = clamp(self.pivot.rotation_x, -60, 60)
            return
        
        # Check for pinch gesture
        if hand_data.is_pinching:
            hovered_node = self.canvas.get_hovered_node()
            
            if self.fsm.state == AppState.DRAG:
                # Continue dragging
                if self.fsm.dragged_node:
                    # Move node to cursor position plus offset
                    new_pos = Vec3(*cursor_pos) + self.fsm.drag_offset
                    self.fsm.dragged_node.position = new_pos
                    
            elif hovered_node:
                # Start dragging
                self.fsm.transition_to(AppState.DRAG)
                self.fsm.dragged_node = hovered_node
                self.fsm.drag_offset = hovered_node.position - Vec3(*cursor_pos)
                hovered_node.set_dragging(True)
            
            return
        
        # Not pinching and not fist - check hover
        hovered_node = self.canvas.get_hovered_node()
        
        if hovered_node:
            self.fsm.transition_to(AppState.HOVER)
        else:
            self.fsm.transition_to(AppState.IDLE)
    
    def _update_debug_display(self, hand_data):
        """Update debug information display"""
        if not hand_data.detected:
            self.debug_text.text = "No hand detected"
            return
        
        cursor = self.tracker.get_cursor_world_position()
        
        self.debug_text.text = (
            f"State: {self.fsm.state.name}\n"
            f"Hand: {hand_data.handedness}\n"
            f"Cursor: ({cursor[0]:.2f}, {cursor[1]:.2f})\n"
            f"Pinch: {'YES' if hand_data.is_pinching else 'NO'}\n"
            f"Fist: {'YES' if hand_data.is_fist else 'NO'}\n"
            f"FPS: {1/max(time.dt, 0.001):.0f}"
        )
    
    def input(self, key):
        """Handle keyboard input"""
        if key == 'g':
            # Generate new diagram
            self._generate_diagram()
        
        elif key == 'c':
            # Clear canvas
            self.canvas.clear()
            self.status_text.text = 'Canvas cleared'
        
        elif key == 'd':
            # Toggle debug mode
            self.debug_mode = not self.debug_mode
            self.debug_text.visible = self.debug_mode
            
            if self.tracker:
                self.tracker.set_debug_visualization(self.debug_mode)
            
            self.status_text.text = f"Debug mode: {'ON' if self.debug_mode else 'OFF'}"
        
        elif key == 'r':
            # Reset camera
            self.pivot.rotation = Vec3(0, 0, 0)
            camera.position = (0, 0, -20)
            self.status_text.text = 'Camera reset'
        
        elif key == 'escape' or key == 'q':
            # Quit
            self._cleanup()
            application.quit()
        
        elif key == 't':
            # Toggle tracking
            self.tracking_enabled = not self.tracking_enabled
            self.status_text.text = f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}"
    
    def _generate_diagram(self):
        """Generate a new diagram using the mock AI function"""
        prompts = [
            "Create a login flow",
            "Design an API request flow",
            "Show a decision tree",
            "Build a process hierarchy"
        ]
        
        import random
        prompt = random.choice(prompts)
        
        self.status_text.text = f'Generating: {prompt}'
        
        dsl = generate_from_prompt(prompt)
        graph = parse_dsl(dsl)
        
        # Apply random palette
        palettes = ["default", "warm", "cool", "pastel"]
        apply_palette(graph, random.choice(palettes))
        
        self.canvas.load_from_graph(graph)
        self.status_text.text = f'Generated: {len(graph.nodes)} nodes'
    
    def _cleanup(self):
        """Clean up resources before exit"""
        self.stop_tracking()
        print("[SpatialFlow] Cleanup complete")
    
    def run(self):
        """
        Main entry point - start the application.
        """
        print("\n" + "=" * 60)
        print("  SpatialFlow - Hand-Gesture-Controlled 3D Diagramming")
        print("=" * 60)
        print("\nControls:")
        print("  • Point finger: Move cursor")
        print("  • Pinch (thumb + index): Drag nodes")
        print("  • Fist: Orbit camera")
        print("  • [G] Generate | [C] Clear | [D] Debug | [R] Reset | [ESC] Quit")
        print("\n" + "=" * 60 + "\n")
        
        # Start hand tracking
        self.start_tracking()
        
        # Set up Ursina update callback
        def update():
            self.update()
        
        # Make update and input accessible to Ursina
        import builtins
        builtins.update = update
        builtins.input = lambda key: self.input(key)
        
        try:
            self.app.run()
        finally:
            self._cleanup()


# ==============================================================================
# Application Entry Point
# ==============================================================================

# Create global app instance for Ursina callbacks
app_instance: Optional[SpatialFlowApp] = None


def update():
    """Global update function called by Ursina"""
    if app_instance:
        app_instance.update()


def input(key):
    """Global input function called by Ursina"""
    if app_instance:
        app_instance.input(key)


if __name__ == "__main__":
    # Create and run the application
    app_instance = SpatialFlowApp()
    
    # Override Ursina's global functions
    import builtins
    builtins.update = update
    builtins.input = input
    
    # Start hand tracking BEFORE running the app loop
    app_instance.start_tracking()
    
    try:
        app_instance.app.run()
    except KeyboardInterrupt:
        print("\n[SpatialFlow] Interrupted by user")
    finally:
        if app_instance:
            app_instance._cleanup()

