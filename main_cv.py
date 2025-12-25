"""
SpatialFlow Design Edition - Hand-Gesture Shape Builder
=========================================================

Create diagrams by selecting and placing shapes using hand gestures!

Gestures:
    - Point: Move cursor
    - Pinch (Thumb + Index): Select shape from palette / Place shape / Drag shape
    - Fist: Pan the canvas
    - Peace Sign (Index + Middle): Connect two nodes with an edge
    
Keyboard:
    - DELETE/BACKSPACE: Delete selected node
    - C: Clear canvas
    - D: Toggle debug
    - U: Undo last action
    - S: Save diagram
    - ESC/Q: Quit
"""

import pygame
import cv2
import numpy as np
import sys
import os
import math
import json
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sensorium import HandTracker


# ==============================================================================
# Window Configuration
# ==============================================================================

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
PALETTE_WIDTH = 200 # Widened from 120 for easier access
CANVAS_OFFSET_X = PALETTE_WIDTH


# Colors (RGB)
COLORS = {
    'background': (25, 28, 35),
    'canvas': (35, 38, 48),
    'palette_bg': (20, 22, 28),
    'palette_item': (50, 55, 65),
    'palette_hover': (70, 80, 100),
    'palette_selected': (74, 144, 217),
    'grid': (45, 48, 58),
    'node_default': (74, 144, 217),
    'node_hover': (255, 200, 50),
    'node_selected': (100, 220, 100),
    'node_drag': (50, 205, 50),
    'edge': (120, 130, 160),
    'edge_pending': (255, 200, 100),
    'cursor': (255, 100, 100),
    'cursor_glow': (255, 150, 150),
    'text': (255, 255, 255),
    'text_dim': (150, 150, 160),
    'status': (200, 200, 200),
    'success': (100, 220, 100),
    'warning': (255, 200, 50),
}

# Shape types available in palette
# Shape types and Tools available in palette
SHAPE_TYPES = [
    # Tools
    {'id': 'tool_connect', 'name': 'Connect', 'icon': 'link', 'color': (200, 200, 200)},
    {'id': 'tool_trash', 'name': 'Delete', 'icon': 'trash', 'color': (255, 100, 100)},
    {'id': 'tool_clear', 'name': 'Clear All', 'icon': 'x', 'color': (255, 50, 50)},
    
    # Shapes
    {'id': 'rectangle', 'name': 'Rectangle', 'icon': 'rect', 'color': (74, 144, 217)},
    {'id': 'rounded_rect', 'name': 'Rounded', 'icon': 'rrect', 'color': (123, 104, 238)},
    {'id': 'diamond', 'name': 'Decision', 'icon': 'diamond', 'color': (255, 165, 0)},
    {'id': 'circle', 'name': 'Circle', 'icon': 'circle', 'color': (50, 205, 50)},
    {'id': 'hexagon', 'name': 'Process', 'icon': 'hex', 'color': (255, 107, 107)},
    {'id': 'parallelogram', 'name': 'I/O', 'icon': 'para', 'color': (64, 224, 208)},
    {'id': 'cylinder', 'name': 'Database', 'icon': 'cyl', 'color': (186, 85, 211)},
    {'id': 'cloud', 'name': 'Cloud', 'icon': 'cloud', 'color': (135, 206, 250)},
]



# ==============================================================================
# State Machine
# ==============================================================================

class AppState(Enum):
    IDLE = auto()
    PALETTE_HOVER = auto()
    PLACING_SHAPE = auto()
    NODE_HOVER = auto()
    DRAGGING = auto()
    RESIZING_NEW_SHAPE = auto()       # New state for draw-to-size
    CONNECTING_START = auto()

    CONNECTING_END = auto()
    PAN = auto()
    EDITING_TEXT = auto()
    NO_HAND = auto()



class Tool(Enum):
    SELECT = auto()
    PLACE = auto()
    CONNECT = auto()
    DELETE = auto()


# ==============================================================================
# Node and Edge Classes
# ==============================================================================

@dataclass
class Node:
    id: str
    shape_type: str
    label: str
    x: float
    y: float
    width: int = 100
    height: int = 60
    color: Tuple[int, int, int] = (74, 144, 217)
    is_hovered: bool = False
    is_selected: bool = False
    is_dragging: bool = False
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return (left, top, right, bottom)"""
        return (
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.x + self.width // 2,
            self.y + self.height // 2
        )
    
    def contains(self, px: float, py: float) -> bool:
        left, top, right, bottom = self.get_bounds()
        return left <= px <= right and top <= py <= bottom
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        # Determine color based on state
        if self.is_dragging:
            fill_color = COLORS['node_drag']
        elif self.is_selected:
            fill_color = COLORS['node_selected']
        elif self.is_hovered:
            fill_color = COLORS['node_hover']
        else:
            fill_color = self.color
        
        left, top, right, bottom = self.get_bounds()
        rect = pygame.Rect(left, top, self.width, self.height)
        
        # Draw glow effect for selected/hovered
        if self.is_hovered or self.is_selected or self.is_dragging:
            glow_rect = rect.inflate(10, 10)
            glow_color = (*fill_color, 80)
            pygame.draw.rect(surface, fill_color, glow_rect, border_radius=12)
        
        # Draw shape based on type
        if self.shape_type == 'circle':
            radius = min(self.width, self.height) // 2
            pygame.draw.circle(surface, fill_color, (int(self.x), int(self.y)), radius)
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), radius, 2)
            
        elif self.shape_type == 'diamond':
            points = [
                (self.x, top),
                (right, self.y),
                (self.x, bottom),
                (left, self.y)
            ]
            pygame.draw.polygon(surface, fill_color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 2)
            
        elif self.shape_type == 'hexagon':
            w, h = self.width // 2, self.height // 2
            inset = w // 4
            points = [
                (left + inset, top),
                (right - inset, top),
                (right, self.y),
                (right - inset, bottom),
                (left + inset, bottom),
                (left, self.y)
            ]
            pygame.draw.polygon(surface, fill_color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 2)
            
        elif self.shape_type == 'parallelogram':
            skew = self.width // 5
            points = [
                (left + skew, top),
                (right, top),
                (right - skew, bottom),
                (left, bottom)
            ]
            pygame.draw.polygon(surface, fill_color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 2)
            
        elif self.shape_type == 'cylinder':
            # Draw cylinder body
            body_rect = pygame.Rect(left, top + 10, self.width, self.height - 20)
            pygame.draw.rect(surface, fill_color, body_rect)
            # Top ellipse
            pygame.draw.ellipse(surface, fill_color, pygame.Rect(left, top, self.width, 20))
            pygame.draw.ellipse(surface, (255, 255, 255), pygame.Rect(left, top, self.width, 20), 2)
            # Bottom ellipse
            pygame.draw.ellipse(surface, fill_color, pygame.Rect(left, bottom - 20, self.width, 20))
            pygame.draw.ellipse(surface, (255, 255, 255), pygame.Rect(left, bottom - 20, self.width, 20), 2)
            # Side lines
            pygame.draw.line(surface, (255, 255, 255), (left, top + 10), (left, bottom - 10), 2)
            pygame.draw.line(surface, (255, 255, 255), (right, top + 10), (right, bottom - 10), 2)
            
        elif self.shape_type == 'cloud':
            # Simplified cloud shape using circles
            r = self.height // 3
            positions = [
                (self.x - self.width//4, self.y),
                (self.x, self.y - r//2),
                (self.x + self.width//4, self.y),
                (self.x, self.y + r//3)
            ]
            for pos in positions:
                pygame.draw.circle(surface, fill_color, (int(pos[0]), int(pos[1])), r)
            for pos in positions:
                pygame.draw.circle(surface, (255, 255, 255), (int(pos[0]), int(pos[1])), r, 2)
                
        elif self.shape_type == 'rounded_rect':
            pygame.draw.rect(surface, fill_color, rect, border_radius=15)
            pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=15)
            
        else:  # rectangle
            pygame.draw.rect(surface, fill_color, rect)
            pygame.draw.rect(surface, (255, 255, 255), rect, 2)
        
        # Draw label
        text_surf = font.render(self.label, True, COLORS['text'])
        text_rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(text_surf, text_rect)
    
    def get_connection_point(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Get the edge connection point facing the target"""
        dx = target_x - self.x
        dy = target_y - self.y
        
        if abs(dx) < 1 and abs(dy) < 1:
            return (self.x + self.width // 2, self.y)
        
        # Calculate angle to target
        angle = math.atan2(dy, dx)
        
        # Simple box intersection
        half_w = self.width // 2
        half_h = self.height // 2
        
        # Determine which side to use
        if abs(dx) * half_h > abs(dy) * half_w:
            # Left or right
            if dx > 0:
                return (self.x + half_w, self.y)
            else:
                return (self.x - half_w, self.y)
        else:
            # Top or bottom
            if dy > 0:
                return (self.x, self.y + half_h)
            else:
                return (self.x, self.y - half_h)


@dataclass
class Edge:
    id: str
    source_id: str
    target_id: str
    color: Tuple[int, int, int] = (120, 130, 160)
    is_selected: bool = False
    
    def draw(self, surface: pygame.Surface, nodes: Dict[str, Node]):
        if self.source_id not in nodes or self.target_id not in nodes:
            return
        
        source = nodes[self.source_id]
        target = nodes[self.target_id]
        
        # Get connection points
        start = source.get_connection_point(target.x, target.y)
        end = target.get_connection_point(source.x, source.y)
        
        edge_color = COLORS['node_selected'] if self.is_selected else self.color
        
        # Draw line
        pygame.draw.line(surface, edge_color, 
                        (int(start[0]), int(start[1])), 
                        (int(end[0]), int(end[1])), 3)
        
        # Draw arrow head
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist > 1:
            dx /= dist
            dy /= dist
            
            arrow_size = 12
            angle = math.atan2(dy, dx)
            arrow_angle = math.pi / 6
            
            left_x = end[0] - arrow_size * math.cos(angle - arrow_angle)
            left_y = end[1] - arrow_size * math.sin(angle - arrow_angle)
            right_x = end[0] - arrow_size * math.cos(angle + arrow_angle)
            right_y = end[1] - arrow_size * math.sin(angle + arrow_angle)
            
            pygame.draw.polygon(surface, edge_color, [
                (int(end[0]), int(end[1])),
                (int(left_x), int(left_y)),
                (int(right_x), int(right_y))
            ])


# ==============================================================================
# Palette Item
# ==============================================================================

@dataclass
class PaletteItem:
    shape_info: dict

    x: int
    y: int
    width: int = 160  # Widen to fill most of the 200px palette
    height: int = 80  # Slightly taller
    is_hovered: bool = False
    is_selected: bool = False
    
    def contains(self, px: float, py: float) -> bool:
        # Generous hit box
        return (self.x - 10 <= px <= self.x + self.width + 10 and 
                self.y - 10 <= py <= self.y + self.height + 10)

    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        if self.is_selected:
            bg_color = COLORS['palette_selected']
            pygame.draw.rect(surface, bg_color, rect, border_radius=8)
            pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=8)
        elif self.is_hovered:
            bg_color = COLORS['palette_hover']
            pygame.draw.rect(surface, bg_color, rect, border_radius=8)
            # Thick 'Target Locked' border
            pygame.draw.rect(surface, (255, 215, 0), rect, 3, border_radius=8)
        else:
            bg_color = COLORS['palette_item']
            pygame.draw.rect(surface, bg_color, rect, border_radius=8)

        
        # Draw mini shape preview
        cx = self.x + self.width // 2
        cy = self.y + 25
        shape_color = self.shape_info['color']
        
        shape_type = self.shape_info['id']
        if shape_type == 'circle':
            pygame.draw.circle(surface, shape_color, (cx, cy), 15)
        elif shape_type == 'diamond':
            points = [(cx, cy-15), (cx+15, cy), (cx, cy+15), (cx-15, cy)]
            pygame.draw.polygon(surface, shape_color, points)
        elif shape_type == 'hexagon':
            points = [(cx-10, cy-12), (cx+10, cy-12), (cx+15, cy), 
                     (cx+10, cy+12), (cx-10, cy+12), (cx-15, cy)]
            pygame.draw.polygon(surface, shape_color, points)
        else:
            pygame.draw.rect(surface, shape_color, (cx-20, cy-12, 40, 24), border_radius=4)
        
        # Draw label
        label = self.shape_info['name']
        text_surf = font.render(label, True, COLORS['text_dim'])
        text_rect = text_surf.get_rect(center=(cx, self.y + self.height - 15))
        surface.blit(text_surf, text_rect)


# ==============================================================================
# Main Application
# ==============================================================================

class SpatialFlowDesign:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("SpatialFlow Design - Hand Gesture Shape Builder")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        
        # Hand tracking
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        self.tracker = HandTracker(
            camera_index=0,
            model_path=model_path,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            filter_min_cutoff=1.0,
            filter_beta=0.2
        )
        
        # State
        self.state = AppState.NO_HAND
        self.current_tool = Tool.SELECT
        self.selected_shape_type: Optional[str] = None
        
        # Diagram elements
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.node_counter = 0
        self.edge_counter = 0
        
        # Palette items
        self.palette_items: List[PaletteItem] = []
        self._setup_palette()
        
        # Cursor
        self.cursor_x = WINDOW_WIDTH // 2
        self.cursor_y = WINDOW_HEIGHT // 2
        
        # Interaction state
        self.hovered_node: Optional[Node] = None
        self.selected_node: Optional[Node] = None
        self.dragged_node: Optional[Node] = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        # Connection state
        self.connection_source: Optional[Node] = None
        
        # Pan state
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Resize state
        self.resize_start_x = 0
        self.resize_start_y = 0
        self.active_sizing_node: Optional[Node] = None
        
        # Pinch state tracking

        self.was_pinching = False
        self.pinch_handled = False
        
        # Undo history
        self.history: List[dict] = []
        
        # Debug and status
        self.debug_mode = False
        self.status_message = "Select a shape from the palette, then pinch to place!"
        self.status_time = 0
        
        self.running = True
    
    def _setup_palette(self):
        """Create palette items"""
        y_offset = 80
        for i, shape_info in enumerate(SHAPE_TYPES):
            item = PaletteItem(
                shape_info=shape_info,
                x=20, # Centered (200 - 160) / 2
                y=y_offset + i * 90 # Increased spacing
            )
            self.palette_items.append(item)

    
    def _show_status(self, message: str, duration: float = 3.0):
        """Display a status message"""
        self.status_message = message
        self.status_time = pygame.time.get_ticks() + duration * 1000
    
    def _create_node(self, shape_type: str, x: float, y: float) -> Node:
        """Create a new node"""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        # Find shape info
        shape_info = next((s for s in SHAPE_TYPES if s['id'] == shape_type), SHAPE_TYPES[0])
        
        node = Node(
            id=node_id,
            shape_type=shape_type,
            label=f"{shape_info['name']}",
            x=x,
            y=y,
            color=shape_info['color']
        )
        
        self.nodes[node_id] = node
        self._save_history()
        return node
    
    def _create_edge(self, source: Node, target: Node) -> Edge:
        """Create a new edge between two nodes"""
        self.edge_counter += 1
        edge_id = f"edge_{self.edge_counter}"
        
        edge = Edge(
            id=edge_id,
            source_id=source.id,
            target_id=target.id
        )
        
        self.edges[edge_id] = edge
        self._save_history()
        return edge
    
    def _delete_node(self, node: Node):
        """Delete a node and its connected edges"""
        # Remove connected edges
        edges_to_remove = [
            eid for eid, e in self.edges.items()
            if e.source_id == node.id or e.target_id == node.id
        ]
        for eid in edges_to_remove:
            del self.edges[eid]
        
        # Remove node
        if node.id in self.nodes:
            del self.nodes[node.id]
        
        self._save_history()
        self._show_status(f"Deleted {node.label}")
    
    def _save_history(self):
        """Save current state to history"""
        state = {
            'nodes': {nid: {'shape_type': n.shape_type, 'label': n.label, 
                          'x': n.x, 'y': n.y, 'color': n.color}
                     for nid, n in self.nodes.items()},
            'edges': {eid: {'source_id': e.source_id, 'target_id': e.target_id}
                     for eid, e in self.edges.items()}
        }
        self.history.append(state)
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
    
    def _undo(self):
        """Undo last action"""
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            prev_state = self.history[-1]
            
            # Restore nodes
            self.nodes.clear()
            for nid, data in prev_state['nodes'].items():
                self.nodes[nid] = Node(
                    id=nid,
                    shape_type=data['shape_type'],
                    label=data['label'],
                    x=data['x'],
                    y=data['y'],
                    color=data['color']
                )
            
            # Restore edges
            self.edges.clear()
            for eid, data in prev_state['edges'].items():
                self.edges[eid] = Edge(
                    id=eid,
                    source_id=data['source_id'],
                    target_id=data['target_id']
                )
            
            self._show_status("Undo successful")
    
    def _screen_to_cursor(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """Convert normalized hand coordinates to screen position"""
        # Since we are using a mirrored frame for detection, X is already consistent with the view
        screen_x = norm_x * WINDOW_WIDTH
        screen_y = norm_y * WINDOW_HEIGHT
        return screen_x, screen_y

    
    def _get_hovered_node(self) -> Optional[Node]:
        """Find node under cursor"""
        for node in self.nodes.values():
            if node.contains(self.cursor_x, self.cursor_y):
                return node
        return None
    
    def _get_hovered_palette_item(self) -> Optional[PaletteItem]:
        """Find palette item under cursor"""
        for item in self.palette_items:
            if item.contains(self.cursor_x, self.cursor_y):
                return item
        return None
    
    def _update(self):
        """Main update loop"""
        hand_data = self.tracker.update()
        
        # Reset hover states
        for node in self.nodes.values():
            node.is_hovered = False
        for item in self.palette_items:
            item.is_hovered = False
        
        if not hand_data.detected:
            self.state = AppState.NO_HAND
            self.was_pinching = False
            return
        
        if self.state == AppState.EDITING_TEXT:
            # Don't update cursor or interactions while editing text
            self.was_pinching = is_pinching
            return

        # Update cursor position
        idx_x, idx_y, _ = hand_data.index_tip
        self.cursor_x, self.cursor_y = self._screen_to_cursor(idx_x, idx_y)

        
        is_pinching = hand_data.is_pinching
        is_fist = hand_data.is_fist
        pinch_started = is_pinching and not self.was_pinching
        pinch_ended = not is_pinching and self.was_pinching
        
        # =============================================================
        # FIST = PAN
        # =============================================================
        if is_fist:
            if self.state != AppState.PAN:
                self.state = AppState.PAN
                self.pan_start_x = self.cursor_x
                self.pan_start_y = self.cursor_y
            else:
                dx = self.cursor_x - self.pan_start_x
                dy = self.cursor_y - self.pan_start_y
                for node in self.nodes.values():
                    node.x += dx
                    node.y += dy
                self.pan_start_x = self.cursor_x
                self.pan_start_y = self.cursor_y
            self.was_pinching = is_pinching
            return



        
        # =============================================================
        # CHECK PALETTE HOVER (Selection)
        # =============================================================
        hovered_palette = self._get_hovered_palette_item()
        if hovered_palette:
            hovered_palette.is_hovered = True
            
            # Allow switching selection if:
            # 1. Pinch triggered (Edge) OR
            # 2. Pinching and moving over a NEW item (Continuous)
            is_new_selection = (hovered_palette.shape_info['id'] != self.selected_shape_type)
            
            if (pinch_started) or (is_pinching and is_new_selection):
                item_id = hovered_palette.shape_info['id']
                
                # Handle Tools vs Shapes
                if item_id == 'tool_clear':
                    self.nodes.clear()
                    self.edges.clear()
                    self._show_status("Canvas Cleared!")
                    self.current_tool = Tool.SELECT
                    self.selected_shape_type = None
                    self.pinch_handled = True
                    return # Don't select the button
                
                # 1. Deselect others
                for item in self.palette_items:
                    item.is_selected = False
                
                # 2. Select this one
                hovered_palette.is_selected = True
                self.selected_shape_type = item_id # Store ID even for tools
                
                # 3. Set Tool State
                if item_id == 'tool_connect':
                    self.current_tool = Tool.CONNECT
                    self._show_status("MODE: CONNECTION - Drag from one node to another")
                elif item_id == 'tool_trash':
                    self.current_tool = Tool.DELETE
                    self._show_status("MODE: DELETE - Pinch nodes to destroy them")
                else:
                    self.current_tool = Tool.PLACE
                    self._show_status(f"LOCKED: {hovered_palette.shape_info['name']}")
                
                self.pinch_handled = True
            
            self.state = AppState.PALETTE_HOVER
            self.was_pinching = is_pinching
            return



        
        # =============================================================
        # CHECK NODE HOVER
        # =============================================================
        hovered_node = self._get_hovered_node()
        if hovered_node:
            hovered_node.is_hovered = True
            self.hovered_node = hovered_node
        else:
            self.hovered_node = None
        
        # =============================================================
        # DRAGGING STATE
        # =============================================================
        if self.state == AppState.DRAGGING:
            if is_pinching and self.dragged_node:
                self.dragged_node.x = self.cursor_x + self.drag_offset_x
                self.dragged_node.y = self.cursor_y + self.drag_offset_y
                
                # Deletion gesture: Drag to left edge (palette) to delete
                if self.cursor_x < CANVAS_OFFSET_X:
                    self._delete_node(self.dragged_node)
                    self.dragged_node = None
                    self.state = AppState.IDLE
                    self._show_status("Node deleted")
            else:
                # Release drag
                if self.dragged_node:
                    self.dragged_node.is_dragging = False
                    self._save_history()
                self.dragged_node = None
                self.state = AppState.IDLE
            self.was_pinching = is_pinching
            return

        
        # =============================================================
        # CONNECTING STATE (Drag & Drop)
        # =============================================================
        if self.state == AppState.CONNECTING_START:
            # We are currently dragging a connection line...
            if not is_pinching:
                # Released pinch
                # Perform a fresh hit-test to be absolutely sure
                target_node = self._get_hovered_node()
                
                if target_node and target_node != self.connection_source:
                    # Valid target
                    self._create_edge(self.connection_source, target_node)
                    self._show_status(f"Connected {self.connection_source.label} → {target_node.label}")
                else:
                    self._show_status("Connection cancelled")
                
                # Cleanup
                if self.connection_source:
                    self.connection_source.is_selected = False
                self.connection_source = None
                self.state = AppState.IDLE
                self.pinch_handled = True
            
            self.was_pinching = is_pinching
            return


        
        # =============================================================
        # PINCH ACTIONS
        # =============================================================
        if pinch_started and not self.pinch_handled:
            # Check if cursor is on canvas (not palette)
            if self.cursor_x > CANVAS_OFFSET_X:
                
                # FORCE SNAP for Connection Tool
                # If we are in Connect mode, check for nodes nearby even if not perfectly hovering
                if self.current_tool == Tool.CONNECT and not hovered_node:
                    # Find nearest node within generous radius
                    nearest = None
                    min_dist = 60 # Snap distance
                    for node in self.nodes.values():
                        # Simple distance check to center
                        cx, cy = node.x + node.width/2, node.y + node.height/2
                        dist = ((self.cursor_x - cx)**2 + (self.cursor_y - cy)**2)**0.5
                        if dist < min_dist:
                            nearest = node
                            min_dist = dist
                    
                    if nearest:
                        hovered_node = nearest # Snap to it!

                if hovered_node:
                    # TOOL: CONNECT (Priority)
                    if self.current_tool == Tool.CONNECT:
                        print(f"DEBUG: STARTING CONNECT from {hovered_node.label}")
                        self.connection_source = hovered_node
                        self.state = AppState.CONNECTING_START
                        self._show_status(f"Connecting from {hovered_node.label}...")
                        self.pinch_handled = True
                        self.was_pinching = is_pinching
                        return # CRITICAL: Return to prevent state overwrite!
                    
                    # TOOL: DELETE
                    elif self.current_tool == Tool.DELETE:


                        self._delete_node(hovered_node)
                        self.pinch_handled = True
                        
                    # TOOL: SELECT / PLACE -> Drag Node
                    else:
                        # Drag Existing Node
                        self.dragged_node = hovered_node
                        hovered_node.is_dragging = True
                        
                        # Select it
                        if self.selected_node:
                            self.selected_node.is_selected = False
                        self.selected_node = hovered_node
                        self.selected_node.is_selected = True
                        
                        self.drag_offset_x = hovered_node.x - self.cursor_x
                        self.drag_offset_y = hovered_node.y - self.cursor_y
                        self.state = AppState.DRAGGING
                        self._show_status(f"Dragging {hovered_node.label}")
                        
                elif self.current_tool == Tool.PLACE and self.selected_shape_type:
                    # Create New Shape


                    # Start Creating New Shape (Draw-to-Size)
                    self.state = AppState.RESIZING_NEW_SHAPE
                    self.resize_start_x = self.cursor_x
                    self.resize_start_y = self.cursor_y
                    
                    # Create initial node with 0 size
                    self.active_sizing_node = self._create_node(
                        self.selected_shape_type, 
                        self.cursor_x, self.cursor_y
                    )
                    self.active_sizing_node.width = 10 # Start small visible
                    self.active_sizing_node.height = 10
                    self.active_sizing_node.is_selected = True # visual feedback
                    
                    self._show_status("Drag to size the shape...")
                else:
                    self._show_status("Select a shape from the palette first!")
        
        # =============================================================
        # RESIZING NEW SHAPE STATE
        # =============================================================
        if self.state == AppState.RESIZING_NEW_SHAPE:
            if is_pinching and self.active_sizing_node:
                # Calculate new dimensions based on drag
                current_w = abs(self.cursor_x - self.resize_start_x)
                current_h = abs(self.cursor_y - self.resize_start_y)
                
                # GRID SNAPPING (Fixes unevenness/jitter)
                snap_size = 20
                width_snapped = max(snap_size, round(current_w / snap_size) * snap_size)
                height_snapped = max(snap_size, round(current_h / snap_size) * snap_size)
                
                # Square snap for symmetrical shapes (optional, but helps circles etc)
                if self.active_sizing_node.shape_type in ['circle', 'diamond']:
                    # Force 1:1 aspect for these if nearly square?
                    # For now just let user control it, but the grid snap helps.
                    pass

                self.active_sizing_node.width = width_snapped
                self.active_sizing_node.height = height_snapped
                
                # Update center position (Anchored resize)
                center_x = self.resize_start_x + (self.cursor_x - self.resize_start_x) / 2
                center_y = self.resize_start_y + (self.cursor_y - self.resize_start_y) / 2
                
                self.active_sizing_node.x = center_x
                self.active_sizing_node.y = center_y

                
            elif not is_pinching:
                # Finished sizing
                if self.active_sizing_node:
                    self.active_sizing_node.is_selected = False
                    self._show_status(f"Created {self.active_sizing_node.label}")
                    self.active_sizing_node = None
                    self._save_history()
                self.state = AppState.IDLE
                self.pinch_handled = True # Prevent immediate re-trigger
            
            self.was_pinching = is_pinching
            return

        
        if pinch_ended:
            self.pinch_handled = False
        
        # Update state based on hover
        if hovered_node:
            self.state = AppState.NODE_HOVER
        elif self.cursor_x > CANVAS_OFFSET_X:
            self.state = AppState.PLACING_SHAPE if self.selected_shape_type else AppState.IDLE
        else:
            self.state = AppState.IDLE
        
        self.was_pinching = is_pinching
    
    def _draw(self):
        """Render the scene"""
        # 1. Draw Camera Background (AR View)
        if hasattr(self.tracker, 'latest_frame') and self.tracker.latest_frame is not None:
            # Convert OpenCV (BGR) to Pygame (RGB)
            frame_rgb = cv2.cvtColor(self.tracker.latest_frame, cv2.COLOR_BGR2RGB)
            # Resize to window size if needed (keeping aspect ratio usually better, but stretch for full immersion)
            frame_rgb = cv2.resize(frame_rgb, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            self.screen.blit(frame_surf, (0, 0))
            
            # Dark overlay for better UI contrast
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 80)) # Semi-transparent black
            self.screen.blit(overlay, (0, 0))
        else:
            self.screen.fill(COLORS['background'])
        
        # 2. Draw Palette (Left Sidebar - Semi-Transparent)
        palette_surf = pygame.Surface((PALETTE_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        palette_surf.fill((*COLORS['palette_bg'][:3], 200)) # Alpha 200
        self.screen.blit(palette_surf, (0, 0))
        
        # Draw palette title
        title_surf = self.title_font.render("Shapes", True, COLORS['text'])
        self.screen.blit(title_surf, (15, 20))
        
        # Draw palette separator
        pygame.draw.line(self.screen, COLORS['grid'], 
                        (PALETTE_WIDTH, 0), (PALETTE_WIDTH, WINDOW_HEIGHT), 2)
        
        # Draw palette items
        for item in self.palette_items:
            item.draw(self.screen, self.small_font)
        
        # 3. Draw Grid (Subtle overlay on video)
        grid_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        grid_spacing = 40
        grid_color = (*COLORS['grid'][:3], 40) # Very subtle grid
        
        for x in range(CANVAS_OFFSET_X, WINDOW_WIDTH, grid_spacing):
            pygame.draw.line(grid_surf, grid_color, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, grid_spacing):
            pygame.draw.line(grid_surf, grid_color, 
                           (CANVAS_OFFSET_X, y), (WINDOW_WIDTH, y))
        self.screen.blit(grid_surf, (0, 0))

        
        # Draw edges (Permanent)
        for edge in self.edges.values():
            edge.draw(self.screen, self.nodes)
        
        # Draw PENDING connection line (While dragging)
        if self.state == AppState.CONNECTING_START and self.connection_source:
            # Calculate start point (center of source)
            start_x = self.connection_source.x + self.connection_source.width / 2
            start_y = self.connection_source.y + self.connection_source.height / 2
            
            # End point is cursor
            end_x = self.cursor_x
            end_y = self.cursor_y
            
            # Draw distinct line (Yellow/Orange) with circle at end
            pygame.draw.line(self.screen, (255, 200, 0), (start_x, start_y), (end_x, end_y), 4)
            pygame.draw.circle(self.screen, (255, 200, 0), (int(end_x), int(end_y)), 6)

        
        # Draw nodes
        for node in self.nodes.values():
            node.draw(self.screen, self.font)
        
        # Draw cursor
        if self.state != AppState.NO_HAND:
            # Cursor glow
            glow_size = 25 if self.was_pinching else 15
            
            # Determine color
            if self.was_pinching:
                glow_color = (100, 255, 100) # Green (Active)
            elif self.current_tool == Tool.CONNECT:
                glow_color = (0, 255, 255) # Cyan (Connect Mode)
                glow_size = 20
            elif self.current_tool == Tool.DELETE:
                glow_color = (255, 50, 50) # Red (Delete Mode)
            else:
                glow_color = COLORS['cursor_glow']

            pygame.draw.circle(self.screen, glow_color, 
                             (int(self.cursor_x), int(self.cursor_y)), glow_size)

            # Cursor core
            core_size = 12 if self.was_pinching else 8
            pygame.draw.circle(self.screen, COLORS['cursor'], 
                             (int(self.cursor_x), int(self.cursor_y)), core_size)
            
            # Pinch text indicator
            if self.was_pinching:
                label = self.small_font.render("PINCH", True, (100, 255, 100))
                self.screen.blit(label, (self.cursor_x + 20, self.cursor_y - 20))

            
            # Show placement preview
            if self.selected_shape_type and self.cursor_x > CANVAS_OFFSET_X and not self.hovered_node:
                shape_info = next((s for s in SHAPE_TYPES if s['id'] == self.selected_shape_type), None)
                if shape_info:
                    preview_color = (*shape_info['color'][:3],)
                    # Draw semi-transparent preview
                    preview_surf = pygame.Surface((100, 60), pygame.SRCALPHA)
                    pygame.draw.rect(preview_surf, (*preview_color, 80), (0, 0, 100, 60), border_radius=8)
                    self.screen.blit(preview_surf, (self.cursor_x - 50, self.cursor_y - 30))
        
        # Draw status bar
        status_bg = pygame.Rect(CANVAS_OFFSET_X, WINDOW_HEIGHT - 40, 
                               WINDOW_WIDTH - CANVAS_OFFSET_X, 40)
        pygame.draw.rect(self.screen, (20, 22, 28), status_bg)
        
        # Status message
        if pygame.time.get_ticks() < self.status_time:
            status_surf = self.font.render(self.status_message, True, COLORS['status'])
            self.screen.blit(status_surf, (CANVAS_OFFSET_X + 20, WINDOW_HEIGHT - 28))

        # Draw Debug Gesture Info (Always useful for troubleshooting)
        state_text = f"State: {self.state.name}"
        if hasattr(self.tracker, 'hand_data'):
            hd = self.tracker.hand_data
            if hd.detected:
                state_text += f" | Pinch: {hd.is_pinching} | Fist: {hd.is_fist}"
        
        debug_col = (255, 0, 255) if self.state != AppState.IDLE else (255, 255, 0)
        debug_surf = self.title_font.render(state_text, True, debug_col)
        self.screen.blit(debug_surf, (CANVAS_OFFSET_X + 20, 20))


        
        # Tool info
        tool_text = f"Tool: {'PLACE' if self.selected_shape_type else 'SELECT'}"
        if self.current_tool == Tool.CONNECT:
            tool_text = "Tool: CONNECT"
        tool_surf = self.font.render(tool_text, True, COLORS['text'])
        self.screen.blit(tool_surf, (CANVAS_OFFSET_X + 20, WINDOW_HEIGHT - 60))
        
        # Draw LARGE Selected Shape Indicator (Top Center)
        if self.selected_shape_type:
            shape_info = next((s for s in SHAPE_TYPES if s['id'] == self.selected_shape_type), None)
            if shape_info:
                # Background
                indicator_rect = pygame.Rect(WINDOW_WIDTH//2 - 100, 20, 200, 50)
                pygame.draw.rect(self.screen, (0, 0, 0, 150), indicator_rect, border_radius=25)
                pygame.draw.rect(self.screen, shape_info['color'], indicator_rect, 2, border_radius=25)
                
                # Text
                label = self.title_font.render(f"ACTIVE: {shape_info['name']}", True, shape_info['color'])
                label_rect = label.get_rect(center=indicator_rect.center)
                self.screen.blit(label, label_rect)

        
        # Instructions
        inst = "[C] Clear | [E] Connect Mode | [DEL] Delete | [U] Undo | [D] Debug | [ESC] Quit"
        inst_surf = self.small_font.render(inst, True, COLORS['text_dim'])
        self.screen.blit(inst_surf, (CANVAS_OFFSET_X + 15, 10))
        
        # Node count
        count_text = f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}"
        count_surf = self.small_font.render(count_text, True, COLORS['text_dim'])
        self.screen.blit(count_surf, (WINDOW_WIDTH - 150, 10))
        
        # Debug info
        if self.debug_mode:
            debug_lines = [
                f"State: {self.state.name}",
                f"Cursor: ({self.cursor_x:.0f}, {self.cursor_y:.0f})",
                f"Shape: {self.selected_shape_type or 'None'}",
                f"FPS: {self.clock.get_fps():.0f}",
            ]
            
            if hasattr(self.tracker, 'hand_data'):
                hand = self.tracker.hand_data
                debug_lines.append(f"Pinch: {'YES' if hand.is_pinching else 'NO'}")
                debug_lines.append(f"Fist: {'YES' if hand.is_fist else 'NO'}")
            
            for i, line in enumerate(debug_lines):
                surf = self.small_font.render(line, True, (200, 200, 100))
                self.screen.blit(surf, (WINDOW_WIDTH - 180, 40 + i * 18))
        
        pygame.display.flip()
    
    def _save_diagram(self):
        """Save diagram to disk"""
        try:
            data = {
                'nodes': [{'id': n.id, 'shape_type': n.shape_type, 'label': n.label, 
                          'x': n.x, 'y': n.y, 'color': n.color} 
                         for n in self.nodes.values()],
                'edges': [{'id': e.id, 'source_id': e.source_id, 'target_id': e.target_id} 
                         for e in self.edges.values()]
            }
            with open('diagram.json', 'w') as f:
                json.dump(data, f, indent=2)
            self._show_status("Diagram saved to diagram.json")
        except Exception as e:
            self._show_status(f"Error saving: {e}")

    def _load_diagram(self):
        """Load diagram from disk"""
        try:
            if not os.path.exists('diagram.json'):
                self._show_status("No saved diagram found")
                return
            
            with open('diagram.json', 'r') as f:
                data = json.load(f)
            
            self.nodes.clear()
            self.edges.clear()
            
            for n_data in data['nodes']:
                node = Node(
                    id=n_data['id'],
                    shape_type=n_data['shape_type'],
                    label=n_data['label'],
                    x=n_data['x'],
                    y=n_data['y'],
                    color=tuple(n_data['color'])
                )
                self.nodes[node.id] = node
                # Update counter to avoid ID collisions
                nid_num = int(node.id.split('_')[1])
                self.node_counter = max(self.node_counter, nid_num)
            
            for e_data in data['edges']:
                edge = Edge(
                    id=e_data['id'],
                    source_id=e_data['source_id'],
                    target_id=e_data['target_id']
                )
                self.edges[edge.id] = edge
                # Update counter
                eid_num = int(edge.id.split('_')[1])
                self.edge_counter = max(self.edge_counter, eid_num)
                
            self._show_status("Diagram loaded from diagram.json")
            self._save_history() # Checkpoint
        except Exception as e:
            self._show_status(f"Error loading: {e}")

    def _handle_events(self):
        """Handle keyboard/mouse events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Text Editing for Selected Node
                if self.selected_node:
                    if event.key == pygame.K_RETURN:
                        # Deselect on Enter
                        self.selected_node.is_selected = False
                        self.selected_node = None
                        self._save_history()
                        self._show_status("Finished editing")
                    elif event.key == pygame.K_BACKSPACE:
                        self.selected_node.label = self.selected_node.label[:-1]
                    elif event.unicode.isprintable():
                        # Append character
                        self.selected_node.label += event.unicode
                    continue # Skip other checks
                    
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    
                elif event.key == pygame.K_c:
                    self.nodes.clear()
                    self.edges.clear()
                    self._show_status("Canvas cleared")
                    
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                    
                elif event.key == pygame.K_u:
                    self._undo()
                
                elif event.key == pygame.K_s:
                    self._save_diagram()
                
                elif event.key == pygame.K_l:
                    self._load_diagram()
                    
                elif event.key == pygame.K_e:
                    # Toggle connect mode
                    if self.current_tool == Tool.CONNECT:
                        self.current_tool = Tool.SELECT
                        self._show_status("Connect mode OFF")
                    else:
                        self.current_tool = Tool.CONNECT
                        self._show_status("Connect mode ON - pinch two nodes to connect!")
                    
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    if self.hovered_node:
                        self._delete_node(self.hovered_node)
                        self.hovered_node = None

    
    def run(self):
        """Main loop"""
        print("\n" + "=" * 65)
        print("  SpatialFlow Design - Hand Gesture Shape Builder")
        print("=" * 65)
        print("\nGestures:")
        print("  • Point finger: Move cursor")
        print("  • Pinch on palette: Select shape")
        print("  • Pinch on canvas: Place shape / Drag existing shape")
        print("  • Fist: Pan the canvas")
        print("\nKeyboard:")
        print("  • [E] Toggle connect mode (pinch two nodes to connect)")
        print("  • [DEL] Delete hovered node")
        print("  • [U] Undo | [C] Clear | [D] Debug | [ESC] Quit")
        print("\n" + "=" * 65 + "\n")
        
        if not self.tracker.start():
            print("Failed to start hand tracker!")
            return
        
        print("Detailed Log: Application started, entering main loop.")
        try:
            while self.running:
                try:
                    self._handle_events()
                    self._update()
                    self._draw()
                    self.clock.tick(60)
                except Exception as frame_error:
                    print(f"Error in frame loop: {frame_error}")
                    import traceback
                    traceback.print_exc()
                    # Don't exit, just continue to next frame
                    
        except KeyboardInterrupt:
            print("Quitting via KeyboardInterrupt")
        finally:
            self.tracker.stop()
            pygame.quit()
            import sys
            sys.exit()


if __name__ == "__main__":
    app = SpatialFlowDesign()
    app.run()

