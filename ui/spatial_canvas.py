"""
SpatialFlow Spatial Canvas Module
==================================

Ursina Engine entities for 3D diagram rendering.
Provides NodeEntity, EdgeEntity, and CursorTrail for the spatial interface.

Features:
    - NodeEntity: Interactive 3D nodes with text labels
    - EdgeEntity: Dynamic line meshes between nodes
    - CursorTrail: Debug visualization for tracking stability
    - SpatialCanvas: High-level manager for the 3D scene
"""

from ursina import *
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from collections import deque


# Custom colors for the spatial interface
class Colors:
    """Color palette for the spatial canvas"""
    BACKGROUND = color.rgb(40, 44, 52)  # Dark gray (visible background)
    NODE_DEFAULT = color.rgb(74, 144, 217)
    NODE_HOVER = color.rgb(255, 215, 0)
    NODE_SELECTED = color.rgb(50, 205, 50)
    EDGE_DEFAULT = color.rgba(200, 200, 200, 200)  # Brighter edges
    CURSOR = color.rgb(255, 100, 100)
    TRAIL = color.rgba(255, 100, 100, 150)
    GRID = color.rgba(80, 80, 100, 80)


class NodeEntity(Entity):
    """
    3D Node entity for diagram visualization.
    
    Features:
        - Cube or sphere mesh with customizable appearance
        - Text label that billboards towards camera
        - Hover and selection states
        - Draggable interaction support
    """
    
    def __init__(
        self,
        node_id: str,
        label: str = "",
        position: Tuple[float, float, float] = (0, 0, 0),
        color_rgb: Tuple[int, int, int] = (74, 144, 217),
        size: float = 1.0,
        shape: str = "cube",
        **kwargs
    ):
        """
        Initialize a node entity.
        
        Args:
            node_id: Unique identifier for this node
            label: Display label text
            position: 3D position (x, y, z)
            color_rgb: RGB color tuple
            size: Scale multiplier
            shape: "cube" or "sphere"
        """
        # Determine model based on shape
        model_name = 'cube' if shape == 'cube' else 'sphere'
        
        super().__init__(
            model=model_name,
            position=Vec3(*position),
            color=color.rgb(*color_rgb),
            scale=size * 0.8,
            collider='box',
            unlit=True,  # Bypass shader issues on macOS
            **kwargs
        )
        
        self.node_id = node_id
        self.label_text = label or node_id
        self.base_color = color.rgb(*color_rgb)
        self.base_scale = self.scale
        
        # State
        self.is_hovered = False
        self.is_selected = False
        self.is_dragging = False
        
        # Create text label
        self._label = Text(
            text=self.label_text,
            parent=self,
            position=(0, 0.8, 0),
            scale=15,
            origin=(0, 0),
            color=color.white,
            billboard=True
        )
        
        # Smooth animation targets
        self._target_scale = self.scale
        self._target_color = self.base_color
        
        # Callbacks (use private attributes to avoid Ursina property conflicts)
        self._click_callback: Optional[Callable] = None
        self._hover_enter_callback: Optional[Callable] = None
        self._hover_exit_callback: Optional[Callable] = None
    
    @property
    def click_callback(self) -> Optional[Callable]:
        return self._click_callback
    
    @click_callback.setter
    def click_callback(self, value: Optional[Callable]):
        self._click_callback = value
    
    @property
    def hover_enter_callback(self) -> Optional[Callable]:
        return self._hover_enter_callback
    
    @hover_enter_callback.setter  
    def hover_enter_callback(self, value: Optional[Callable]):
        self._hover_enter_callback = value
    
    @property
    def hover_exit_callback(self) -> Optional[Callable]:
        return self._hover_exit_callback
    
    @hover_exit_callback.setter
    def hover_exit_callback(self, value: Optional[Callable]):
        self._hover_exit_callback = value
    
    def update(self):
        """Per-frame update for smooth animations"""
        # Smooth scale interpolation
        self.scale = lerp(self.scale, self._target_scale, time.dt * 10)
        
        # Smooth color interpolation
        self.color = lerp(self.color, self._target_color, time.dt * 8)
    
    def set_hovered(self, hovered: bool):
        """Set hover state with visual feedback"""
        if hovered == self.is_hovered:
            return
            
        self.is_hovered = hovered
        
        if hovered:
            self._target_scale = self.base_scale * 1.15
            self._target_color = Colors.NODE_HOVER
            if self._hover_enter_callback:
                self._hover_enter_callback(self)
        else:
            self._target_scale = self.base_scale
            self._target_color = self.base_color
            if self._hover_exit_callback:
                self._hover_exit_callback(self)
    
    def set_selected(self, selected: bool):
        """Set selection state"""
        self.is_selected = selected
        if selected:
            self.base_color = Colors.NODE_SELECTED
            self._target_color = Colors.NODE_SELECTED
        else:
            # Reset to original color
            self._target_color = self.base_color
    
    def set_dragging(self, dragging: bool):
        """Set dragging state"""
        self.is_dragging = dragging
        if dragging:
            self._target_scale = self.base_scale * 1.2
        else:
            self._target_scale = self.base_scale * 1.15 if self.is_hovered else self.base_scale
    
    def on_mouse_enter(self):
        """Built-in Ursina hover enter handler"""
        self.set_hovered(True)
    
    def on_mouse_exit(self):
        """Built-in Ursina hover exit handler"""
        self.set_hovered(False)
    
    def input(self, key):
        """Handle input events"""
        if key == 'left mouse down' and self.hovered:
            if self._click_callback:
                self._click_callback(self)


class EdgeEntity(Entity):
    """
    Dynamic edge entity connecting two nodes.
    
    Uses Ursina's Mesh class to draw lines that update
    when node positions change.
    """
    
    def __init__(
        self,
        edge_id: str,
        source_node: NodeEntity,
        target_node: NodeEntity,
        directed: bool = True,
        color_rgba: Tuple[int, int, int, int] = (255, 255, 255, 180),
        thickness: float = 2.0,
        **kwargs
    ):
        """
        Initialize an edge entity.
        
        Args:
            edge_id: Unique identifier for this edge
            source_node: Starting node entity
            target_node: Ending node entity
            directed: Whether to show arrow head
            color_rgba: RGBA color tuple
            thickness: Line thickness
        """
        super().__init__(**kwargs)
        
        self.edge_id = edge_id
        self.source_node = source_node
        self.target_node = target_node
        self.directed = directed
        self.edge_color = color.rgba(*[c/255 for c in color_rgba])
        self.thickness = thickness
        
        # Create line mesh entity
        self._line = Entity(
            parent=self,
            model=None,
            color=self.edge_color,
            unlit=True
        )
        
        # Arrow head for directed edges
        self._arrow: Optional[Entity] = None
        if directed:
            self._arrow = Entity(
                parent=self,
                model='cone',
                color=self.edge_color,
                scale=0.15,
                unlit=True
            )
        
        # Initial update
        self._update_geometry()
    
    def update(self):
        """Update geometry each frame to track node positions"""
        self._update_geometry()
    
    def _update_geometry(self):
        """Rebuild the line mesh based on current node positions"""
        start = self.source_node.position
        end = self.target_node.position
        
        # Calculate direction
        direction = end - start
        length = direction.length()
        
        if length < 0.001:
            return
        
        direction_normalized = direction.normalized()
        
        # Offset start and end to not overlap with nodes
        node_radius = self.source_node.scale_x * 0.6
        start_offset = start + direction_normalized * node_radius
        end_offset = end - direction_normalized * node_radius
        
        # Create line mesh
        vertices = [
            start_offset,
            end_offset
        ]
        
        self._line.model = Mesh(
            vertices=vertices,
            mode='line',
            thickness=self.thickness
        )
        
        # Update arrow head
        if self._arrow and self.directed:
            # Position arrow at the end
            self._arrow.position = end_offset
            
            # Orient arrow towards target
            self._arrow.look_at(end)
            self._arrow.rotation_x = -90  # Adjust for cone model
    
    def set_color(self, r: int, g: int, b: int, a: int = 255):
        """Update edge color"""
        self.edge_color = color.rgba(r/255, g/255, b/255, a/255)
        self._line.color = self.edge_color
        if self._arrow:
            self._arrow.color = self.edge_color
    
    def cleanup(self):
        """Clean up the edge entity's children"""
        if self._line:
            destroy(self._line)
            self._line = None
        if self._arrow:
            destroy(self._arrow)
            self._arrow = None


class CursorTrail(Entity):
    """
    Trail renderer for cursor visualization.
    
    Creates a fading trail behind the cursor to visualize
    tracking stability. Shows jitter when tracking is unstable.
    """
    
    def __init__(
        self,
        max_points: int = 50,
        trail_color: Tuple[int, int, int] = (255, 100, 100),
        fade_speed: float = 3.0,
        **kwargs
    ):
        """
        Initialize the cursor trail.
        
        Args:
            max_points: Maximum points in the trail
            trail_color: RGB color for the trail
            fade_speed: How fast the trail fades
        """
        super().__init__(**kwargs)
        
        self.max_points = max_points
        self.trail_color = trail_color
        self.fade_speed = fade_speed
        
        # Trail points with timestamps
        self._points: deque = deque(maxlen=max_points)
        
        # Trail mesh entity
        self._trail_mesh = Entity(
            parent=self,
            model=None,
            unlit=True
        )
    
    def add_point(self, position: Vec3):
        """Add a new point to the trail"""
        self._points.append({
            'pos': Vec3(position),
            'time': time.time()
        })
    
    def update(self):
        """Update the trail mesh"""
        if len(self._points) < 2:
            self._trail_mesh.model = None
            return
        
        current_time = time.time()
        
        # Build vertices with fading alpha
        vertices = []
        colors = []
        
        for i, point in enumerate(self._points):
            age = current_time - point['time']
            alpha = max(0, 1 - age * self.fade_speed)
            alpha *= (i / len(self._points))  # Additional fade by position
            
            vertices.append(point['pos'])
            colors.append(color.rgba(
                self.trail_color[0]/255,
                self.trail_color[1]/255,
                self.trail_color[2]/255,
                alpha * 0.8
            ))
        
        # Create line strip mesh
        if len(vertices) >= 2:
            self._trail_mesh.model = Mesh(
                vertices=vertices,
                mode='line',
                thickness=3,
                colors=colors
            )
    
    def clear(self):
        """Clear all trail points"""
        self._points.clear()
        self._trail_mesh.model = None


class CursorEntity(Entity):
    """
    3D cursor entity that follows hand position.
    
    Visualizes the current cursor position in 3D space.
    """
    
    def __init__(
        self,
        size: float = 0.3,
        color_rgb: Tuple[int, int, int] = (255, 100, 100),
        **kwargs
    ):
        """
        Initialize the cursor entity.
        
        Args:
            size: Cursor size
            color_rgb: RGB color tuple
        """
        super().__init__(
            model='sphere',
            color=color.rgb(*color_rgb),
            scale=size,
            unlit=True,
            **kwargs
        )
        
        # Inner glow
        self._glow = Entity(
            parent=self,
            model='sphere',
            color=color.rgba(1, 1, 1, 0.5),
            scale=0.6,
            unlit=True
        )
        
        # Pulse animation
        self._pulse_time = 0
    
    def update(self):
        """Animate the cursor"""
        self._pulse_time += time.dt * 3
        pulse = 1 + np.sin(self._pulse_time) * 0.1
        self._glow.scale = 0.6 * pulse


class GridPlane(Entity):
    """
    3D grid plane for spatial reference.
    """
    
    def __init__(
        self,
        size: int = 20,
        spacing: float = 1.0,
        color_rgba: Tuple[int, int, int, int] = (100, 100, 100, 50),
        **kwargs
    ):
        """
        Initialize the grid plane.
        
        Args:
            size: Grid size (number of lines)
            spacing: Distance between grid lines
            color_rgba: RGBA color
        """
        super().__init__(**kwargs)
        
        grid_color = color.rgba(*[c/255 for c in color_rgba])
        
        # Create grid lines
        vertices = []
        
        half = size * spacing / 2
        
        # X-axis parallel lines
        for i in range(-size//2, size//2 + 1):
            x = i * spacing
            vertices.append(Vec3(x, 0, -half))
            vertices.append(Vec3(x, 0, half))
        
        # Z-axis parallel lines
        for i in range(-size//2, size//2 + 1):
            z = i * spacing
            vertices.append(Vec3(-half, 0, z))
            vertices.append(Vec3(half, 0, z))
        
        self.model = Mesh(
            vertices=vertices,
            mode='line',
            thickness=1
        )
        self.color = grid_color
        self.unlit = True


class SpatialCanvas:
    """
    High-level manager for the 3D diagram canvas.
    
    Coordinates between the graph data and visual entities.
    """
    
    def __init__(self):
        """Initialize the spatial canvas"""
        self.nodes: Dict[str, NodeEntity] = {}
        self.edges: Dict[str, EdgeEntity] = {}
        
        # Cursor and trail for hand tracking
        self.cursor = CursorEntity()
        self.trail = CursorTrail()
        
        # Grid for spatial reference (brighter for dark background)
        self.grid = GridPlane(size=20, spacing=2, color_rgba=(80, 80, 120, 100))
        self.grid.position = (0, -5, 0)
        
        # Pivot entity for camera orbiting
        self.pivot = Entity()
        
        # Callbacks
        self.on_node_clicked: Optional[Callable] = None
        self.on_node_hovered: Optional[Callable] = None
    
    def create_node(
        self,
        node_id: str,
        label: str = "",
        position: Tuple[float, float, float] = (0, 0, 0),
        color_rgb: Tuple[int, int, int] = (74, 144, 217),
        size: float = 1.0
    ) -> NodeEntity:
        """
        Create a new node entity.
        
        Args:
            node_id: Unique identifier
            label: Display label
            position: 3D position
            color_rgb: Node color
            size: Scale multiplier
            
        Returns:
            The created NodeEntity
        """
        node = NodeEntity(
            node_id=node_id,
            label=label,
            position=position,
            color_rgb=color_rgb,
            size=size
        )
        
        # Set up callbacks (using new property names)
        node.click_callback = self._handle_node_click
        node.hover_enter_callback = self._handle_node_hover_enter
        
        self.nodes[node_id] = node
        return node
    
    def create_edge(
        self,
        source_id: str,
        target_id: str,
        directed: bool = True
    ) -> Optional[EdgeEntity]:
        """
        Create an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            directed: Whether edge is directed
            
        Returns:
            The created EdgeEntity, or None if nodes don't exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            print(f"[SpatialCanvas] Cannot create edge: nodes {source_id} or {target_id} not found")
            return None
        
        edge_id = f"{source_id}__{target_id}"
        
        edge = EdgeEntity(
            edge_id=edge_id,
            source_node=self.nodes[source_id],
            target_node=self.nodes[target_id],
            directed=directed
        )
        
        self.edges[edge_id] = edge
        return edge
    
    def remove_node(self, node_id: str):
        """Remove a node and its connected edges"""
        if node_id not in self.nodes:
            return
        
        # Remove connected edges
        edges_to_remove = [
            eid for eid in self.edges
            if node_id in eid.split('__')
        ]
        for edge_id in edges_to_remove:
            self.remove_edge_by_id(edge_id)
        
        # Remove node
        destroy(self.nodes[node_id])
        del self.nodes[node_id]
    
    def remove_edge_by_id(self, edge_id: str):
        """Remove an edge by its ID"""
        if edge_id in self.edges:
            edge = self.edges[edge_id]
            edge.cleanup()  # Clean up children first
            destroy(edge)   # Then destroy the edge entity
            del self.edges[edge_id]
    
    def clear(self):
        """Remove all nodes and edges"""
        for edge_id in list(self.edges.keys()):
            self.remove_edge_by_id(edge_id)
        for node_id in list(self.nodes.keys()):
            self.remove_node(node_id)
    
    def update_cursor(self, position: Tuple[float, float, float]):
        """
        Update cursor position and add trail point.
        
        Args:
            position: New cursor position
        """
        pos = Vec3(*position)
        self.cursor.position = pos
        self.trail.add_point(pos)
    
    def get_hovered_node(self) -> Optional[NodeEntity]:
        """Get the currently hovered node"""
        for node in self.nodes.values():
            if node.is_hovered:
                return node
        return None
    
    def _handle_node_click(self, node: NodeEntity):
        """Internal node click handler"""
        if self.on_node_clicked:
            self.on_node_clicked(node)
    
    def _handle_node_hover_enter(self, node: NodeEntity):
        """Internal node hover handler"""
        if self.on_node_hovered:
            self.on_node_hovered(node)
    
    def load_from_graph(self, graph):
        """
        Load entities from a DiagramGraph.
        
        Args:
            graph: DiagramGraph instance from graph_engine
        """
        self.clear()
        
        # Create nodes
        for node_id in graph.nodes:
            node_data = graph.get_node(node_id)
            if node_data:
                # Parse color from hex string
                hex_color = node_data.color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                
                self.create_node(
                    node_id=node_id,
                    label=node_data.label,
                    position=node_data.position,
                    color_rgb=(r, g, b),
                    size=node_data.size
                )
        
        # Create edges
        for source, target in graph.edges:
            self.create_edge(source, target, directed=True)
        
        print(f"[SpatialCanvas] Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")


# Standalone test
if __name__ == "__main__":
    app = Ursina()
    
    # Set up camera
    camera.position = (0, 5, -20)
    camera.look_at((0, 0, 0))
    
    # Create canvas
    canvas = SpatialCanvas()
    
    # Create some test nodes
    canvas.create_node("A", "Start", (-5, 0, 0), (74, 144, 217))
    canvas.create_node("B", "Process", (0, 0, 0), (123, 104, 238))
    canvas.create_node("C", "End", (5, 0, 0), (255, 107, 107))
    
    # Create edges
    canvas.create_edge("A", "B")
    canvas.create_edge("B", "C")
    
    # Test cursor movement
    def update():
        # Simulate cursor movement in a circle
        t = time.time()
        x = np.sin(t) * 3
        y = np.cos(t) * 3
        canvas.update_cursor((x, y, -5))
    
    # Add sky
    Sky(color=Colors.BACKGROUND)
    
    # Add ambient light
    AmbientLight(color=color.rgba(100, 100, 100, 100))
    
    app.run()
