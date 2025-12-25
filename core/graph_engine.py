"""
SpatialFlow Graph Engine Module
================================

NetworkX-based graph logic with "Diagram-as-Code" DSL parser.
Supports directed and undirected graphs with 3D force-directed layouts.

Features:
    - DSL Parser: "A -> B -> C" syntax for directed graphs
    - 3D Spring Layout using NetworkX
    - Graph serialization/deserialization
    - AI generation mock (ready for LLM integration)
"""

import re
import json
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    """Types of edges in the diagram"""
    DIRECTED = "->"
    UNDIRECTED = "--"
    BIDIRECTIONAL = "<->"


@dataclass
class NodeData:
    """Data associated with a graph node"""
    label: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: str = "#4A90D9"
    shape: str = "cube"
    size: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeData:
    """Data associated with a graph edge"""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECTED
    label: str = ""
    color: str = "#FFFFFF"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DiagramGraph:
    """
    High-level wrapper around NetworkX graph for diagram operations.
    
    Provides:
        - Node/Edge management with rich metadata
        - 3D layout calculation
        - Serialization for persistence
    """
    
    def __init__(self, directed: bool = True):
        """
        Initialize the diagram graph.
        
        Args:
            directed: Whether the graph is directed
        """
        self._directed = directed
        self._graph: nx.DiGraph | nx.Graph = nx.DiGraph() if directed else nx.Graph()
        self._node_data: Dict[str, NodeData] = {}
        self._edge_data: Dict[Tuple[str, str], EdgeData] = {}
        self._layout_scale: float = 5.0
    
    @property
    def graph(self) -> nx.DiGraph | nx.Graph:
        """Access the underlying NetworkX graph"""
        return self._graph
    
    @property
    def nodes(self) -> List[str]:
        """Get all node IDs"""
        return list(self._graph.nodes())
    
    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Get all edge tuples"""
        return list(self._graph.edges())
    
    @property
    def node_count(self) -> int:
        """Get the number of nodes"""
        return self._graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Get the number of edges"""
        return self._graph.number_of_edges()
    
    def add_node(
        self,
        node_id: str,
        label: Optional[str] = None,
        position: Optional[Tuple[float, float, float]] = None,
        color: str = "#4A90D9",
        shape: str = "cube",
        size: float = 1.0,
        **kwargs
    ) -> NodeData:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            label: Display label (defaults to node_id)
            position: 3D position (will be calculated by layout if None)
            color: Node color (hex string)
            shape: Node shape ("cube", "sphere", etc.)
            size: Node size multiplier
            **kwargs: Additional metadata
            
        Returns:
            NodeData for the created node
        """
        label = label or node_id
        position = position or (0.0, 0.0, 0.0)
        
        node_data = NodeData(
            label=label,
            position=position,
            color=color,
            shape=shape,
            size=size,
            metadata=kwargs
        )
        
        self._graph.add_node(node_id)
        self._node_data[node_id] = node_data
        
        return node_data
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.DIRECTED,
        label: str = "",
        color: str = "#FFFFFF",
        weight: float = 1.0,
        **kwargs
    ) -> EdgeData:
        """
        Add an edge between two nodes.
        
        If the nodes don't exist, they will be created automatically.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge connection
            label: Edge label
            color: Edge color
            weight: Edge weight
            **kwargs: Additional metadata
            
        Returns:
            EdgeData for the created edge
        """
        # Auto-create nodes if they don't exist
        if source not in self._node_data:
            self.add_node(source)
        if target not in self._node_data:
            self.add_node(target)
        
        edge_data = EdgeData(
            source=source,
            target=target,
            edge_type=edge_type,
            label=label,
            color=color,
            weight=weight,
            metadata=kwargs
        )
        
        self._graph.add_edge(source, target, weight=weight)
        self._edge_data[(source, target)] = edge_data
        
        # Handle bidirectional edges
        if edge_type == EdgeType.BIDIRECTIONAL and self._directed:
            self._graph.add_edge(target, source, weight=weight)
            reverse_data = EdgeData(
                source=target,
                target=source,
                edge_type=EdgeType.DIRECTED,
                label=label,
                color=color,
                weight=weight,
                metadata=kwargs
            )
            self._edge_data[(target, source)] = reverse_data
        
        return edge_data
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its connected edges.
        
        Args:
            node_id: Node to remove
            
        Returns:
            True if removed, False if node didn't exist
        """
        if node_id not in self._node_data:
            return False
        
        # Remove associated edges
        edges_to_remove = [
            e for e in self._edge_data.keys()
            if e[0] == node_id or e[1] == node_id
        ]
        for edge in edges_to_remove:
            del self._edge_data[edge]
        
        self._graph.remove_node(node_id)
        del self._node_data[node_id]
        
        return True
    
    def remove_edge(self, source: str, target: str) -> bool:
        """
        Remove an edge between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            True if removed, False if edge didn't exist
        """
        edge_key = (source, target)
        if edge_key not in self._edge_data:
            return False
        
        self._graph.remove_edge(source, target)
        del self._edge_data[edge_key]
        
        return True
    
    def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get node data by ID"""
        return self._node_data.get(node_id)
    
    def get_edge(self, source: str, target: str) -> Optional[EdgeData]:
        """Get edge data by source and target"""
        return self._edge_data.get((source, target))
    
    def set_node_position(self, node_id: str, position: Tuple[float, float, float]):
        """Set a node's 3D position"""
        if node_id in self._node_data:
            self._node_data[node_id].position = position
    
    def calculate_layout(
        self,
        algorithm: str = "spring",
        scale: float = 5.0,
        seed: int = 42
    ):
        """
        Calculate 3D positions for all nodes using a layout algorithm.
        
        Args:
            algorithm: Layout algorithm ("spring", "shell", "kamada_kawai")
            scale: Scale factor for positions
            seed: Random seed for reproducibility
        """
        self._layout_scale = scale
        
        if self.node_count == 0:
            return
        
        if self.node_count == 1:
            # Single node at origin
            node_id = self.nodes[0]
            self._node_data[node_id].position = (0.0, 0.0, 0.0)
            return
        
        # Calculate 3D layout
        if algorithm == "spring":
            pos = nx.spring_layout(
                self._graph,
                dim=3,
                scale=scale,
                seed=seed,
                k=2.0,  # Optimal distance between nodes
                iterations=50
            )
        elif algorithm == "shell":
            # Shell layout is 2D, we'll add Z variation
            pos_2d = nx.shell_layout(self._graph, scale=scale)
            pos = {
                node: (x, y, np.random.uniform(-scale/4, scale/4))
                for node, (x, y) in pos_2d.items()
            }
        elif algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(
                self._graph,
                dim=3,
                scale=scale
            )
        else:
            # Default to spring layout
            pos = nx.spring_layout(self._graph, dim=3, scale=scale, seed=seed)
        
        # Apply positions to node data
        for node_id, coords in pos.items():
            if node_id in self._node_data:
                self._node_data[node_id].position = tuple(coords)
    
    def clear(self):
        """Clear all nodes and edges from the graph"""
        self._graph.clear()
        self._node_data.clear()
        self._edge_data.clear()
    
    def to_dict(self) -> Dict:
        """
        Serialize the graph to a dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "directed": self._directed,
            "nodes": {
                node_id: {
                    "label": data.label,
                    "position": list(data.position),
                    "color": data.color,
                    "shape": data.shape,
                    "size": data.size,
                    "metadata": data.metadata
                }
                for node_id, data in self._node_data.items()
            },
            "edges": [
                {
                    "source": data.source,
                    "target": data.target,
                    "type": data.edge_type.value,
                    "label": data.label,
                    "color": data.color,
                    "weight": data.weight,
                    "metadata": data.metadata
                }
                for data in self._edge_data.values()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DiagramGraph":
        """
        Create a graph from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            DiagramGraph instance
        """
        graph = cls(directed=data.get("directed", True))
        
        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            graph.add_node(
                node_id,
                label=node_data.get("label", node_id),
                position=tuple(node_data.get("position", [0, 0, 0])),
                color=node_data.get("color", "#4A90D9"),
                shape=node_data.get("shape", "cube"),
                size=node_data.get("size", 1.0),
                **node_data.get("metadata", {})
            )
        
        # Add edges
        for edge_data in data.get("edges", []):
            edge_type = EdgeType(edge_data.get("type", "->"))
            graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                edge_type=edge_type,
                label=edge_data.get("label", ""),
                color=edge_data.get("color", "#FFFFFF"),
                weight=edge_data.get("weight", 1.0),
                **edge_data.get("metadata", {})
            )
        
        return graph
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DiagramGraph":
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


def parse_dsl(dsl_text: str, auto_layout: bool = True) -> DiagramGraph:
    """
    Parse a Diagram-as-Code DSL string into a DiagramGraph.
    
    Syntax:
        - "A -> B" : Directed edge from A to B
        - "A -- B" : Undirected edge
        - "A <-> B" : Bidirectional edge
        - "A -> B -> C" : Chain of directed edges
        - "[A:Start] -> B" : Node with custom label
        - "A // comment" : Comments are ignored
        
    Examples:
        >>> graph = parse_dsl("Start -> Process -> End")
        >>> graph = parse_dsl('''
        ...     [A:User Input] -> [B:Validation]
        ...     B -> [C:Processing]
        ...     B -> [D:Error Handler]
        ...     C -> [E:Output]
        ... ''')
    
    Args:
        dsl_text: DSL string to parse
        auto_layout: Whether to calculate layout after parsing
        
    Returns:
        DiagramGraph with nodes and edges
    """
    graph = DiagramGraph(directed=True)
    
    # Pattern for node with optional label: [id:label] or just id
    node_pattern = r'\[(\w+):([^\]]+)\]|(\w+)'
    
    # Pattern for edges
    edge_patterns = [
        (r'<->', EdgeType.BIDIRECTIONAL),
        (r'->', EdgeType.DIRECTED),
        (r'--', EdgeType.UNDIRECTED),
    ]
    
    # Process each line
    lines = dsl_text.strip().split('\n')
    
    for line in lines:
        # Remove comments
        line = line.split('//')[0].strip()
        if not line:
            continue
        
        # Determine edge type for this line
        edge_type = EdgeType.DIRECTED
        edge_symbol = '->'
        
        for pattern, etype in edge_patterns:
            if re.search(pattern, line):
                edge_type = etype
                edge_symbol = pattern
                break
        
        # Split by edge symbol
        parts = re.split(edge_symbol.replace('-', r'\-'), line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) < 2:
            # Single node or invalid syntax
            for part in parts:
                match = re.match(r'\[(\w+):([^\]]+)\]|(\w+)', part)
                if match:
                    if match.group(1):
                        graph.add_node(match.group(1), label=match.group(2).strip())
                    elif match.group(3):
                        graph.add_node(match.group(3))
            continue
        
        # Process node chain
        node_ids = []
        for part in parts:
            match = re.match(r'\[(\w+):([^\]]+)\]|(\w+)', part)
            if match:
                if match.group(1):
                    node_id = match.group(1)
                    label = match.group(2).strip()
                else:
                    node_id = match.group(3)
                    label = node_id
                
                if node_id not in graph._node_data:
                    graph.add_node(node_id, label=label)
                node_ids.append(node_id)
        
        # Create edges between consecutive nodes
        for i in range(len(node_ids) - 1):
            graph.add_edge(node_ids[i], node_ids[i + 1], edge_type=edge_type)
    
    # Calculate layout
    if auto_layout and graph.node_count > 0:
        graph.calculate_layout()
    
    return graph


def generate_from_prompt(user_prompt: str) -> str:
    """
    Generate DSL text from a natural language prompt.
    
    This is a MOCK implementation that returns sample diagrams.
    In production, this would call Google's Generative AI API.
    
    Args:
        user_prompt: Natural language description of the diagram
        
    Returns:
        DSL string that can be parsed by parse_dsl()
    """
    # Mock responses based on keywords
    prompt_lower = user_prompt.lower()
    
    if "login" in prompt_lower or "auth" in prompt_lower:
        return """
[Start:User Clicks Login] -> [Input:Enter Credentials]
Input -> [Validate:Validate Input]
Validate -> [Check:Check Database]
Check -> [Success:Login Success]
Check -> [Fail:Login Failed]
Fail -> Input
Success -> [Dashboard:Show Dashboard]
"""
    
    elif "api" in prompt_lower or "request" in prompt_lower:
        return """
[Client:Client App] -> [Request:HTTP Request]
Request -> [Gateway:API Gateway]
Gateway -> [Auth:Auth Service]
Auth -> [API:API Handler]
API -> [DB:Database]
DB -> API
API -> [Response:HTTP Response]
Response -> Client
"""
    
    elif "flow" in prompt_lower or "process" in prompt_lower:
        return """
[Start:Start] -> [Step1:Step 1]
Step1 -> [Decision:Decision Point]
Decision -> [Path1:Path A]
Decision -> [Path2:Path B]
Path1 -> [End:End]
Path2 -> End
"""
    
    elif "tree" in prompt_lower or "hierarchy" in prompt_lower:
        return """
[Root:Root Node] -> [Child1:Child A]
Root -> [Child2:Child B]
Root -> [Child3:Child C]
Child1 -> [Leaf1:Leaf 1]
Child1 -> [Leaf2:Leaf 2]
Child2 -> [Leaf3:Leaf 3]
Child3 -> [Leaf4:Leaf 4]
Child3 -> [Leaf5:Leaf 5]
"""
    
    else:
        # Default simple flow
        return """
[A:Start] -> [B:Process]
B -> [C:Decision]
C -> [D:Action]
C -> [E:Alternative]
D -> [F:End]
E -> F
"""


# Color palettes for node styling
NODE_PALETTES = {
    "default": ["#4A90D9", "#7B68EE", "#20B2AA", "#FF6B6B", "#FFD93D"],
    "warm": ["#FF6B6B", "#FFA07A", "#FFD93D", "#FF8C42", "#E85D04"],
    "cool": ["#4A90D9", "#7B68EE", "#20B2AA", "#48CAE4", "#90BE6D"],
    "monochrome": ["#2B2D42", "#3D405B", "#5C677D", "#7F8C8D", "#95A5A6"],
    "pastel": ["#B5838D", "#E5989B", "#FFB4A2", "#FFCAB1", "#6D6875"]
}


def apply_palette(graph: DiagramGraph, palette_name: str = "default"):
    """
    Apply a color palette to the nodes in the graph.
    
    Args:
        graph: DiagramGraph to colorize
        palette_name: Name of the palette to use
    """
    colors = NODE_PALETTES.get(palette_name, NODE_PALETTES["default"])
    
    for i, node_id in enumerate(graph.nodes):
        node = graph.get_node(node_id)
        if node:
            node.color = colors[i % len(colors)]


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("Graph Engine Demo")
    print("=" * 60)
    
    # Test DSL parsing
    dsl = """
    [Start:Begin Here] -> [ProcessA:Process A]
    ProcessA -> [Decision:Make Decision]
    Decision -> [PathA:Path A]
    Decision -> [PathB:Path B]
    PathA -> [End:Finish]
    PathB -> End
    """
    
    print("\nParsing DSL:")
    print(dsl)
    
    graph = parse_dsl(dsl)
    
    print(f"\nNodes: {graph.node_count}")
    for node_id in graph.nodes:
        node = graph.get_node(node_id)
        print(f"  - {node_id}: '{node.label}' at {node.position}")
    
    print(f"\nEdges: {graph.edge_count}")
    for source, target in graph.edges:
        edge = graph.get_edge(source, target)
        print(f"  - {source} {edge.edge_type.value} {target}")
    
    print("\n" + "=" * 60)
    print("AI Generation Demo")
    print("=" * 60)
    
    prompt = "Create a login flow diagram"
    print(f"\nPrompt: {prompt}")
    print("\nGenerated DSL:")
    print(generate_from_prompt(prompt))
