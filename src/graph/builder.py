import networkx as nx
import os
import pickle
from typing import Dict, Any, Optional, List
from src.core.logger import log

# Node Types and their required attributes as defined in ARCHITECTURE.md
NODE_TYPES = {
    'URL': {
        'attributes': ['url', 'label', 'first_seen', 'last_seen']
    },
    'DOMAIN': {
        'attributes': ['domain', 'tld', 'creation_date', 'trust_score', 'label']
    },
    'IP': {
        'attributes': ['ip_address', 'country', 'asn', 'trust_score', 'label']
    },
    'REGISTRAR': {
        'attributes': ['name', 'reputation_score']
    },
    'NAMESERVER': {
        'attributes': ['hostname', 'trust_score']
    },
    'SSL_ISSUER': {
        'attributes': ['issuer_name', 'trust_level']
    }
}

# Edge Types and their default weights as defined in ARCHITECTURE.md
EDGE_TYPES = {
    'URL_TO_DOMAIN': {'weight': 1.0},
    'DOMAIN_TO_IP': {'weight': 0.8},
    'DOMAIN_TO_REGISTRAR': {'weight': 0.7},
    'DOMAIN_TO_NAMESERVER': {'weight': 0.6},
    'DOMAIN_TO_SSL_ISSUER': {'weight': 0.5},
    'URL_TO_URL': {'weight': 0.9, 'type': 'redirect'},
    'DOMAIN_TO_DOMAIN': {'weight': 0.4, 'type': 'shares_ip'}
}

class GraphBuilder:
    """
    Service for constructing and managing the TGIS (Trust Graph Intelligence System).
    Uses NetworkX to maintain a directed graph of relationships between network entities.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        log.info("Initialized empty TGIS Graph.")

    def add_node(self, node_type: str, node_id: str, **attributes) -> bool:
        """
        Add a node to the graph with schema validation.
        
        Args:
            node_type (str): Key in NODE_TYPES.
            node_id (str): Unique identifier for the node.
            **attributes: Key-value pairs matching NODE_TYPES[node_type]['attributes'].
            
        Returns:
            bool: True if node was added successfully.
        """
        if node_type not in NODE_TYPES:
            log.error(f"Invalid node type: {node_type}")
            return False
            
        # Ensure only specified attributes are stored (plus node_type)
        allowed_attrs = NODE_TYPES[node_type]['attributes']
        node_attrs = {k: v for k, v in attributes.items() if k in allowed_attrs}
        node_attrs['type'] = node_type
        
        self.graph.add_node(node_id, **node_attrs)
        # log.debug(f"Added {node_type} node: {node_id}")
        return True

    def add_edge(self, source_id: str, target_id: str, edge_type: str, **extra_attrs) -> bool:
        """
        Add a weighted edge between two nodes.
        
        Args:
            source_id (str): Source node identifier.
            target_id (str): Target node identifier.
            edge_type (str): Key in EDGE_TYPES.
            **extra_attrs: Additional data to store on the edge.
            
        Returns:
            bool: True if edge was added successfully.
        """
        if edge_type not in EDGE_TYPES:
            log.error(f"Invalid edge type: {edge_type}")
            return False
            
        if source_id not in self.graph or target_id not in self.graph:
            log.warning(f"Nodes {source_id} or {target_id} not found in graph. Adding edge anyway.")
            
        edge_data = EDGE_TYPES[edge_type].copy()
        edge_data.update(extra_attrs)
        edge_data['edge_type'] = edge_type
        
        self.graph.add_edge(source_id, target_id, **edge_data)
        # log.debug(f"Added {edge_type} edge: {source_id} -> {target_id}")
        return True

    def save_graph(self, path: str):
        """Serialization for the graph to a file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.graph, f)
            log.info(f"Trust graph saved to: {path}")
        except Exception as e:
            log.error(f"Failed to save trust graph: {e}")

    def load_graph(self, path: str):
        """Deserialization for the graph from a file."""
        if not os.path.exists(path):
            log.warning(f"Graph file not found: {path}. Starting with empty graph.")
            return
            
        try:
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
            log.info(f"Trust graph loaded from: {path} ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)")
        except Exception as e:
            log.error(f"Failed to load trust graph: {e}")

    def get_node_count(self) -> int:
        return len(self.graph.nodes)

    def get_edge_count(self) -> int:
        return len(self.graph.edges)
