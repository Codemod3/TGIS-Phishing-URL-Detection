import networkx as nx
import community as community_louvain
from typing import Dict, Any
from src.core.logger import log

def detect_communities(graph: nx.DiGraph) -> Dict[str, int]:
    """
    Louvain algorithm for community detection based on structural density.
    Identifies clusters of nodes (phish-farms, corporate networks, CDN nodes).
    
    Args:
        graph (nx.DiGraph): The Trust Graph.
        
    Returns:
        Dict[str, int]: A dictionary mapping node IDs to community IDs.
    """
    if not graph or len(graph.nodes) == 0:
        return {}

    log.info("Starting Louvain Community Detection...")
    
    try:
        # Louvain algorithm requires an undirected graph
        undirected_graph = graph.to_undirected()
        
        # Calculate best partition (resolution=1.0 by default)
        partition = community_louvain.best_partition(undirected_graph)
        
        # Count unique communities
        num_communities = len(set(partition.values()))
        log.success(f"Community Detection completed. Identified {num_communities} communities.")
        
        return partition
    except Exception as e:
        log.error(f"Failed to detect communities: {e}")
        return {}
