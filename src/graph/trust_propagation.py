import networkx as nx
from typing import Dict, Any
from src.core.logger import log

def calculate_trust_score(graph: nx.DiGraph, damping_factor: float = 0.3, iterations: int = 10) -> Dict[str, float]:
    """
    Iterative Trust Propagation Algorithm.
    Propagates trust through the graph based on known labels and edge structure.
    
    Args:
        graph (nx.DiGraph): The Trust Graph.
        damping_factor (float): The influence of neighbors (alpha).
        iterations (int): Strategy iterations for convergence.
        
    Returns:
        Dict[str, float]: Final trust score for every node [0.0 - 1.0].
    """
    if not graph or len(graph.nodes) == 0:
        return {}

    log.info(f"Starting Trust Propagation (Iterations: {iterations}, Damping: {damping_factor})")
    
    # 1. Initialize trust scores
    # safe: 1.0, phishing: 0.0, unknown: 0.5
    trust_scores = {}
    for node, data in graph.nodes(data=True):
        label = data.get('label', 'unknown')
        if label == 'safe':
            trust_scores[node] = 1.0
        elif label == 'phishing':
            trust_scores[node] = 0.0
        else:
            trust_scores[node] = 0.5

    # 2. Iterative propagation
    for i in range(iterations):
        new_scores = {}
        for node, data in graph.nodes(data=True):
            # Labeled nodes act as static anchors and do not drift
            label = data.get('label', 'unknown')
            if label in ['safe', 'phishing']:
                new_scores[node] = trust_scores[node]
                continue
                
            neighbors = list(graph.neighbors(node)) + list(graph.predecessors(node))
            if not neighbors:
                new_scores[node] = trust_scores[node]
                continue
                
            neighbor_sum = 0.0
            total_weight = 0.0
            
            for nb in set(neighbors):
                # Get edge weight (default to 1.0 if not specified)
                weight = 1.0
                if graph.has_edge(nb, node):
                    weight = graph[nb][node].get('weight', 1.0)
                elif graph.has_edge(node, nb):
                    weight = graph[node][nb].get('weight', 1.0)
                
                neighbor_sum += trust_scores[nb] * weight
                total_weight += weight
            
            weighted_avg = neighbor_sum / total_weight if total_weight > 0 else 0.5
            new_scores[node] = (1 - damping_factor) * trust_scores[node] + damping_factor * weighted_avg
            
        trust_scores = new_scores
        # log.debug(f"Iteration {i+1} completed.")

    # 3. Final normalization / clamping
    for node in trust_scores:
        trust_scores[node] = round(max(0.0, min(1.0, trust_scores[node])), 4)
        
    log.success("Trust Propagation completed for all nodes.")
    return trust_scores
