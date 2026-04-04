import networkx as nx
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
from src.features.base import FeatureExtractor
from src.graph.trust_propagation import calculate_trust_score
from src.graph.analyzer import detect_communities
from src.core.logger import log

class GraphFeatureExtractor(FeatureExtractor):
    """
    Extracts 10 structural and behavioral features from the Trust Graph (TGIS).
    These features represent relationships, community membership, and propagation patterns.
    Uses np.nan for missing data to ensure proper ML imputation later.
    """
    
    def extract(self, url: str, graph: nx.DiGraph = None, **kwargs) -> Dict[str, Any]:
        """
        Extract graph-based features for a given URL using its node and surrounding clusters.
        Assumes the graph has been pre-processed with global metrics (trust_score, community_id, centrality).
        
        Args:
            url (str): The target URL.
            graph (nx.DiGraph): The current TGIS graph (pre-processed).
            
        Returns:
            Dict[str, Any]: 10 graph features.
        """
        if graph is None or len(graph.nodes) == 0:
            return self._get_default_features(np.nan)
            
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]
        
        # Primary node could be the URL itself or the DOMAIN
        node_id = url if url in graph else domain
        
        features = {}
        
        if node_id not in graph:
            log.warning(f"Node {node_id} not found in graph. Returning default graph features.")
            return self._get_default_features(np.nan)

        # 1. Access pre-calculated node attributes
        node_data = graph.nodes[node_id]
        community_id = node_data.get('community_id', np.nan)
        
        # 1. domain_cluster_size: Total nodes in the same community
        if not np.isnan(community_id):
            cluster_nodes = [n for n, d in graph.nodes(data=True) if d.get('community_id') == community_id]
            features['domain_cluster_size'] = len(cluster_nodes)
        else:
            cluster_nodes = []
            features['domain_cluster_size'] = np.nan
        
        # 2. suspicious_neighbor_count: 1-hop phishing neighbors
        neighbors = list(graph.neighbors(node_id)) + list(graph.predecessors(node_id))
        phishing_neighbors = [n for n in set(neighbors) if graph.nodes[n].get('label') == 'phishing']
        features['suspicious_neighbor_count'] = len(phishing_neighbors)
        
        # 3. cluster_phishing_ratio: Ratio of phishing nodes in community
        if features['domain_cluster_size'] > 0 and not np.isnan(features['domain_cluster_size']):
            phish_in_cluster = [n for n in cluster_nodes if graph.nodes[n].get('label') == 'phishing']
            features['cluster_phishing_ratio'] = len(phish_in_cluster) / features['domain_cluster_size']
        else:
            features['cluster_phishing_ratio'] = np.nan
            
        # 4. registrar_trust_score: Trust level of Registrar node
        registrars = [n for n in graph.successors(domain) if graph.nodes[n].get('type') == 'REGISTRAR']
        features['registrar_trust_score'] = graph.nodes[registrars[0]].get('trust_score', np.nan) if registrars else np.nan
        
        # 5. ip_shared_domains_count: How many domains share the same IP
        ips = [n for n in graph.successors(domain) if graph.nodes[n].get('type') == 'IP']
        if ips:
            shared_domains = [n for n in graph.predecessors(ips[0]) if graph.nodes[n].get('type') == 'DOMAIN']
            features['ip_shared_domains_count'] = len(shared_domains)
        else:
            features['ip_shared_domains_count'] = np.nan
            
        # 6. nameserver_trust_score: Average trust of NS nodes
        ns_nodes = [n for n in graph.successors(domain) if graph.nodes[n].get('type') == 'NAMESERVER']
        if ns_nodes:
            avg_ns_trust = sum(graph.nodes[ns].get('trust_score', 0.5) for ns in ns_nodes) / len(ns_nodes)
            features['nameserver_trust_score'] = round(avg_ns_trust, 4)
        else:
            features['nameserver_trust_score'] = np.nan
            
        # 7. ssl_issuer_trust_score: Trust level of SSL_ISSUER
        issuers = [n for n in graph.successors(domain) if graph.nodes[n].get('type') == 'SSL_ISSUER']
        features['ssl_issuer_trust_score'] = graph.nodes[issuers[0]].get('trust_score', np.nan) if issuers else np.nan
        
        # 8. graph_centrality_score: Betweenness centrality (from pre-calculated attribute)
        features['graph_centrality_score'] = node_data.get('centrality', np.nan)
        
        # 9. community_detection_label: Community ID (already retrieved)
        features['community_detection_label'] = community_id
        
        # 10. anomaly_score_in_cluster: Local deviation (from pre-calculated trust_score)
        if features['domain_cluster_size'] > 1 and not np.isnan(features['domain_cluster_size']):
            my_trust = node_data.get('trust_score', 0.5)
            cluster_trusts = [graph.nodes[n].get('trust_score', 0.5) for n in cluster_nodes]
            avg_cluster_trust = sum(cluster_trusts) / len(cluster_trusts)
            features['anomaly_score_in_cluster'] = round(abs(my_trust - avg_cluster_trust), 4)
        else:
            features['anomaly_score_in_cluster'] = np.nan
            
        return features

    def _get_default_features(self, default_val=np.nan) -> Dict[str, Any]:
        return {
            'domain_cluster_size': default_val,
            'suspicious_neighbor_count': default_val,
            'cluster_phishing_ratio': default_val,
            'registrar_trust_score': default_val,
            'ip_shared_domains_count': default_val,
            'nameserver_trust_score': default_val,
            'ssl_issuer_trust_score': default_val,
            'graph_centrality_score': default_val,
            'community_detection_label': default_val,
            'anomaly_score_in_cluster': default_val
        }
