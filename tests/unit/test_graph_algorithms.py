import os
import unittest
from src.graph.builder import GraphBuilder
from src.graph.trust_propagation import calculate_trust_score
from src.graph.analyzer import detect_communities
from src.core.logger import log

class TestGraphAlgorithms(unittest.TestCase):

    def setUp(self):
        self.builder = GraphBuilder()
        
        # Scenario: 
        # Node A (Labeled Safe: 1.0)
        # Node B (Unknown: 0.5)
        # Node C (Unknown: 0.5)
        # Node D (Labeled Phishing: 0.0) - at the edge
        
        # Safe Cluster
        self.builder.add_node('URL', 'url_safe', label='safe')
        self.builder.add_node('DOMAIN', 'google.com', domain='google.com')
        self.builder.add_node('IP', '8.8.8.8', ip_address='8.8.8.8')
        
        # Connections
        self.builder.add_edge('url_safe', 'google.com', 'URL_TO_DOMAIN')
        self.builder.add_edge('google.com', '8.8.8.8', 'DOMAIN_TO_IP')
        
        # Unknown but connected to Safe
        self.builder.add_node('URL', 'url_pending')
        self.builder.add_edge('url_pending', 'google.com', 'URL_TO_DOMAIN')
        
    def test_trust_propagation(self):
        # Initial scores: Safe=1.0, Pending=0.5
        # After 1 iteration: Pending should increase because it's linked to a 'Safe' node
        
        final_scores = calculate_trust_score(self.builder.graph, damping_factor=0.5, iterations=5)
        
        # Verify Pending became MORE trusted
        self.assertGreater(final_scores['url_pending'], 0.5)
        self.assertEqual(final_scores['url_safe'], 1.0)
        
        log.info(f"Final Scores: {final_scores}")

    def test_community_detection(self):
        # Add another cluster of Phish nodes
        self.builder.add_node('URL', 'url_phish_1', label='phishing')
        self.builder.add_node('URL', 'url_phish_2')
        self.builder.add_node('DOMAIN', 'evil.com')
        self.builder.add_edge('url_phish_1', 'evil.com', 'URL_TO_DOMAIN')
        self.builder.add_edge('url_phish_2', 'evil.com', 'URL_TO_DOMAIN')
        
        communities = detect_communities(self.builder.graph)
        
        # Verify same community for phish URLs on same domain
        self.assertEqual(communities['url_phish_1'], communities['url_phish_2'])
        self.assertEqual(communities['url_phish_1'], communities['evil.com'])
        
        # Verify different communities for separate clusters
        self.assertNotEqual(communities['url_safe'], communities['url_phish_1'])
        
        log.info(f"Communities: {communities}")

if __name__ == "__main__":
    unittest.main()
