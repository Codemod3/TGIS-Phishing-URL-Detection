import unittest
import networkx as nx
from src.graph.builder import GraphBuilder
from src.features.graph_features import GraphFeatureExtractor
from src.core.logger import log

class TestGraphFeatures(unittest.TestCase):

    def setUp(self):
        self.builder = GraphBuilder()
        self.extractor = GraphFeatureExtractor()
        
        # Build 1: Safe Cluster (google.com)
        self.builder.add_node('DOMAIN', 'google.com', domain='google.com', label='safe')
        self.builder.add_node('IP', '8.8.8.8', ip_address='8.8.8.8')
        self.builder.add_node('REGISTRAR', 'markmonitor', name='MarkMonitor', reputation_score=1.0)
        self.builder.add_node('NAMESERVER', 'ns1.google.com', hostname='ns1.google.com')
        self.builder.add_node('SSL_ISSUER', 'gts_ca', issuer_name='Google Trust Services')
        
        self.builder.add_edge('google.com', '8.8.8.8', 'DOMAIN_TO_IP')
        self.builder.add_edge('google.com', 'markmonitor', 'DOMAIN_TO_REGISTRAR')
        self.builder.add_edge('google.com', 'ns1.google.com', 'DOMAIN_TO_NAMESERVER')
        self.builder.add_edge('google.com', 'gts_ca', 'DOMAIN_TO_SSL_ISSUER')
        
        # Build 2: Phishing Cluster (evil-bank.com)
        self.builder.add_node('DOMAIN', 'evil-bank.com', domain='evil-bank.com', label='phishing')
        self.builder.add_node('URL', 'http://evil-bank.com/login', label='phishing')
        self.builder.add_edge('http://evil-bank.com/login', 'evil-bank.com', 'URL_TO_DOMAIN')

    def test_safe_domain_features(self):
        features = self.extractor.extract("https://google.com", self.builder.graph)
        print(f"\nSafe features: {features}")
        
        # Verification
        self.assertEqual(len(features), 10)
        self.assertGreaterEqual(features['domain_cluster_size'], 5)
        self.assertEqual(features['suspicious_neighbor_count'], 0)
        self.assertEqual(features['cluster_phishing_ratio'], 0.0)
        self.assertEqual(features['ip_shared_domains_count'], 1)
        self.assertGreaterEqual(features['nameserver_trust_score'], 0.5)
        self.assertGreaterEqual(features['graph_centrality_score'], 0.0)

    def test_phishing_cluster_features(self):
        url = "http://evil-bank.com/login"
        features = self.extractor.extract(url, self.builder.graph)
        print(f"\nPhish features: {features}")
        
        # Verification
        self.assertEqual(features['domain_cluster_size'], 2)
        self.assertEqual(features['suspicious_neighbor_count'], 1)
        self.assertEqual(features['cluster_phishing_ratio'], 1.0)

    def test_fallback_unseen_url(self):
        # URL not in graph
        features = self.extractor.extract("https://unknown-site.org", self.builder.graph)
        
        # Verification: should use -1 fallbacks
        self.assertEqual(features['domain_cluster_size'], -1)
        self.assertEqual(features['suspicious_neighbor_count'], -1)
        self.assertEqual(features['community_detection_label'], -1)

if __name__ == "__main__":
    unittest.main()
