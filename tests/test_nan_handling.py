import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.features.domain_features import DomainFeatureExtractor
from src.features.content_features import ContentFeatureExtractor
from src.features.graph_features import GraphFeatureExtractor

def test_domain_nan():
    print("Testing DomainFeatureExtractor NaN handling...")
    extractor = DomainFeatureExtractor()
    
    # Mock external clients to return None (failure)
    extractor.whois_client.lookup = MagicMock(return_value=None)
    extractor.dns_resolver.resolve = MagicMock(return_value=None)
    extractor.ssl_checker.check = MagicMock(return_value=None)
    
    features = extractor.extract("http://failed-domain.com")
    
    # Check key numerical features
    assert np.isnan(features['domain_age_days']), "domain_age_days should be NaN"
    assert np.isnan(features['dns_record_count']), "dns_record_count should be NaN"
    assert np.isnan(features['ssl_certificate_age_days']), "ssl_certificate_age_days should be NaN"
    # Boolean logic preserved (0/1) for binary properties
    assert features['is_registered'] == 0, "is_registered should be 0 (Boolean logic preserved)"
    print("✅ DomainFeatureExtractor passed.")

def test_content_nan():
    print("Testing ContentFeatureExtractor NaN handling...")
    extractor = ContentFeatureExtractor()
    
    # Mock requests.get to raise an exception
    with patch('requests.get', side_effect=Exception("Network Timeout")):
        features = extractor.extract("http://timeout-site.com")
        
    # All numerical/structural features should be NaN
    for feature_name, val in features.items():
        assert np.isnan(val), f"Feature {feature_name} should be NaN on network failure"
    print("✅ ContentFeatureExtractor passed.")

def test_graph_nan():
    print("Testing GraphFeatureExtractor NaN handling...")
    extractor = GraphFeatureExtractor()
    
    # Extract with empty graph
    features = extractor.extract("http://any-url.com", graph=None)
    
    # All features should be NaN
    for feature_name, val in features.items():
        assert np.isnan(val), f"Feature {feature_name} should be NaN on missing graph"
    print("✅ GraphFeatureExtractor passed.")

if __name__ == "__main__":
    try:
        test_domain_nan()
        test_content_nan()
        test_graph_nan()
        print("\n✨ ALL NaN HANDLING TESTS PASSED! ✨")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)
