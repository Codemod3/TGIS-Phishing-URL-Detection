import math
import re
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
from src.features.base import FeatureExtractor
from src.external.whois_client import WHOISClient
from src.external.dns_resolver import DNSResolver
from src.external.ssl_checker import SSLChecker
from src.core.logger import log

class DomainFeatureExtractor(FeatureExtractor):
    """
    Extracts 20 domain-level metadata and behavioral features.
    Combines external lookups (WHOIS, DNS, SSL) with string-based pattern analysis.
    Uses np.nan for missing data to ensure proper ML imputation later.
    """
    
    # 16. domain_in_brand_list
    BRAND_LIST = {
        "google", "facebook", "amazon", "microsoft", "apple", "netflix", "paypal", 
        "linkedin", "twitter", "ebay", "instagram", "whatsapp", "adobe", "bankofamerica", 
        "chase", "wellsfargo", "citi", "hsbc", "barclays", "blockchain", "binance", 
        "coinbase", "kraken", "metamask", "outlook", "gmail", "yahoo", "icloud"
    }
    
    # 17. tld_suspicious
    SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".icu", ".buzz", ".cn"}

    def __init__(self):
        self.whois_client = WHOISClient()
        self.dns_resolver = DNSResolver()
        self.ssl_checker = SSLChecker()

    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Extract the 20 architectural domain features.
        
        Args:
            url (str): The target URL.
            
        Returns:
            Dict[str, Any]: 20 features covering WHOIS, DNS, SSL, and String Analysis.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0] # Remove port if exists
        
        log.info(f"Extracting domain features for: {domain}")
        
        # 1. Fetch metadata from external clients
        whois_data = self.whois_client.lookup(domain)
        dns_data = self.dns_resolver.resolve(domain)
        ssl_data = self.ssl_checker.check(domain)
        
        features = {}
        now = datetime.now()
        
        # --- WHOIS Features (1-5) ---
        features['domain_age_days'] = self._calc_days_diff(whois_data.get('creation_date'), now, reverse=True) if whois_data else np.nan
        features['domain_expiry_days'] = self._calc_days_diff(whois_data.get('expiration_date'), now) if whois_data else np.nan
        features['domain_registration_length'] = self._calc_days_diff(whois_data.get('expiration_date'), whois_data.get('creation_date')) if whois_data else np.nan
        features['is_registered'] = 1 if whois_data and whois_data.get('domain_name') else 0
        features['registrar_reputation_score'] = self._evaluate_registrar(whois_data.get('registrar')) if whois_data else np.nan
        
        # --- DNS Features (6-9) ---
        features['dns_record_count'] = dns_data.get('dns_record_count', np.nan) if dns_data else np.nan
        features['has_mx_record'] = 1 if dns_data and dns_data.get('MX_records') else 0
        features['has_spf_record'] = 1 if dns_data and any('v=spf1' in str(txt) for txt in dns_data.get('TXT_records', [])) else 0
        features['num_nameservers'] = len(dns_data.get('NS_records', [])) if dns_data else np.nan
        
        # --- SSL Features (10-12) ---
        features['ssl_certificate_valid'] = 1 if ssl_data and ssl_data.get('is_valid') else 0
        features['ssl_certificate_age_days'] = self._calc_days_diff(ssl_data.get('not_before'), now, reverse=True) if ssl_data else np.nan
        features['ssl_issuer_trusted'] = 1 if ssl_data and ssl_data.get('is_trusted') else 0
        
        # --- Reputation Placeholders (13-15) ---
        features['alexa_rank'] = np.nan # Placeholder
        features['google_indexed'] = np.nan # Placeholder
        features['page_rank_score'] = np.nan # Placeholder
        
        # --- String Analysis (16-20) ---
        features['domain_in_brand_list'] = 1 if any(brand in domain.lower() for brand in self.BRAND_LIST) else 0
        features['tld_suspicious'] = 1 if any(domain.lower().endswith(tld) for tld in self.SUSPICIOUS_TLDS) else 0
        
        # Word lengths based on alphanumeric segments
        words = re.findall(r'[a-zA-Z0-9]+', domain.split('.')[0])
        features['shortest_word_length'] = min(len(w) for w in words) if words else 0
        features['longest_word_length'] = max(len(w) for w in words) if words else 0
        
        # Domain Entropy
        features['domain_entropy'] = self._calculate_entropy(domain)
        
        log.debug(f"Extracted {len(features)} domain features.")
        return features

    def _calc_days_diff(self, date1: Any, date2: Any, reverse: bool = False) -> Any:
        """
        Helper to calculate difference in days between two dates.
        Handles ISO strings, lists of dates, and timezone mismatches.
        Returns np.nan on error.
        """
        if date1 is None:
            return np.nan
            
        try:
            # 1. Handle cases where python-whois returns a list (e.g. multi-registrar domains)
            if isinstance(date1, list) and len(date1) > 0:
                date1 = date1[0]
            
            # 2. Parse ISO strings to datetime objects
            if isinstance(date1, str):
                date1 = datetime.fromisoformat(date1)
            if isinstance(date2, str):
                date2 = datetime.fromisoformat(date2)
            
            # 3. Standardize to timezone-naive to prevent "offset-aware vs naive" crashes
            if hasattr(date1, 'tzinfo') and date1.tzinfo is not None:
                date1 = date1.replace(tzinfo=None)
            if hasattr(date2, 'tzinfo') and date2.tzinfo is not None:
                date2 = date2.replace(tzinfo=None)
            
            # 4. Calculate day difference
            diff = (date1 - date2).days if not reverse else (date2 - date1).days
            return float(max(0, diff))
        except Exception as e:
            log.warning(f"Date calculation failure: {e}")
            return np.nan

    def _evaluate_registrar(self, registrar: Optional[str]) -> float:
        """Mocked registrar reputation score. High reputation for major registrars."""
        if registrar is None:
            return np.nan
        top_registrars = {"markmonitor", "godaddy", "namecheap", "google", "amazon", "cloudflare"}
        if any(top in registrar.lower() for top in top_registrars):
            return 0.9
        return 0.5

    def _calculate_entropy(self, text: str) -> float:
        """Shannon Entropy calculation."""
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum([p * math.log2(p) for p in prob])
        return round(entropy, 4)
