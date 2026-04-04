import dns.resolver
from typing import Optional, Dict, Any, List
from src.external.cache_manager import cache_manager
from src.core.logger import log

class DNSResolver:
    """
    Comprehensive DNS record resolver with integrated caching and robust error handling.
    Used for extracting domain metadata to identify suspicious patterns.
    """
    
    CACHE_PREFIX = "dns:"
    CACHE_TTL = 86400  # 24 hours in seconds

    def resolve(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Query multiple DNS records for a domain with caching.
        
        Args:
            domain (str): The domain to query.
            
        Returns:
            Optional[Dict[str, Any]]: Consolidated DNS record data.
        """
        cache_key = f"{self.CACHE_PREFIX}{domain}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data

        log.info(f"Resolving DNS records for domain: {domain}")
        results = {}
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
        
        try:
            for rtype in record_types:
                results[f"{rtype}_records"] = self._query_record(domain, rtype)
            
            # Additional check: total dns record count
            results['dns_record_count'] = sum(len(v) for k, v in results.items() if isinstance(v, list))
            
            # Store in cache
            if results:
                cache_manager.set(cache_key, results, ttl=self.CACHE_TTL)
            
            return results
        except Exception as e:
            log.error(f"DNS Resolution failed for {domain}: {e}")
            return None

    def _query_record(self, domain: str, rtype: str) -> List[str]:
        """
        Helper to safely query a specific DNS record type with a strict timeout.
        """
        try:
            # Set a 2s timeout so that failures don't bottleneck the entire pipeline
            answers = dns.resolver.resolve(domain, rtype, lifetime=2.0)
            # Normalize list of answers based on record type
            if rtype == 'MX':
                return [str(r.exchange).rstrip('.') for r in answers]
            elif rtype == 'SOA':
                # Simplified SOA string
                return [str(answers[0])]
            else:
                return [str(r).rstrip('.') for r in answers]
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout):
            return []
        except Exception as e:
            log.warning(f"Unexpected error querying {rtype} for {domain}: {e}")
            return []
