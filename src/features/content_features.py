import requests
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from src.features.base import FeatureExtractor
from src.features.domain_features import DomainFeatureExtractor
from src.core.logger import log

class ContentFeatureExtractor(FeatureExtractor):
    """
    Extracts 15 features from the HTML content of a page.
    Handles network errors gracefully by defaulting to np.nan for numerical features.
    """
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def extract(self, url: str, html_content: str = None) -> Dict[str, Any]:
        """
        Extract content features from a URL.
        
        Args:
            url (str): Target URL.
            html_content (str, optional): Pre-fetched HTML if available.
            
        Returns:
            Dict[str, Any]: 15 features.
        """
        log.info(f"Extracting content features for: {url}")
        
        try:
            features = {}
            if not html_content:
                response = requests.get(url, timeout=self.timeout, headers=self.headers, allow_redirects=True)
                html_content = response.text
                features['num_redirects'] = len(response.history)
                features['uses_https'] = 1 if response.url.startswith('https') else 0
            else:
                features['num_redirects'] = 0
                features['uses_https'] = 1 if url.startswith('https') else 0

            soup = BeautifulSoup(html_content, 'html.parser')
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # 1. has_login_form
            forms = soup.find_all('form')
            features['num_forms'] = len(forms)
            has_login = 0
            has_password = 0
            for form in forms:
                if form.find('input', {'type': 'password'}):
                    has_password = 1
                # Common login form indicators in action/id/class
                form_text = str(form).lower()
                if any(k in form_text for k in ['login', 'signin', 'auth']):
                    has_login = 1
            
            features['has_login_form'] = has_login
            features['form_has_password_field'] = has_password

            # 2. Link Analysis
            links = soup.find_all('a', href=True)
            external = 0
            internal = 0
            for link in links:
                href = link['href']
                link_domain = urlparse(href).netloc
                if not link_domain or link_domain == domain:
                    internal += 1
                else:
                    external += 1
            
            features['num_external_links'] = external
            features['num_internal_links'] = internal
            features['external_internal_ratio'] = external / (internal + 1)

            # 5. has_iframe
            features['has_iframe'] = 1 if soup.find('iframe') else 0

            # 7. favicon_matches_domain
            icon_link = soup.find('link', rel=re.compile(r'icon', re.I))
            if icon_link and icon_link.get('href'):
                icon_domain = urlparse(icon_link['href']).netloc
                features['favicon_matches_domain'] = 1 if not icon_domain or icon_domain == domain else 0
            else:
                features['favicon_matches_domain'] = 1 # Default or no icon

            # 8. has_popup & 9. JS Obfuscation
            scripts = soup.find_all('script')
            script_content = "".join([s.text for s in scripts if s.text])
            features['has_popup'] = 1 if 'window.open' in script_content else 0
            
            obfuscation_patterns = [r'eval\(', r'unescape\(', r'String\.fromCharCode', r'atob\(']
            features['uses_javascript_obfuscation'] = 1 if any(re.search(p, script_content) for p in obfuscation_patterns) else 0

            # 10. html_title_brand_mismatch
            title = soup.title.string.lower() if soup.title else ""
            features['html_title_brand_mismatch'] = 0
            if title:
                for brand in DomainFeatureExtractor.BRAND_LIST:
                    if brand in title and brand not in domain.lower():
                        features['html_title_brand_mismatch'] = 1
                        break

            # 11. num_images
            features['num_images'] = len(soup.find_all('img'))

            # 15. has_mixed_content
            if features['uses_https']:
                resources = soup.find_all(['img', 'script', 'link'], src=True)
                mixed = 0
                for res in resources:
                    src = res.get('src', res.get('href'))
                    if src and src.startswith('http:'):
                        mixed = 1
                        break
                features['has_mixed_content'] = mixed
            else:
                features['has_mixed_content'] = 0

            return features

        except Exception as e:
            log.warning(f"Failed to extract content features for {url}: {e}")
            return self._get_default_features(np.nan)

    def _get_default_features(self, default_val=np.nan) -> Dict[str, Any]:
        return {
            'has_login_form': default_val,
            'num_external_links': default_val,
            'num_internal_links': default_val,
            'external_internal_ratio': default_val,
            'has_iframe': default_val,
            'num_redirects': default_val,
            'favicon_matches_domain': default_val,
            'has_popup': default_val,
            'uses_javascript_obfuscation': default_val,
            'html_title_brand_mismatch': default_val,
            'num_images': default_val,
            'num_forms': default_val,
            'form_has_password_field': default_val,
            'uses_https': default_val,
            'has_mixed_content': default_val
        }
