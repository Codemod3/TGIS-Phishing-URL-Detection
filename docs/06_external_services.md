# External Services, Content Extraction & Community Detection

This file covers the parts of the system that reach outside the process to gather data:
fetching web pages, querying DNS servers, reading SSL certificates, checking Google's
threat database, caching results in Redis, and grouping domains into communities.

---

## 1. Content Feature Extraction — `src/features/content_features.py`

This is the most expensive feature extractor because it actually **downloads and parses
the web page** at the target URL. It extracts 15 features from the HTML.

### Fetching the Page

```python
# src/features/content_features.py  (line 17-45)
self.headers = {
    "User-Agent": "Mozilla/5.0 ... Chrome/91.0 ..."  # Pretend to be a browser
}
response = requests.get(url, timeout=self.timeout, headers=self.headers, allow_redirects=True)
html_content = response.text
features['num_redirects'] = len(response.history)   # How many hops before landing
features['uses_https']     = 1 if response.url.startswith('https') else 0
```

The `User-Agent` header makes the request look like it comes from a real Chrome browser.
Some phishing sites serve different content to known bots — impersonating Chrome bypasses that.
`response.history` is a list of intermediate redirect responses.
A URL that redirects 5 times before landing is suspicious.

### Parsing HTML with BeautifulSoup

```python
# src/features/content_features.py  (line 47)
soup = BeautifulSoup(html_content, 'html.parser')
```

**BeautifulSoup** converts the raw HTML string into a tree structure you can navigate
with Python. Think of it as a map of the webpage:

```python
# Find all HTML forms on the page
forms = soup.find_all('form')

# Check if any form contains a password field
for form in forms:
    if form.find('input', {'type': 'password'}):
        has_password = 1
```

`soup.find_all('form')` returns a list of all `<form>` elements.
`{'type': 'password'}` is a filter — only return `<input>` tags with that attribute.
Phishing pages almost always have a login form with a password field to steal credentials.

### Link Analysis

```python
# src/features/content_features.py  (line 68-81)
links = soup.find_all('a', href=True)   # All hyperlinks
for link in links:
    href = link['href']
    link_domain = urlparse(href).netloc
    if not link_domain or link_domain == domain:
        internal += 1   # Points to the same domain
    else:
        external += 1   # Points somewhere else

features['external_internal_ratio'] = external / (internal + 1)
# The +1 prevents division by zero when there are no internal links
```

A legitimate bank website links mostly to itself (account pages, FAQs, etc.).
A phishing page built from a stolen template often has many external links pointing to
the real bank's servers (for images, CSS) while the form submits to the attacker.
A high `external_internal_ratio` is a strong phishing signal.

### JavaScript Obfuscation Detection

```python
# src/features/content_features.py  (line 95-100)
script_content = "".join([s.text for s in soup.find_all('script') if s.text])

obfuscation_patterns = [r'eval\(', r'unescape\(', r'String\.fromCharCode', r'atob\(']
features['uses_javascript_obfuscation'] = 1 if any(
    re.search(p, script_content) for p in obfuscation_patterns
) else 0
```

Phishing pages often hide their malicious behaviour in obfuscated JavaScript.
- `eval(...)` executes a dynamically built string as code — a classic hiding technique.
- `String.fromCharCode(72, 101, 108, ...)` converts character codes to the string "Hel..."
  — a way to spell out URLs and code without ever writing them literally.
- `atob(...)` decodes Base64 strings — another way to hide URLs.

`re.search(p, script_content)` checks if the regex pattern `p` appears anywhere in the
combined JavaScript. `any(...)` returns True if at least one pattern matches.

### Brand Mismatch in Title

```python
# src/features/content_features.py  (line 102-109)
title = soup.title.string.lower() if soup.title else ""
for brand in DomainFeatureExtractor.BRAND_LIST:
    if brand in title and brand not in domain.lower():
        features['html_title_brand_mismatch'] = 1
        break
```

If the page `<title>` says "PayPal Login" but the domain is `xk9f.tk`, that's a red flag.
The page is impersonating a brand it doesn't belong to.

### Mixed Content

```python
# src/features/content_features.py  (line 115-125)
if features['uses_https']:
    resources = soup.find_all(['img', 'script', 'link'], src=True)
    for res in resources:
        src = res.get('src', res.get('href'))
        if src and src.startswith('http:'):   # http over https page = mixed
            mixed = 1
            break
```

Mixed content means an HTTPS page loads some resources over plain HTTP.
This is both a security issue and a tell-tale sign of a hastily assembled phishing page
that copied a legitimate site's HTTPS template but forgot to update resource URLs.

---

## 2. DNS Resolver — `src/external/dns_resolver.py`

DNS (Domain Name System) is the internet's phone book — it translates domain names
into IP addresses. The resolver queries multiple DNS record types per domain:

```python
# src/external/dns_resolver.py  (line 32-40)
record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
for rtype in record_types:
    results[f"{rtype}_records"] = self._query_record(domain, rtype)

results['dns_record_count'] = sum(len(v) for k, v in results.items() if isinstance(v, list))
```

| Record Type | What it maps | Phishing signal |
|------------|-------------|-----------------|
| A | Domain → IPv4 address | Brand-new domains have 1–2 records |
| MX | Domain → Mail server | No MX = domain can't receive email (throwaway) |
| NS | Domain → Name servers | Cheap/free name servers are common with phishing |
| TXT | Miscellaneous (SPF, DKIM) | No SPF = no email authentication |
| CNAME | Alias to another domain | |

```python
# src/external/dns_resolver.py  (line 56-67)
def _query_record(self, domain, rtype):
    try:
        answers = dns.resolver.resolve(domain, rtype, lifetime=2.0)  # 2 second timeout
        if rtype == 'MX':
            return [str(r.exchange).rstrip('.') for r in answers]  # Strip trailing dots
        return [str(r).rstrip('.') for r in answers]
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, ...):
        return []   # Empty list, not an exception — handled gracefully
```

`lifetime=2.0` is critical: DNS queries can hang if a nameserver is down. Without a timeout,
one slow domain could block the entire prediction for 30+ seconds.
`.rstrip('.')` removes the trailing dot that DNS records include (e.g. `mail.google.com.`).

---

## 3. SSL Certificate Checker — `src/external/ssl_checker.py`

An SSL certificate proves a website is who it claims to be and encrypts traffic.
Phishing sites often have no certificate, a self-signed certificate, or a certificate
from a free provider issued just hours before the attack.

```python
# src/external/ssl_checker.py  (line 37-52)
context = ssl.create_default_context()   # Use the OS's trusted CA certificates
with socket.create_connection((domain, port), timeout=5) as sock:
    with context.wrap_socket(sock, server_hostname=domain) as ssock:
        cert = ssock.getpeercert()
        result["is_valid"] = True
        result["is_trusted"] = True  # If the TLS handshake succeeded, the cert is trusted
```

`socket.create_connection` opens a raw TCP connection to port 443.
`context.wrap_socket` upgrades it to TLS, performing the certificate handshake.
If the certificate is expired, self-signed, or doesn't match the domain, Python raises an
`ssl.SSLError` automatically — no manual check needed.

```python
# src/external/ssl_checker.py  (line 69-75)
def parse_date(date_str: str):
    dt = datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")
    # "Mar 11 23:59:59 2024 GMT" → datetime object
    return dt.isoformat()  # → "2024-03-11T23:59:59"
```

SSL certificates have an unusual date format. `strptime` parses it using a format string
where `%b` = abbreviated month, `%d` = day, `%H:%M:%S` = time, `%Y` = year, `%Z` = timezone.

---

## 4. Redis Caching — `src/external/cache_manager.py`

Every external lookup (WHOIS, DNS, SSL) takes 1–5 seconds. For popular domains like
`google.com`, re-fetching that data on every prediction would be wasteful and slow.
Redis (an in-memory database) caches the results with an expiry time.

```python
# src/external/cache_manager.py  (line 13-28)
self.client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                           decode_responses=True)
self.client.ping()   # If this fails, Redis isn't running
self.enabled = True
```

If Redis isn't running, the system doesn't crash. It sets `self.enabled = False` and
falls back to fetching data fresh every time. This is **graceful degradation** — the system
works slower without Redis, but it works.

```python
# src/external/cache_manager.py  (line 54-76)
def set(self, key, value, ttl=3600):
    serialized_value = json.dumps(value, default=str)   # Convert dict to JSON string
    self.client.set(key, serialized_value, ex=ttl)      # ex=TTL sets expiry in seconds

def get(self, key):
    data = self.client.get(key)      # Returns the JSON string or None
    if data:
        return json.loads(data)      # Convert JSON string back to dict
    return None
```

Redis stores only strings. `json.dumps` converts a Python dictionary to a JSON string for
storage. `json.loads` reverses that on retrieval. `default=str` handles any non-serializable
values (like datetime objects) by converting them to strings.

**TTL Strategy:**
| Data | TTL | Reason |
|------|-----|--------|
| WHOIS | 7 days | Domain registration rarely changes |
| DNS | 24 hours | DNS records can change, but not that often |
| SSL | 24 hours | Certificates last months, but revocation is rare |

---

## 5. Community Detection (Louvain Algorithm) — `src/graph/analyzer.py`

The TGIS graph groups domains into **communities** — clusters of nodes that are densely
connected to each other but loosely connected to the rest of the graph.

```python
# src/graph/analyzer.py  (line 6-36)
def detect_communities(graph: nx.DiGraph) -> Dict[str, int]:
    undirected_graph = graph.to_undirected()   # Louvain needs undirected graph
    partition = community_louvain.best_partition(undirected_graph)
    # Result: {'google.com': 0, 'gmail.com': 0, 'evil.tk': 3, 'phish.xyz': 3, ...}
    return partition
```

**Why convert to undirected?** The Louvain algorithm works on undirected graphs.
In an undirected graph, a connection between A and B is the same as B→A. Directed edges
(A→B but not B→A) would break the community detection math.

**What is the Louvain algorithm?**
It starts by putting each node in its own community. Then it repeatedly tries moving each
node into a neighbor's community — if the move increases "modularity" (a measure of how
dense connections are within communities vs between them), it keeps the move. It repeats
until no move improves the score.

The result: a community ID number for each domain. Domains in the same community tend to
share infrastructure (same registrar, same IP range, same nameserver). A domain sitting in
a community where 80% of other members are labeled "phishing" is highly suspicious,
even if we've never seen the domain before.

This community label is stored as `community_detection_label` in the feature vector and
used to calculate `cluster_phishing_ratio` and `anomaly_score_in_cluster`.
