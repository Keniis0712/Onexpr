"""HTTP client demo using `requests`.

Hits a few public HTTP endpoints to exercise sessions, retries, JSON
decoding, streaming, exception handling — without needing a local
server.

Usage:
    python examples/http_client.py
or onexpr-obfuscated:
    python onexpr.py --input examples/http_client.py --output obf.py
    python obf.py
"""

import requests
from requests.adapters import HTTPAdapter, Retry


def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "POST", "HEAD"),
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "onexpr-demo/1.0",
        "Accept": "text/html,application/json,*/*;q=0.5",
    })
    return s


def main():
    with make_session() as s:
        print("--- HEAD https://example.com/ ---")
        r = s.head("https://example.com/", timeout=10)
        print(r.status_code, r.headers.get("content-type"))

        print("\n--- GET https://example.com/ ---")
        r = s.get("https://example.com/", timeout=10)
        print(r.status_code, len(r.text), "bytes")
        # Pull out the <title>
        import re
        m = re.search(r"<title>(.*?)</title>", r.text, re.IGNORECASE)
        print("title:", m.group(1) if m else "<missing>")

        print("\n--- GET https://httpbin.org/get?x=1&y=2 (json) ---")
        try:
            r = s.get("https://httpbin.org/get", params={"x": 1, "y": 2}, timeout=10)
            r.raise_for_status()
            data = r.json()
            print(r.status_code, "args:", data.get("args"))
        except requests.RequestException as e:
            print("skipped (no network):", type(e).__name__, e)

        print("\n--- GET https://httpbin.org/status/418 ---")
        try:
            r = s.get("https://httpbin.org/status/418", timeout=10)
            r.raise_for_status()
            print("unexpected ok", r.status_code)
        except requests.HTTPError as e:
            print("raised:", e.response.status_code)
        except requests.RequestException as e:
            print("skipped:", type(e).__name__)

        print("\n--- POST https://httpbin.org/post ---")
        try:
            r = s.post(
                "https://httpbin.org/post",
                json={"hello": "world", "n": 7},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            print(r.status_code, "echoed:", data.get("json"))
        except requests.RequestException as e:
            print("skipped:", type(e).__name__)

        print("\n--- streaming GET https://httpbin.org/stream/3 ---")
        try:
            with s.get("https://httpbin.org/stream/3", stream=True, timeout=10) as r:
                count = 0
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        count += 1
                print(r.status_code, "lines:", count)
        except requests.RequestException as e:
            print("skipped:", type(e).__name__)


if __name__ == "__main__":
    main()
