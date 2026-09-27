#!/usr/bin/env python3
"""g5-infer gateway (phase 1): OpenAI-compatible front door in stdlib only.

- Exact-prefix cache: identical (model,prompt,params) reuses full response.
- Fan-out: forwards to llama-server slots concurrently (server runs -np 4).
- Metrics: per-request decode tok/s appended to gateway.log.
Upstream: 127.0.0.1:8080 (llama-server). Listens: 127.0.0.1:8791.
"""
import hashlib, http.server, json, threading, time, urllib.request

UP = "http://127.0.0.1:8080"
PORT = 8791
CACHE, LOCK = {}, threading.Lock()
LOG = "/tmp/g5-infer-gateway.log"

def key(model, prompt, params):
    h = hashlib.sha256()
    h.update(json.dumps([model, prompt, params], sort_keys=True).encode())
    return h.hexdigest()

def fwd(path, body):
    req = urllib.request.Request(UP + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

class H(http.server.ThreadingHTTPServer):
    daemon_threads = True

class R(http.server.BaseHTTPRequestHandler):
    server_version = "G5Infer/0.1"
    def log_message(self, *a):
        pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == "/ping":
            return self._json(200, {"ok": True, "gateway": "g5-infer"})
        if self.path == "/stats":
            with LOCK:
                n = len(CACHE)
            return self._json(200, {"ok": True, "prefixEntries": n})
        return self._json(404, {"ok": False})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad json"})
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            return self._json(404, {"ok": False})
        model = body.get("model", "mimo")
        prompt = body.get("prompt", "") or json.dumps(body.get("messages", ""))
        params = {k: body.get(k) for k in ("temperature", "max_tokens", "top_p") if k in body}
        k = key(model, prompt, params)
        with LOCK:
            hit = CACHE.get(k)
        if hit:
            with open(LOG, "a") as f:
                f.write("%s CACHE_HIT prompt=%d\n" % (time.strftime("%H:%M:%S"), len(prompt)))
            return self._json(200, hit)
        t0 = time.time()
        try:
            out = fwd(self.path, body)
        except Exception as e:
            return self._json(502, {"ok": False, "error": str(e)[:200]})
        dt = time.time() - t0
        t = out.get("timings", {}) if isinstance(out, dict) else {}
        with open(LOG, "a") as f:
            f.write("%s MISS dt=%.1fs decode=%s\n" % (time.strftime("%H:%M:%S"), dt, t.get("predicted_per_second")))
        with LOCK:
            if len(CACHE) < 512:
                CACHE[k] = out
        return self._json(200, out)

if __name__ == "__main__":
    print("g5-infer gateway on 127.0.0.1:%d -> %s" % (PORT, UP), flush=True)
    H(("127.0.0.1", PORT), R).serve_forever()
