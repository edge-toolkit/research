#!/usr/bin/env python3
"""Static file server for the GPU fingerprint tool, with a registration endpoint.

Serves the files in this directory, a paste-and-register form at /submit, and a POST /register endpoint that
appends new device signatures to registry.yaml so a device recognised once stays recognised. Standard library
only; runs over HTTP, or over HTTPS when given a certificate and key (the WebGPU path off localhost needs a
secure context).

Examples:
    python3 server.py                                   # http://127.0.0.1:8099
    python3 server.py --bind 0.0.0.0 --port 8100        # http on the LAN
    python3 server.py --bind 0.0.0.0 --port 8443 --cert cert.pem --key key.pem   # https on the LAN
"""

import argparse
import json
import os
import re
import ssl
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry.yaml")
LOCK = threading.Lock()

# The page decides which fingerprinting mechanisms exist: an entry's interface is whatever name the page files a
# result under, and its results are whatever names and values that mechanism reports, so a mechanism added to the
# page needs no change here. What is checked is only that each name and value is a plain token, which is what keeps
# a request from writing anything but one well-formed block into the registry.
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
VALUE = re.compile(r"^[0-9A-Za-z?_.+-]{1,128}$")
MAX_RESULTS = 256

# A run is timed as the first since the page loaded or as a repeat, and the two are kept apart: a browser keeps
# compiled shaders, so a repeat is far quicker. Each kind has the fastest and slowest time seen, in milliseconds.
RUNS = ("first", "repeat")
TIMING_KEYS = tuple(f"{run}_ms_{bound}" for run in RUNS for bound in ("min", "max"))

# A minimal paste-and-register page, served at /submit. It exists only here, so it is never served by a static
# host; paste one or more registry lines (as the tool prints for a new device) and it posts each to /register.
SUBMIT_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Submit GPU Fingerprints</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}
h1{font-size:1.3rem} p.lead{color:#666}
textarea{width:100%;box-sizing:border-box;font-size:.82rem;padding:.5rem;
font-family:ui-monospace,Menlo,Consolas,monospace}
button{font:inherit;padding:.4rem .9rem;margin:.5rem 0;cursor:pointer}
.row{font-size:.85rem;padding:.15rem 0;border-bottom:1px solid #8884}
.ok{color:#2a7}.dup{color:#888}.bad{color:#b23}.summary{margin-top:.5rem;font-weight:600}
a{color:inherit}
</style></head><body>
<h1>Submit GPU Fingerprints</h1>
<p class="lead">Paste one or more registry lines (JSON, one per line) copied from the fingerprint tool, then
register them. Useful when the tool runs in a browser that cannot reach this server directly (for example Tor
Browser on a phone viewing the GitHub Pages copy): copy the line there and paste it here.</p>
<textarea id="in" rows="8"
placeholder='{"interface":"webgpu","label":"...","elapsed_ms":12.3,"results":{"multiply_add":"...","fma":"..."}}'>
</textarea>
<div><button id="go">Register</button></div>
<div id="out"></div>
<p><a href="./">Back to the fingerprint tool</a></p>
<script>
const out = document.getElementById("out");
document.getElementById("go").addEventListener("click", async () => {
out.textContent = "";
const lines = document.getElementById("in").value.split("\\n").map((l) => l.trim()).filter(Boolean);
if (!lines.length) { out.textContent = "Nothing to submit."; return; }
let added = 0, present = 0, failed = 0;
for (const line of lines) {
const row = document.createElement("div");
row.className = "row";
let entry;
try {
entry = JSON.parse(line);
} catch (e) {
row.textContent = "invalid JSON: " + line.slice(0, 70);
row.className = "row bad";
out.append(row);
failed++;
continue;
}
try {
const opts = {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(entry)};
const res = await fetch("/register", opts);
const reply = await res.json().catch(() => ({}));
if (res.ok && reply.ok) {
row.textContent = "added: " + (entry.label || "(no label)");
row.className = "row ok";
added++;
} else if (/already present/i.test(reply.message || "")) {
row.textContent = reply.message;
row.className = "row dup";
present++;
} else {
row.textContent = reply.message || ("HTTP " + res.status);
row.className = "row bad";
failed++;
}
} catch (e) {
row.textContent = "request failed: " + (e.message || e);
row.className = "row bad";
failed++;
}
out.append(row);
}
const s = document.createElement("div");
s.className = "summary";
s.textContent = "added " + added + ", already present " + present + ", failed " + failed;
out.append(s);
});
</script></body></html>
"""


def _yaml_scalar(text):
    """Decode a scalar written by format_entry: trimmed, with surrounding double quotes and escapes removed."""
    s = text.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return s


def _num(text):
    """Parse a numeric scalar (an elapsed-range bound) to a float, or None if blank or not a number."""
    try:
        return float(_yaml_scalar(text))
    except ValueError:
        return None


def _fmt_ms(value):
    """Format a millisecond value without a trailing '.0' (e.g. 14.8 or 14)."""
    return f"{value:g}"


def read_registry():
    """Read registry.yaml into a list of {interface, label, results} dicts, each with any of TIMING_KEYS it has.

    Only the fixed block shape format_entry writes is parsed, not YAML in general, so the server needs no
    third-party YAML library. Lines that do not fit the shape are ignored.
    """
    entries = []
    current = None
    in_results = False
    if not os.path.exists(REGISTRY):
        return entries
    with open(REGISTRY, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith("- interface:"):
                current = {"interface": _yaml_scalar(line.split(":", 1)[1]), "label": "", "results": {}}
                entries.append(current)
                in_results = False
            elif current is None:
                continue
            elif in_results and line.startswith("    ") and ":" in stripped:
                # A result is told from an entry's own fields by its deeper indent, not by its name, so a mechanism
                # is free to call a result "label" or "reason".
                key, value = stripped.split(":", 1)
                current["results"][_yaml_scalar(key)] = _yaml_scalar(value)
            elif stripped.startswith("label:"):
                current["label"] = _yaml_scalar(stripped.split(":", 1)[1])
                in_results = False
            elif stripped.startswith("window:"):
                current["window"] = _yaml_scalar(stripped.split(":", 1)[1])
                in_results = False
            elif stripped.split(":", 1)[0] in TIMING_KEYS:
                key, value = stripped.split(":", 1)
                current[key] = _num(value)
                in_results = False
            elif stripped.startswith("blocked:"):
                current["blocked"] = _yaml_scalar(stripped.split(":", 1)[1]).lower() == "true"
                in_results = False
            elif stripped.startswith("reason:"):
                current["reason"] = _yaml_scalar(stripped.split(":", 1)[1])
                in_results = False
            elif stripped == "results:":
                in_results = True
    return entries


def _window(entry):
    """The kind of window an entry was taken in, normal or private; an entry that does not say is normal.

    A private window can change what a browser returns -- some add noise to canvas reads only there -- so its
    results are kept in entries of their own and never folded into a normal window's.
    """
    return "private" if entry.get("window") == "private" else "normal"


def _quote(text):
    """Double-quote a scalar for the YAML block, escaping backslashes and quotes so _yaml_scalar reverses it."""
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def format_entry(entry):
    """Serialize one registry entry as a YAML block (a signature with results, or a blocked interface with a reason)."""
    lines = [f"- interface: {entry['interface']}", f"  label: {_quote(entry.get('label', ''))}"]
    if _window(entry) == "private":
        lines.append("  window: private")
    if entry.get("blocked"):
        lines.append("  blocked: true")
        lines.append(f"  reason: {_quote(entry.get('reason', ''))}")
        return "\n".join(lines) + "\n"
    lines.extend(f"  {key}: {_fmt_ms(entry[key])}" for key in TIMING_KEYS if entry.get(key) is not None)
    lines.append("  results:")
    lines.extend(f'    {key}: "{value}"' for key, value in entry["results"].items())
    return "\n".join(lines) + "\n"


def _read_header():
    """Return the leading comment lines of registry.yaml (everything before the first entry)."""
    if not os.path.exists(REGISTRY):
        return ""
    header = []
    with open(REGISTRY, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("- interface:"):
                break
            header.append(line)
    return "".join(header)


def _write_registry(entries):
    """Rewrite registry.yaml from entries, preserving the header comment. Used only to update a known entry."""
    header = _read_header()
    with open(REGISTRY, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.writelines(format_entry(entry) for entry in entries)


def add_entry(entry):
    """Add a signature to registry.yaml, or fold a repeat of a known one into that entry's timing range.

    A new signature is appended as a block. A repeat (its results map already present for this interface) is not
    duplicated; its measured elapsed_ms, if given, only widens that entry's range for the kind of run it was, first
    or repeat. Returns (ok, message); validation is strict so a stray request cannot corrupt the registry.
    """
    iface = entry.get("interface")
    results = entry.get("results")
    label = str(entry.get("label") or "").strip()[:200]
    elapsed = entry.get("elapsed_ms")
    run = entry.get("run")
    if not isinstance(iface, str) or not NAME.match(iface):
        return False, "interface must be a name of letters, digits, '_', '.' and '-', starting with a letter"
    if entry.get("window") not in (None, "normal", "private"):
        return False, "window must be 'normal' or 'private'"
    window = _window(entry)
    where = " in a private window" if window == "private" else ""
    if entry.get("blocked"):
        if not label:
            return False, "label must not be empty"
        reason = str(entry.get("reason") or "").strip()[:200]
        with LOCK:
            entries = read_registry()
            for row in entries:
                same = row.get("interface") == iface and _window(row) == window and row.get("label") == label
                if same and row.get("blocked"):
                    return True, f"already recorded as blocked{where}: {label}"
            new_row = {"interface": iface, "label": label, "window": window, "blocked": True, "reason": reason}
            with open(REGISTRY, "a", encoding="utf-8") as handle:
                handle.write(format_entry(new_row))
            return True, f"recorded {iface} as blocked{where}; the registry now holds {len(entries) + 1} entries"
    if not isinstance(results, dict) or not results or len(results) > MAX_RESULTS:
        return False, f"results must map 1 to {MAX_RESULTS} result names to values"
    if not all(
        isinstance(k, str) and NAME.match(k) and isinstance(v, str) and VALUE.match(v) for k, v in results.items()
    ):
        return False, "each result needs a name starting with a letter and a value of letters, digits and '?_.+-'"
    if not label:
        return False, "label must not be empty"
    if elapsed is not None and not isinstance(elapsed, (int, float)):
        return False, "elapsed_ms must be a number"
    if elapsed is not None and run not in RUNS:
        return False, "run must be 'first' or 'repeat' when elapsed_ms is given"
    low_key, high_key = f"{run}_ms_min", f"{run}_ms_max"
    with LOCK:
        entries = read_registry()
        for row in entries:
            if row.get("interface") == iface and _window(row) == window and row.get("results") == results:
                name = str(row.get("label", ""))
                if elapsed is None:
                    return True, f"already present (no timing given): {name}"
                lo = row.get(low_key)
                hi = row.get(high_key)
                new_lo = elapsed if lo is None else min(lo, elapsed)
                new_hi = elapsed if hi is None else max(hi, elapsed)
                took = f"{run} run of {_fmt_ms(elapsed)} ms"
                if new_lo == lo and new_hi == hi:
                    return True, f"already present; {took} is within {_fmt_ms(lo)}-{_fmt_ms(hi)} ms: {name}"
                row[low_key] = float(new_lo)
                row[high_key] = float(new_hi)
                _write_registry(entries)
                return True, f"{took} set the {run}-run range to {_fmt_ms(new_lo)}-{_fmt_ms(new_hi)} ms: {name}"
        new_row = {"interface": iface, "label": label, "window": window, "results": results}
        if elapsed is not None:
            new_row[low_key] = float(elapsed)
            new_row[high_key] = float(elapsed)
        with open(REGISTRY, "a", encoding="utf-8") as handle:
            handle.write(format_entry(new_row))
        return True, f"added{where}; the registry now holds {len(entries) + 1} entries"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def end_headers(self):
        # The registry changes during collection, so never let a probe or the data file be cached.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0].rstrip("/") == "/submit":
            body = SUBMIT_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _reply(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/register":
            self._reply(404, {"ok": False, "message": "no such endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            entry = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as error:
            self._reply(400, {"ok": False, "message": f"bad JSON: {error}"})
            return
        ok, message = add_entry(entry)
        # The access log shows only the request line. This records what was submitted and what became of it; for
        # a repeat of a known signature the message names the registry row it matched.
        submitted = f"{entry.get('interface')} {str(entry.get('label') or '').strip()!r}"
        print(f"{self.address_string()} - submitted {submitted}: {message}", flush=True)
        self._reply(200 if ok else 409, {"ok": ok, "message": message})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}", flush=True)


class Server(ThreadingHTTPServer):
    """Threading HTTPS server that logs a client's TLS or connection failure as one line, not a traceback.

    A browser that refuses the self-signed certificate sends a ``bad certificate`` alert, and the default server
    prints a full traceback for it -- which reads like a crash even though the server keeps running fine.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, ssl.SSLError):
            reason = getattr(error, "reason", None) or error
            print(f"{client_address[0]} - TLS rejected by client ({reason})", flush=True)
            return
        if isinstance(error, (ConnectionResetError, BrokenPipeError, TimeoutError, ConnectionAbortedError)):
            print(f"{client_address[0]} - connection dropped ({type(error).__name__})", flush=True)
            return
        super().handle_error(request, client_address)


def main():
    parser = argparse.ArgumentParser(description="GPU fingerprint server with a /register endpoint")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--cert", help="TLS certificate (PEM); enables HTTPS when given with --key")
    parser.add_argument("--key", help="TLS private key (PEM)")
    parser.add_argument("--public-host", help="hostname to also print in the ready message, e.g. a nip.io name")
    args = parser.parse_args()

    server = Server((args.bind, args.port), Handler)
    scheme = "http"
    if args.cert and args.key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    browse = "localhost" if args.bind in {"0.0.0.0", "::", "127.0.0.1", "localhost"} else args.bind
    print(f"serving {scheme} from {HERE}", flush=True)
    print(f"point your browser at  {scheme}://{browse}:{args.port}/", flush=True)
    if args.public_host:
        print(f"from another device:   {scheme}://{args.public_host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
