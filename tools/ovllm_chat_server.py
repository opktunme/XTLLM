#!/usr/bin/env python3
"""Root-independent localhost chat UI for every packaged XTLLM backend."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__MODEL__ · XTLLM</title><style>
:root{color-scheme:dark;--bg:#0c0f14;--panel:#151a22;--line:#293242;--muted:#98a5b8;--accent:#7c9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f8;font:15px/1.5 system-ui,sans-serif}
.app{max-width:940px;height:100vh;margin:auto;display:flex;flex-direction:column;padding:16px}header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:6px 2px 14px}
h1{font-size:18px;margin:0}.sub,.status{color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#41d17d;margin-right:7px}
#chat{flex:1;overflow:auto;padding:20px 0}.msg{max-width:84%;white-space:pre-wrap;padding:12px 15px;border:1px solid var(--line);border-radius:14px;margin:0 0 14px}.user{margin-left:auto;background:#25345f}.assistant{background:var(--panel)}.role{color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px}
.controls{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:12px;margin:0 0 8px}.controls select{background:var(--panel);color:#fff;border:1px solid var(--line);padding:5px;border-radius:7px}.composer{display:flex;gap:10px}textarea{flex:1;resize:none;min-height:68px;max-height:180px;background:var(--panel);color:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit;outline:none}textarea:focus{border-color:var(--accent)}button{border:0;border-radius:10px;padding:0 18px;background:var(--accent);color:#081020;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:wait}.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);padding:5px 10px;font-weight:500}.status{height:25px;padding-top:5px}.error{color:#ff8f8f}.metrics{color:#b5c3d8;font-size:12px;margin-top:9px;padding-top:8px;border-top:1px solid var(--line)}
</style></head><body><div class="app"><header><div><h1>__MODEL__</h1><div class="sub">Native model-specific Vulkan backend · automatic RAM/VRAM policy</div></div><div class="sub"><span class="dot"></span>localhost</div></header>
<main id="chat"><div class="msg assistant"><div class="role">System</div>Ready. Each reply uses the native XTLLM engine. Recent visible turns are supplied as context.</div></main>
<div class="controls"><label>Max output <select id="tokens"><option>64</option><option selected>128</option><option>256</option><option>512</option></select></label><button class="ghost" id="clear">Clear</button></div>
<div class="composer"><textarea id="prompt" placeholder="Message the model… (Ctrl+Enter to send)"></textarea><button id="send">Send</button></div><div class="status" id="status"></div></div>
<script>const chat=document.querySelector('#chat'),prompt=document.querySelector('#prompt'),send=document.querySelector('#send'),status=document.querySelector('#status');let history=[];
function add(role,text,metrics=''){const box=document.createElement('div');box.className='msg '+role;const r=document.createElement('div');r.className='role';r.textContent=role;const t=document.createElement('div');t.textContent=text;box.append(r,t);if(metrics){const m=document.createElement('div');m.className='metrics';m.textContent=metrics;box.append(m)}chat.append(box);chat.scrollTop=chat.scrollHeight}
async function submit(){const text=prompt.value.trim();if(!text||send.disabled)return;history.push({role:'user',content:text});add('user',text);prompt.value='';send.disabled=true;status.className='status';status.textContent='Running the Vulkan model…';try{const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages:history,max_tokens:+document.querySelector('#tokens').value})});const data=await res.json();if(!res.ok)throw new Error(data.error||'Request failed');history.push({role:'assistant',content:data.response});add('assistant',data.response,data.metrics||'');status.textContent='Ready'}catch(e){status.className='status error';status.textContent=e.message}finally{send.disabled=false;prompt.focus()}}
send.onclick=submit;prompt.onkeydown=e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();submit()}};document.querySelector('#clear').onclick=()=>{history=[];chat.innerHTML='';status.textContent='Conversation cleared';prompt.focus()};prompt.focus();</script></body></html>"""

CHAT_MAGIC = b"XTCHAT1\0"
CHAT_REFERENCE_PREFIX = "@XTLLM_CHAT_FILE:"


def write_chat_transcript(path: Path, messages: list[tuple[str, str]]) -> None:
    with path.open("wb") as output:
        output.write(CHAT_MAGIC)
        output.write(struct.pack("<I", len(messages)))
        for role, content in messages:
            encoded = content.encode("utf-8")
            output.write(struct.pack("<BI", 1 if role == "user" else 2,
                                     len(encoded)))
            output.write(encoded)


class State:
    def __init__(self, engine: Path, runtime: Path, model_name: str,
                 engine_options: list[str], engine_environment: dict[str, str],
                 standalone: bool, timeout: int, default_tokens: int):
        self.engine = engine
        self.runtime = runtime
        self.model_name = model_name
        self.engine_options = engine_options
        self.engine_environment = engine_environment
        self.standalone = standalone
        self.timeout = timeout
        self.default_tokens = default_tokens
        self.lock = threading.Lock()

    @staticmethod
    def conversation_messages(messages: list[dict]) -> list[tuple[str, str]]:
        clean: list[tuple[str, str]] = []
        # Requests end with a user turn, so retain an odd number of turns to
        # preserve complete user/assistant pairs and a leading user boundary.
        for item in messages[-9:]:
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                clean.append((role, content[:24000]))
        if not clean or clean[-1][0] != "user":
            raise ValueError("A user message is required")
        if any(role != ("user" if index % 2 == 0 else "assistant")
               for index, (role, _) in enumerate(clean)):
            raise ValueError("Conversation roles must alternate from user")
        return clean

    @staticmethod
    def generated_text(output: str) -> str:
        starts = list(re.finditer(r"RAM cache top-off:[^\r\n]*\r?\n", output))
        if not starts:
            raise RuntimeError("Could not identify generated text in engine output")
        start = starts[-1].end()
        endings = [output.find(marker, start) for marker in
                   ("\ntoken ids:", "\nmodel:", "\ndecode throughput:")]
        endings = [position for position in endings if position >= start]
        if not endings:
            raise RuntimeError("Could not identify generated text in engine output")
        return output[start:min(endings)].strip()

    def infer(self, messages: list[dict], requested_tokens: int) -> dict[str, str]:
        transcript = self.conversation_messages(messages)
        tokens = max(8, min(int(requested_tokens or self.default_tokens), 512))
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("QWEN_", "QWEN36_", "QWEN38_",
                                              "LONGCAT_", "NEMOTRON3_", "DSV4_"))}
        environment.update(self.engine_environment)
        temporary = self.runtime.parent / "tmp"
        temporary.mkdir(exist_ok=True)
        environment["TEMP"] = environment["TMP"] = str(temporary)
        environment["XTLLM_STRUCTURED_CHAT"] = "1"
        handle, transcript_name = tempfile.mkstemp(
            prefix="xtllm-chat-", suffix=".bin", dir=temporary)
        os.close(handle)
        transcript_path = Path(transcript_name)
        write_chat_transcript(transcript_path, transcript)
        prompt = CHAT_REFERENCE_PREFIX + str(transcript_path)
        command = ([str(self.engine), str(self.runtime), prompt, str(tokens)]
                   if self.standalone else
                   [str(self.engine), *self.engine_options, "--tokens", str(tokens),
                    str(self.runtime), prompt])
        try:
            with self.lock:
                completed = subprocess.run(
                    command, cwd=str(self.engine.parent), env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding="utf-8", errors="replace", timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            transcript_path.unlink(missing_ok=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-4000:].strip() or
                               f"inference exited with code {completed.returncode}")
        output = completed.stdout
        pieces: list[str] = []
        for pattern, suffix in (
            (r"decode throughput:\s*([0-9.]+) tok/s", " tok/s"),
            (r"acquisition ms per output:\s*([0-9.]+)", " ms acquisition/token"),
            (r"peak device allocation estimate GiB:\s*([0-9.]+)", " GiB VRAM"),
        ):
            match = re.search(pattern, output)
            if match:
                pieces.append(match.group(1) + suffix)
        return {"response": self.generated_text(output), "metrics": " · ".join(pieces)}


class Handler(BaseHTTPRequestHandler):
    server_version = "XTLLMChat/1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: dict) -> None:
        self.send_bytes(status, "application/json; charset=utf-8",
                        json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/":
            name = html.escape(self.server.state.model_name)
            self.send_bytes(200, "text/html; charset=utf-8",
                            PAGE.replace("__MODEL__", name).encode("utf-8"))
        elif self.path == "/api/health":
            self.send_json(200, {"ok": True, "model": self.server.state.model_name})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 524288:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            self.send_json(200, self.server.state.infer(
                request.get("messages", []), request.get("max_tokens", 128)))
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "Generation exceeded the configured timeout"})
        except Exception as error:
            self.send_json(400, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--engine-option", action="append", default=[])
    parser.add_argument("--engine-env", action="append", default=[])
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    engine, runtime = args.engine.resolve(), args.runtime.resolve()
    if not engine.is_file() or not runtime.is_dir():
        raise SystemExit("Engine or runtime directory is missing")
    engine_environment: dict[str, str] = {}
    for item in args.engine_env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"Invalid --engine-env value: {item}")
        engine_environment[key] = value
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = State(engine, runtime, args.model_name, args.engine_option,
                         engine_environment, args.standalone,
                         args.timeout, args.tokens)
    url = f"http://{args.host}:{args.port}"
    print(f"XTLLM chat: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
