#!/usr/bin/env python3
"""Minimal local web chat for the project's native Vulkan Qwen runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5 · AMD Vulkan</title>
<style>
:root{color-scheme:dark;--bg:#0c0f14;--panel:#151a22;--line:#293242;--muted:#98a5b8;--accent:#7c9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f8;font:15px/1.5 system-ui,sans-serif}
.app{max-width:900px;height:100vh;margin:auto;display:flex;flex-direction:column;padding:16px}
header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:6px 2px 14px}
h1{font-size:18px;margin:0}.sub{color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#41d17d;margin-right:7px}
#chat{flex:1;overflow:auto;padding:20px 0}.msg{max-width:82%;white-space:pre-wrap;padding:12px 15px;border:1px solid var(--line);border-radius:14px;margin:0 0 14px}
.user{margin-left:auto;background:#25345f}.assistant{background:var(--panel)}.role{color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px}
.controls{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:12px;margin:0 0 8px}.controls select{background:var(--panel);color:#fff;border:1px solid var(--line);padding:5px;border-radius:7px}
.composer{display:flex;gap:10px}textarea{flex:1;resize:none;min-height:68px;max-height:180px;background:var(--panel);color:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit;outline:none}
textarea:focus{border-color:var(--accent)}button{border:0;border-radius:10px;padding:0 18px;background:var(--accent);color:#081020;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:wait}
.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);padding:5px 10px;font-weight:500}.status{height:25px;color:var(--muted);font-size:12px;padding-top:5px}.error{color:#ff8f8f}.metrics{color:#b5c3d8;font-size:12px;margin-top:9px;padding-top:8px;border-top:1px solid var(--line)}
</style>
</head>
<body><div class="app">
<header><div><h1>Qwen3.5-122B-A10B</h1><div class="sub">Our AMD/Vulkan engine · Q4G64T · 16 GiB RAM profile</div></div><div class="sub"><span class="dot"></span>localhost</div></header>
<main id="chat"><div class="msg assistant"><div class="role">System</div>Ready. The first reply includes model initialization time; subsequent messages currently launch a fresh isolated inference process. Recent visible turns are supplied as conversation context.</div></main>
<div class="controls">
  <label>Max output <select id="tokens"><option>64</option><option selected>128</option><option>256</option><option>512</option></select></label>
  <label><input id="thinking" type="checkbox"> Thinking mode</label>
  <button class="ghost" id="clear">Clear</button>
</div>
<div class="composer"><textarea id="prompt" placeholder="Message Qwen… (Ctrl+Enter to send)"></textarea><button id="send">Send</button></div>
<div class="status" id="status"></div>
</div>
<script>
const chat=document.querySelector('#chat'), prompt=document.querySelector('#prompt'), send=document.querySelector('#send'), status=document.querySelector('#status');
let history=[];
function add(role,text,metrics=''){const box=document.createElement('div');box.className='msg '+role;const r=document.createElement('div');r.className='role';r.textContent=role;const t=document.createElement('div');t.textContent=text;box.append(r,t);if(metrics){const m=document.createElement('div');m.className='metrics';m.textContent=metrics;box.append(m)}chat.append(box);chat.scrollTop=chat.scrollHeight}
async function submit(){const text=prompt.value.trim();if(!text||send.disabled)return;history.push({role:'user',content:text});add('user',text);prompt.value='';send.disabled=true;status.className='status';status.textContent='Running the Vulkan model…';
try{const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages:history,max_tokens:+document.querySelector('#tokens').value,thinking:document.querySelector('#thinking').checked})});const data=await res.json();if(!res.ok)throw new Error(data.error||'Request failed');history.push({role:'assistant',content:data.response});add('assistant',data.response,data.metrics||'');status.textContent='Ready'}catch(e){status.className='status error';status.textContent=e.message}finally{send.disabled=false;prompt.focus()}}
send.onclick=submit;prompt.onkeydown=e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();submit()}};document.querySelector('#clear').onclick=()=>{history=[];chat.innerHTML='';status.textContent='Conversation cleared';prompt.focus()};prompt.focus();
</script></body></html>"""


class State:
    def __init__(self, executable: Path, runtime: Path, timeout: int):
        self.executable = executable
        self.runtime = runtime
        self.timeout = timeout
        self.lock = threading.Lock()

    @staticmethod
    def conversation_prompt(messages: list[dict]) -> str:
        clean: list[tuple[str, str]] = []
        for item in messages[-8:]:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                clean.append((role, content[:12000]))
        if not clean:
            raise ValueError("A user message is required")
        if len(clean) == 1 and clean[0][0] == "user":
            return clean[0][1]
        lines = [
            "Continue this conversation. Answer only the final user message, using prior turns as context."
        ]
        for role, content in clean:
            lines.append(f"\n{role.title()}: {content}")
        lines.append("\nAssistant:")
        prompt = "".join(lines)
        # The native runtime has a 2048-token short-context cap. Keep the most
        # recent text when a browser conversation grows unusually large.
        return prompt[-16000:]

    def infer(self, messages: list[dict], max_tokens: int, thinking: bool) -> dict:
        prompt = self.conversation_prompt(messages)
        max_tokens = max(8, min(int(max_tokens), 512))
        env = os.environ.copy()
        env["QWEN_RAM_GIB"] = "16"
        env["QWEN_PROGRESSIVE_EXPERTS"] = "1"
        env["TEMP"] = str(self.runtime.parents[2] / "tmp")
        env["TMP"] = env["TEMP"]
        if thinking:
            env.pop("QWEN_NO_THINK", None)
        else:
            env["QWEN_NO_THINK"] = "1"
        with self.lock:
            completed = None
            selected_slots = 0
            # A browser and Windows compositor consume a variable amount of
            # VRAM.  Use the fastest comfortable profile first, then retry
            # only a clean initialization OOM at progressively safer sizes.
            for slots in (29, 28, 26, 24, 17, 12, 8):
                env["QWEN_DEVICE_SLOTS_PER_LAYER"] = str(slots)
                command = [str(self.executable), str(self.runtime), prompt,
                           str(max_tokens)]
                completed = subprocess.run(
                    command,
                    cwd=str(self.executable.parents[1]),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                selected_slots = slots
                if completed.returncode == 0:
                    break
                if "device-local Vulkan allocation failed" not in completed.stdout:
                    break
        assert completed is not None
        output = completed.stdout
        if completed.returncode:
            tail = output[-3000:].strip()
            raise RuntimeError(tail or f"inference exited with code {completed.returncode}")
        start = re.search(r"expert slots device/RAM:[^\r\n]*\r?\n", output)
        end = output.rfind("\ntoken ids:")
        if not start or end < start.end():
            raise RuntimeError("Could not identify the generated response in runtime output")
        response = output[start.end():end].strip()
        speed = re.search(r"decode throughput:\s*([0-9.]+) tok/s", output)
        memory = re.search(r"peak device allocation estimate GiB:\s*([0-9.]+)", output)
        acquisition = re.search(r"acquisition ms per output:\s*([0-9.]+)", output)
        pieces = []
        if speed:
            pieces.append(f"{speed.group(1)} tok/s")
        if acquisition:
            pieces.append(f"{acquisition.group(1)} ms acquisition/token")
        if memory:
            pieces.append(f"{memory.group(1)} GiB VRAM")
        pieces.append(f"{selected_slots} expert slots/layer")
        return {"response": response, "metrics": " · ".join(pieces)}


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenVulkanChat/1"

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
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/health":
            self.send_json(200, {"ok": True, "model": "Qwen3.5-122B-A10B"})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 262144:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            result = self.server.state.infer(
                request.get("messages", []), request.get("max_tokens", 128),
                bool(request.get("thinking", False)))
            self.send_json(200, result)
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "Generation exceeded the server timeout"})
        except Exception as error:
            self.send_json(400, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    executable = root / "build" / "amd_qwen35_122b_progressive.exe"
    runtime = root / "models" / "qwen3.5-122b-a10b" / "runtime"
    for required in (executable, runtime / "model-q4g64.ovs",
                     runtime / "experts-q4g64.ovx", runtime / "tokenizer.ovb"):
        if not required.exists():
            raise SystemExit(f"Required file is missing: {required}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = State(executable, runtime, args.timeout)
    print(f"Qwen Vulkan chat: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
