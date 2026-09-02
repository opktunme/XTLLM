#!/usr/bin/env python3
"""Minimal localhost chat for the experimental DeepSeek Fast Vulkan fork."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile

from ovllm_chat_server import (CHAT_REFERENCE_PREFIX,
                               State as CanonicalChatState,
                               write_chat_transcript)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek V4 Flash · Fast Long Context</title>
<style>
:root{color-scheme:dark;--bg:#0c0f14;--panel:#151a22;--line:#293242;--muted:#98a5b8;--accent:#7c9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f8;font:15px/1.5 system-ui,sans-serif}
.app{max-width:940px;height:100vh;margin:auto;display:flex;flex-direction:column;padding:16px}
header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:6px 2px 14px}
h1{font-size:18px;margin:0}.sub{color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#41d17d;margin-right:7px}
#chat{flex:1;overflow:auto;padding:20px 0}.msg{max-width:84%;white-space:pre-wrap;padding:12px 15px;border:1px solid var(--line);border-radius:14px;margin:0 0 14px}
.user{margin-left:auto;background:#25345f}.assistant{background:var(--panel)}.role{color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px}
.controls{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:12px;margin:0 0 8px}.controls select{background:var(--panel);color:#fff;border:1px solid var(--line);padding:5px;border-radius:7px}
.composer{display:flex;gap:10px}textarea{flex:1;resize:none;min-height:68px;max-height:180px;background:var(--panel);color:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit;outline:none}
textarea:focus{border-color:var(--accent)}button{border:0;border-radius:10px;padding:0 18px;background:var(--accent);color:#081020;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:wait}
.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);padding:5px 10px;font-weight:500}.status{height:25px;color:var(--muted);font-size:12px;padding-top:5px}.error{color:#ff8f8f}.metrics{color:#b5c3d8;font-size:12px;margin-top:9px;padding-top:8px;border-top:1px solid var(--line)}
</style></head><body><div class="app">
<header><div><h1>DeepSeek V4 Flash 0731</h1><div class="sub">Experimental Fast long-context fork · Q4G64T · 16 GiB expert profile · 1M capacity</div></div><div class="sub"><span class="dot"></span>localhost</div></header>
<main id="chat"><div class="msg assistant"><div class="role">System</div>Ready. Each message launches a fresh isolated Fast-mode Vulkan process. Recent visible turns are supplied as conversation context; model initialization is included in wall time.</div></main>
<div class="controls"><label>Max output <select id="tokens"><option selected>64</option><option>128</option><option>256</option></select></label><button class="ghost" id="clear">Clear</button></div>
<div class="composer"><textarea id="prompt" placeholder="Message DeepSeek… (Ctrl+Enter to send)"></textarea><button id="send">Send</button></div><div class="status" id="status"></div>
</div><script>
const chat=document.querySelector('#chat'),prompt=document.querySelector('#prompt'),send=document.querySelector('#send'),status=document.querySelector('#status');let history=[];
function add(role,text,metrics=''){const box=document.createElement('div');box.className='msg '+role;const r=document.createElement('div');r.className='role';r.textContent=role;const t=document.createElement('div');t.textContent=text;box.append(r,t);if(metrics){const m=document.createElement('div');m.className='metrics';m.textContent=metrics;box.append(m)}chat.append(box);chat.scrollTop=chat.scrollHeight}
async function submit(){const text=prompt.value.trim();if(!text||send.disabled)return;history.push({role:'user',content:text});add('user',text);prompt.value='';send.disabled=true;status.className='status';status.textContent='Running DeepSeek Fast on Vulkan…';try{const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages:history,max_tokens:+document.querySelector('#tokens').value})});const data=await res.json();if(!res.ok)throw new Error(data.error||'Request failed');history.push({role:'assistant',content:data.response});add('assistant',data.response,data.metrics||'');status.textContent='Ready'}catch(e){status.className='status error';status.textContent=e.message}finally{send.disabled=false;prompt.focus()}}
send.onclick=submit;prompt.onkeydown=e=>{if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();submit()}};document.querySelector('#clear').onclick=()=>{history=[];chat.innerHTML='';status.textContent='Conversation cleared';prompt.focus()};prompt.focus();
</script></body></html>"""


class State:
    def __init__(self, executable: Path, runtime: Path, timeout: int):
        self.executable = executable
        self.runtime = runtime
        self.timeout = timeout
        self.lock = threading.Lock()

    conversation_messages = staticmethod(CanonicalChatState.conversation_messages)

    def infer(self, messages: list[dict], max_tokens: int) -> dict:
        transcript = self.conversation_messages(messages)
        max_tokens = max(8, min(int(max_tokens), 256))
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("DSV4_")}
        root = self.executable.parents[1]
        env["TEMP"] = str(root / "tmp")
        env["TMP"] = env["TEMP"]
        Path(env["TEMP"]).mkdir(exist_ok=True)
        env["XTLLM_STRUCTURED_CHAT"] = "1"
        handle, transcript_name = tempfile.mkstemp(
            prefix="xtllm-chat-", suffix=".bin", dir=env["TEMP"])
        os.close(handle)
        transcript_path = Path(transcript_name)
        write_chat_transcript(transcript_path, transcript)
        prompt = CHAT_REFERENCE_PREFIX + str(transcript_path)
        command = [
            str(self.executable), "--ram-gib", "16", "--context-gib", "7",
            "--context-tokens", "1048576", "--long-mode", "fast",
            str(self.runtime), prompt, str(max_tokens),
        ]
        try:
            with self.lock:
                completed = subprocess.run(
                    command, cwd=str(root), env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                    timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        finally:
            transcript_path.unlink(missing_ok=True)
        output = completed.stdout
        if completed.returncode:
            raise RuntimeError(output[-3000:].strip() or
                               f"inference exited with code {completed.returncode}")
        starts = list(re.finditer(r"RAM cache top-off:[^\r\n]*\r?\n", output))
        end = output.rfind("\ntoken ids:")
        if not starts or end < starts[-1].end():
            raise RuntimeError("Could not identify the generated response")
        response = output[starts[-1].end():end].strip()
        speed = re.search(r"decode throughput:\s*([0-9.]+) tok/s", output)
        passes = re.search(r"decode passes:\s*(\d+)", output)
        times = re.search(
            r"attention\+router / acquisition / expert\+shared: "
            r"([0-9.]+) / ([0-9.]+) / ([0-9.]+) s", output)
        peak = re.search(r"peak Vulkan allocations:\s*([0-9.]+) GiB", output)
        host = re.search(r"Vulkan/plain host backing / plain-cache mode: "
                         r"([0-9.]+) /", output)
        pieces = ["Fast mode", "1M context capacity"]
        if speed:
            pieces.insert(0, f"{speed.group(1)} tok/s")
        if times and passes and int(passes.group(1)):
            acquisition_ms = 1000.0 * float(times.group(2)) / int(passes.group(1))
            pieces.append(f"{acquisition_ms:.1f} ms acquisition/token")
        if peak and host:
            device = float(peak.group(1)) - float(host.group(1))
            pieces.append(f"~{device:.2f} GiB VRAM")
        return {"response": response, "metrics": " · ".join(pieces)}


class Handler(BaseHTTPRequestHandler):
    server_version = "DeepSeekFastVulkanChat/1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def send_value(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: dict) -> None:
        self.send_value(status, "application/json; charset=utf-8",
                        json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_value(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/api/health":
            self.send_json(200, {"ok": True, "model": "DeepSeek-V4-Flash-0731",
                                 "mode": "fast"})
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
                request.get("messages", []), request.get("max_tokens", 64)))
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "Generation exceeded the server timeout"})
        except Exception as error:
            self.send_json(400, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    executable = root / "build" / "xtllm.exe"
    runtime = root / "models" / "deepseek-v4-flash-0731" / "runtime"
    for required in (executable, runtime / "model-q4g64.ovs",
                     runtime / "experts-q4g64.ovx", runtime / "tokenizer.ovb"):
        if not required.exists():
            raise SystemExit(f"Required file is missing: {required}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = State(executable, runtime, args.timeout)
    print(f"DeepSeek Fast Vulkan chat: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
