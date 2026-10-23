"""Small desktop input device for Arena's pinned G1 record_demos.py.

The simulator, Pink IK, leg controller, success predicate, and HDF5 recorder are
all upstream. This file only turns human browser input into G1's 23-D action.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>G1 desktop teleop</title>
<style>body{font:16px system-ui;background:#141719;color:#eee;margin:24px;max-width:1100px}
img{width:min(100%,960px);background:#333}button{font:inherit;padding:9px;margin:4px}
code{color:#b8e1ff} .row{display:flex;gap:20px;flex-wrap:wrap}</style></head><body>
<h1>G1 human demonstration</h1><p id="status">Connecting…</p>
<img id="view" alt="G1 head camera"><div class="row"><div>
<p><b>Walk:</b> W/S forward/back, A/D left/right, ←/→ turn, Q/E lower/raise body.</p>
<p><b>Arms:</b> 1/2 selects left/right, 3 moves both. I/K forward/back, U/O up/down. J/L moves a single arm sideways, or both arms together/apart. T/G rotates wrists about vertical; with both selected they rotate in opposite directions.</p>
<p><b>Hands:</b> Z toggles left grip, X toggles right grip. R resets attempt.</p>
<p>Hold keys for motion. Release all keys to stop. The action is recorded by Arena.</p>
<button onclick="sendCommand('reset')">Reset attempt</button>
<button onclick="sendCommand('pause')">Pause</button><button onclick="sendCommand('resume')">Resume</button>
</div><pre id="pose"></pre></div>
<script>
const keys=new Set();const held=new Set(['w','s','a','d','arrowleft','arrowright','q','e','i','k','j','l','u','o','t','g']);
async function send(data){try{await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})}catch(e){}}
function sendKeys(){send({keys:[...keys]})}function sendCommand(command){send({command})}
document.addEventListener('keydown',e=>{let k=e.key.toLowerCase();if(document.activeElement.tagName==='BUTTON')return;
 if(held.has(k)){e.preventDefault();keys.add(k);sendKeys()}else if(!e.repeat&&['1','2','3','z','x','r'].includes(k)){e.preventDefault();sendCommand(k)}});
document.addEventListener('keyup',e=>{keys.delete(e.key.toLowerCase());sendKeys()});
window.addEventListener('blur',()=>{keys.clear();sendKeys()});
setInterval(sendKeys,500);
setInterval(async()=>{try{let s=await(await fetch('/state')).json();document.getElementById('status').textContent=s.message;
 document.getElementById('pose').textContent=JSON.stringify({target:s.pose,measured:s.measured},null,2);
 document.getElementById('view').src='/frame.jpg?t='+Date.now()}catch(e){document.getElementById('status').textContent='Simulator disconnected'}},350);
</script></body></html>"""


class DesktopDevice:
    def __init__(self, callbacks, port=8766, *, object_names=("brown_box",), record_mode="arena"):
        self.callbacks = callbacks
        self.object_names = object_names
        self.record_mode = record_mode
        self.step_mode = os.environ.get("I4H_HIL_STEP_MODE") == "1"
        self.step_budget = 0
        self.lock = threading.Lock()
        self.keys = set()
        self.selected = 0
        self.started = record_mode != "i4h"
        self.step = 0
        self.grips = [0.0, 0.0]
        self.height = 0.75
        self.arm = [[0.0] * 3, [0.0] * 3]
        identity_quat = [1.0, 0.0, 0.0, 0.0] if record_mode == "i4h" else [0.0, 0.0, 0.0, 1.0]
        self.quat = [identity_quat[:] for _ in range(2)]
        self.base_quat = [identity_quat[:] for _ in range(2)]
        self.yaw = [0.0, 0.0]
        self.frame = b""
        self.actual = {}
        self.env = None
        self.http = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        print(f"G1 desktop control listening on 127.0.0.1:{port}", flush=True)

    def _handler(self):
        device = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path.startswith("/frame.jpg"):
                    with device.lock:
                        body = device.frame
                    mime = "image/jpeg"
                elif self.path == "/state":
                    with device.lock:
                        state = {"message": f"Simulator step {device.step} · selected arm {['left','right','both'][device.selected]}",
                                 "started": device.started, "step_budget": device.step_budget if device.step_mode else None,
                                 "pose": {"left": device.arm[0], "right": device.arm[1],
                                          "grip": device.grips, "wrist_yaw": device.yaw,
                                          "body_height": round(device.height, 3),
                                          "held_keys": sorted(device.keys)},
                                 "measured": device.actual}
                    body = json.dumps(state).encode()
                    mime = "application/json"
                else:
                    page = PAGE
                    if device.record_mode == "i4h":
                        page = page.replace("G1 human demonstration", "G1 cart transition demonstration")
                        page = page.replace("<button onclick=\"sendCommand('reset')\">Reset attempt</button>",
                                            "<button onclick=\"sendCommand('start')\">Start demonstration</button><button onclick=\"sendCommand('reset')\">Reset attempt</button>")
                        page = page.replace("<button onclick=\"sendCommand('pause')\">Pause</button>",
                                            "<button onclick=\"sendCommand('save')\">Save demonstration</button><button onclick=\"sendCommand('pause')\">Pause</button>")
                    body = page.encode()
                    mime = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/control":
                    self.send_error(404)
                    return
                data = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 4096)))
                with device.lock:
                    if device.step_mode and "steps" in data:
                        device.step_budget += max(0, min(int(data["steps"]), 2000))
                    if "keys" in data:
                        device.keys = set(data["keys"])
                    command = data.get("command")
                    if command == "1":
                        device.selected = 0
                    elif command == "2":
                        device.selected = 1
                    elif command == "3":
                        device.selected = 2
                    elif command == "z":
                        device.grips[0] = 1.0 - device.grips[0]
                    elif command == "x":
                        device.grips[1] = 1.0 - device.grips[1]
                    elif command == "start":
                        device.started = True
                    elif device.record_mode == "i4h" and command == "pause":
                        device.started = False
                        device.keys.clear()
                    elif device.record_mode == "i4h" and command == "resume":
                        device.started = True
                if command in ("r", "reset"):
                    callback = device.callbacks.get("RESET") or device.callbacks.get("R")
                elif command == "save":
                    callback = device.callbacks.get("N")
                elif command == "pause":
                    callback = device.callbacks.get("STOP")
                elif command == "resume":
                    callback = device.callbacks.get("START")
                else:
                    callback = None
                if callback is not None:
                    callback()
                self.send_response(204)
                self.end_headers()

        return Handler

    def add_callback(self, key, callback):
        self.callbacks[key] = callback

    def bind(self, env, *, reset=True):
        self.env = env
        if reset:
            self.reset()

    def reset(self):
        if self.env is None:
            return
        policy = self.env.obs_buf.get("policy", {})
        with self.lock:
            for i, side in enumerate(("left", "right")):
                pose = policy.get(f"{side}_wrist_pose_pelvis_frame")
                if pose is not None:
                    pose = pose[0].detach().cpu().numpy()
                    self.arm[i] = pose[:3, 3].tolist()
                    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                    self.quat[i] = (
                        quat_xyzw[[3, 0, 1, 2]].tolist() if self.record_mode == "i4h" else quat_xyzw.tolist()
                    )
                else:
                    position = policy.get(f"{side}_eef_pos")
                    rotation = policy.get(f"{side}_eef_quat")
                    if position is not None:
                        self.arm[i] = position[0].detach().cpu().tolist()
                    if rotation is not None:
                        self.quat[i] = rotation[0].detach().cpu().tolist()
                self.base_quat[i] = self.quat[i][:]
                self.yaw[i] = 0.0
            self.keys.clear()
            self.grips = [0.0, 0.0]
            self.height = 0.75
            self.step = 0
            self.started = self.record_mode != "i4h"
            self.step_budget = 0

    def wait_for_step(self):
        if not self.step_mode:
            return
        while True:
            with self.lock:
                if self.step_budget > 0:
                    self.step_budget -= 1
                    return
            time.sleep(0.01)

    def advance(self):
        with self.lock:
            if not self.started:
                return None
            k = self.keys
            speed = 0.004
            for index in ((0, 1) if self.selected == 2 else (self.selected,)):
                arm = self.arm[index]
                arm[0] += speed * (('i' in k) - ('k' in k))
                lateral = speed * (('j' in k) - ('l' in k))
                arm[1] += lateral * (-1 if self.selected == 2 and index == 0 else 1)
                arm[2] += speed * (('u' in k) - ('o' in k))
                yaw_step = 0.01 * (('t' in k) - ('g' in k))
                self.yaw[index] += yaw_step * (-1 if self.selected == 2 and index == 0 else 1)
                base_quat = self.base_quat[index]
                if self.record_mode == "i4h":
                    base_quat = [*base_quat[1:], base_quat[0]]
                quat_xyzw = (
                    Rotation.from_euler('z', self.yaw[index]) * Rotation.from_quat(base_quat)
                ).as_quat()
                self.quat[index] = (
                    quat_xyzw[[3, 0, 1, 2]].tolist() if self.record_mode == "i4h" else quat_xyzw.tolist()
                )
            self.height = float(np.clip(self.height + 0.002 * (('e' in k) - ('q' in k)), 0.48, 0.78))
            action = [*self.grips, *self.arm[0], *self.quat[0], *self.arm[1], *self.quat[1],
                      0.4 * (('w' in k) - ('s' in k)), 0.4 * (('a' in k) - ('d' in k)),
                      0.4 * (('arrowleft' in k) - ('arrowright' in k)), self.height, 0.0, 0.0, 0.0]
            return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def publish(self, obs):
        with self.lock:
            self.step += 1
            if self.step != 1 and self.step % 4:
                return
        cameras = obs.get("camera_obs", {})
        frame = cameras.get("robot_head_cam_rgb")
        if frame is None:
            frame = cameras.get("robot_head_cam")
        policy = obs.get("policy", {})
        if frame is None:
            frame = policy.get("robot_head_cam")
        actual = {}
        for side in ("left", "right"):
            pose = policy.get(f"{side}_wrist_pose_pelvis_frame")
            if pose is not None:
                actual[f"{side}_wrist"] = pose[0, :3, 3].detach().cpu().tolist()
            elif policy.get(f"{side}_eef_pos") is not None:
                actual[f"{side}_wrist"] = policy[f"{side}_eef_pos"][0].detach().cpu().tolist()
        for name in self.object_names:
            try:
                actual[f"{name}_world"] = self.env.arena_world.get_position_w(name)[0].detach().cpu().tolist()
            except (AttributeError, KeyError):
                pass
        try:
            actual["robot_world"] = self.env.arena_world.get_position_w("robot")[0].detach().cpu().tolist()
            actual["robot_pose_world"] = self.env.arena_world.get_pose_w("robot")[0].detach().cpu().tolist()
        except (AttributeError, KeyError):
            for name in ("robot", *self.object_names):
                try:
                    data = self.env.scene[name].data
                    actual[f"{name}_world"] = data.root_pos_w[0].detach().cpu().tolist()
                    if name == "robot":
                        actual["robot_quat_world"] = data.root_quat_w[0].detach().cpu().tolist()
                except (AttributeError, KeyError):
                    pass
        try:
            actual["wbc_navigation_command"] = self.env.action_manager.get_term("g1_action")._navigate_cmd[0].tolist()
        except (AttributeError, KeyError):
            pass
        with self.lock:
            self.actual = actual
        if frame is None:
            return
        frame = frame[0].detach().cpu().numpy()
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255 if frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format="JPEG", quality=75)
        with self.lock:
            self.frame = output.getvalue()
