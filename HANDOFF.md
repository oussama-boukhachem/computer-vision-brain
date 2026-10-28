# cv-lab — handoff notes

Read this first. It exists so a fresh session does not have to rediscover the
layout, the numbers, or the traps below.

## What this is

A local computer-vision workspace. One web page shows a live camera feed with
person tracking, hand landmarks, held-gesture detection, facial expression,
metric distance and a 3D point cloud. A local vision-language model narrates
what it sees. Nothing leaves the machine.

## Runtime facts

| Item | Value |
|---|---|
| Host | The-Black-Pearl (Linux), user \`amesso\` |
| GPU | RTX 4060 Laptop, 8 GB VRAM (shared with Ollama) |
| Python | 3.14.4, venv in the folder the code was developed in |
| Service | \`cv-lab-webui.service\` (systemd --user), port 8099, loopback only |
| Camera | /dev/video0, exclusive: one process at a time |
| VLM | Ollama, \`qwen2.5vl:3b\` at 127.0.0.1:11434 |

## Files

| File | Role |
|---|---|
| \`webui.py\` | HTTP server, capture loop, all workers, the page itself |
| \`gesture.py\` | MediaPipe hand landmarks, finger states, hold timer |
| \`emotion.py\` | MediaPipe face blendshapes to expression scores |
| \`depth.py\` | Metric depth, point sampling, point cloud |
| \`setup.sh\` | venv + torch + ultralytics + opencv |
| \`svc.sh\` | start/stop/status wrapper around systemctl |
| \`models/\` | hand_landmarker.task, face_landmarker.task |
| \`docs/\` | architecture PDF for humans |

## Data flow

1. \`vision_loop\` reads a frame, runs YOLO tracking, draws boxes.
2. Every 3rd frame it keeps an overlay-free copy in \`STATE["raw"]\`.
3. Hands run every other frame on a 0.6-scaled copy; landmarks are scaled back.
4. Expression runs on the clean copy when enabled.
5. The annotated frame is JPEG-encoded into \`STATE["frame"]\`; \`/stream\` pushes it.
6. \`depth_loop\` watches \`STATE["raw"]\` and only works when a depth feature is on.
7. \`vlm_loop\` narrates on motion, and stays silent while depth owns the GPU.

Shared state lives in one dict guarded by \`LOCK\`. Workers are plain threads.

## Endpoints

| Path | Purpose |
|---|---|
| \`/\` | dashboard |
| \`/stream\` | MJPEG |
| \`/snapshot\` | single annotated JPEG |
| \`/events\` | JSON: stats, feeds, features, objects, gesture, emotion |
| \`/measure?x=&y=\` | distance in metres at normalised coordinates |
| \`/depthmap\` | depth heatmap JPEG |
| \`/pointcloud\` | binary float32 x,y,z,r,g,b |
| \`POST /api/toggle\` | \`{"f":"depth|cloud|emotion|objects","on":bool}\` |
| \`POST /api/run\` | \`{"on":bool}\` — start/stop, releases the camera |

## Measured performance

| Stage | Cost |
|---|---|
| YOLO11n tracking | 5-6 ms |
| Hand landmarks (0.6 scale, every 2nd frame) | 10-15 ms |
| Expression | ~10 ms, CPU |
| Metric depth | ~300 ms |
| VLM narration | 1.6-1.9 s warm |
| End to end with hands on | ~40 fps; tracking only ~60 fps |

## Design decisions

- **Person-only by default.** \`classes=[0]\` keeps the display readable and is
  cheaper. The Objects toggle drops the filter to all 80 COCO classes.
- **One GPU lane.** Depth and the VLM never run together; \`vlm_loop\` checks
  \`features\` and \`running\` before speaking.
- **Lazy depth.** The engine is constructed on first use because loading costs
  about 16 s. It is never unloaded, to avoid paying that again.
- **Metric depth, not relative.** The model is fine-tuned on indoor scenes, so it
  returns metres and needs no calibration.
- **Neutral floor in emotion.** Weak blendshape signals must read neutral, not
  angry; the floor is deliberate.
- **Stop releases the camera.** It is not a pause: \`cap.release()\` frees the
  device so other software can use it.

## Traps already hit

1. **Python 3.14 wheels.** The \`cu126\` index has no 3.14 builds. Plain PyPI
   resolves CUDA 13 correctly. Do not "fix" it back to cu126.
2. **Processes die with their shell.** Even \`setsid\` was killed by the exec
   runner. Run the service under systemd --user, not from a shell job.
3. **\`pkill -f webui.py\` kills its own shell**, because the pattern matches the
   command line doing the killing. Use the pidfile or systemctl.
4. **Headless screenshots do not composite canvas.** A blank 3D view in a
   screenshot is not a rendering bug. Verify with \`getImageData\` instead.
5. **Windows are not measurable.** Monocular depth invents a surface behind glass.
   Do not present window distance as fact.
6. **MediaPipe 1.x has no \`solutions\` API.** Use \`mediapipe.tasks.python.vision\`.
7. **Camera is exclusive.** If capture fails, something else holds /dev/video0.

## How to run

\`\`\`bash
cd /home/amesso/projects/cv-lab
bash setup.sh                 # once, creates .venv and installs models
./svc.sh start                # or: python webui.py --port 8099
\`\`\`

The systemd unit currently points at the original working copy. To switch it to
this folder, edit \`ExecStart\` and \`WorkingDirectory\` in
\`~/.config/systemd/user/cv-lab-webui.service\`, then \`systemctl --user daemon-reload\`.

## Next steps

- Record button: write annotated clips to disk.
- Custom detector: label 20-50 frames per class, then \`yolo train\`.
- Multi-sensor rig: three cameras, depth sensor, ultrasonics, mic, speaker.
  Expect clock sync and calibration to be the hard part, not the models.

## Conventions

- Standard library first. \`webui.py\` needs no framework on purpose.
- One shared \`STATE\` dict, one lock. Do not add a second source of truth.
- Every new worker must check \`STATE["running"]\` and release what it holds.
- Keep the page dependency-free: no CDN, no build step.
