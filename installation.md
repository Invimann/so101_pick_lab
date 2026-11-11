# Installation: SO-101 RL Pick Lab (MuJoCo + Gymnasium + Stable-Baselines3)

This lab does **not** need ROS 2, Gazebo, or Docker. It needs only Python and a few pip packages. Pick **one** of the setups below, then run the checks in [Verify your install](#verify-your-install).

| Setup | Best for | GPU needed? |
|---|---|---|
| A. Local Linux / macOS / Windows (WSL2) | Your own machine | No |
| B. Google Colab | No local install, cloud | No |
| C. Course Docker container (optional) | You already use it for other labs | No |

> **A GPU is not required.** The policy networks are small MLPs, and SB3 is usually faster on CPU for these. MuJoCo itself runs on CPU.

---

## Requirements

- Python **3.10 or newer** (3.11 or 3.12 recommended)
- ~3 GB free disk (mostly PyTorch)
- Internet access for `pip`

---

## A. Local install (Linux, macOS, Windows via WSL2)

> **Windows:** use WSL2 (Ubuntu 22.04/24.04) for the least friction. Open an Ubuntu terminal and follow the Linux steps. Native Windows also works for the pip packages, but the lab is tested on Linux/macOS.

### 1. Create a virtual environment

```bash
# Linux / WSL2 (Ubuntu): make sure venv is available
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip

python3 -m venv ~/so101-rl
source ~/so101-rl/bin/activate        # re-run this in every new terminal
python -m pip install --upgrade pip
```

macOS: skip the `apt-get` line (Python from python.org or Homebrew includes `venv`).

### 2. Install PyTorch (CPU build)

```bash
# Linux / Windows(WSL2): small CPU-only wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu

# macOS: the default wheel is already fine
# pip install torch
```

*Have an NVIDIA GPU and want to use it?* Skip the line above and use the selector at <https://pytorch.org/get-started/locally/> to get the right CUDA command. It is optional.

### 3. Install the lab packages

```bash
pip install "mujoco>=3.2" "gymnasium>=1.0" "stable-baselines3[extra]>=2.4" \
            tensorboard imageio imageio-ffmpeg numpy matplotlib
```

### 4. Headless rendering (Linux / WSL2 only)

Needed for recording rollout videos without a display window:

```bash
sudo apt-get install -y libegl1 libgl1 libglu1-mesa libosmesa6 ffmpeg
echo 'export MUJOCO_GL=egl' >> ~/.bashrc    # use 'osmesa' if EGL fails (slower)
export MUJOCO_GL=egl
```

macOS does not need `MUJOCO_GL`. For the interactive viewer on macOS, run scripts with `mjpython` instead of `python`.

### 5. Get the lab code

```bash
git clone <LAB_REPO_URL> so101_pick_lab      # URL provided by your instructor
cd so101_pick_lab
```

Continue to [Verify your install](#verify-your-install).

---

## B. Google Colab

Colab has Python, PyTorch, and NumPy preinstalled. Run these in the **first cells** of your notebook.

**Cell 1: system libs and packages**

```python
!apt-get install -y -qq libegl1 libgl1 libglu1-mesa libosmesa6 ffmpeg
!pip install -q "mujoco>=3.2" "gymnasium>=1.0" "stable-baselines3[extra]>=2.4" tensorboard imageio imageio-ffmpeg
```

**Cell 2: headless rendering (must run before `import mujoco`)**

```python
import os
os.environ["MUJOCO_GL"] = "egl"      # fall back to "osmesa" if EGL errors
```

**Cell 3: get the lab code**

```python
!git clone <LAB_REPO_URL> so101_pick_lab
%cd so101_pick_lab
```

Colab tips:
- A GPU runtime is **not** needed. Use the default CPU runtime. Training one env per CPU core is usually faster.
- If the runtime restarts, re-run Cells 1–3. Files in `/content` are lost, so save checkpoints to Google Drive:
  ```python
  from google.colab import drive
  drive.mount("/content/drive")
  ```
- No interactive viewer in Colab. Use the provided video-rollout script and display the MP4 inline.
- TensorBoard in Colab: `%load_ext tensorboard` then `%tensorboard --logdir runs`.

---

## C. Course Docker container (optional)

The ROS 2 container does not include the RL packages. Inside the container shell:

```bash
pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --break-system-packages "gymnasium>=1.0" "stable-baselines3[extra]>=2.4" \
             tensorboard imageio imageio-ffmpeg matplotlib
sudo apt-get update && sudo apt-get install -y ffmpeg libosmesa6
export MUJOCO_GL=egl
```

MuJoCo is already installed in the container. Optionally add the pip lines to your `Dockerfile` and rebuild so it persists.

---

## Verify your install

Run **all** checks. Each should end with `OK`. If one fails, see [Troubleshooting](#troubleshooting).

### Check 1: versions

```bash
python - <<'EOF'
import sys, mujoco, gymnasium, stable_baselines3, torch, numpy
print("python      ", sys.version.split()[0])
print("mujoco      ", mujoco.__version__)
print("gymnasium   ", gymnasium.__version__)
print("sb3         ", stable_baselines3.__version__)
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy       ", numpy.__version__)
print("OK")
EOF
```

Expected: Python ≥ 3.10, MuJoCo ≥ 3.2, Gymnasium ≥ 1.0, SB3 ≥ 2.4. `cuda: False` is fine.

### Check 2: MuJoCo physics (no display needed)

```bash
python - <<'EOF'
import mujoco
xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
for _ in range(1000):
    mujoco.mj_step(m, d)
z = d.qpos[2]
print("box height after 2 s:", round(z, 3))
assert z < 0.2, "box did not fall: physics problem"
print("OK")
EOF
```

Expected: height ≈ 0.05 (the box has fallen and rests on the plane).

### Check 3: offscreen rendering (needed for videos)

```bash
python - <<'EOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")   # skip this line on macOS
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size=".2"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 240, 320)
r.update_scene(d)
img = r.render()
print("image shape:", img.shape, "| nonzero pixels:", int((img > 0).sum()))
assert img.shape == (240, 320, 3) and img.any()
print("OK")
EOF
```

If this fails on Linux/Colab, try `MUJOCO_GL=osmesa`. See Troubleshooting.

### Check 4: Gymnasium + SB3 training loop (≈30 s)

```bash
python - <<'EOF'
import gymnasium as gym
from stable_baselines3 import SAC
env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, verbose=0, device="cpu")
model.learn(total_timesteps=2000)
obs, _ = env.reset(seed=0)
action, _ = model.predict(obs, deterministic=True)
print("action:", action)
print("OK")
EOF
```

### Check 5: TensorBoard

```bash
tensorboard --version && echo OK
```

### Check 6: lab environment (after cloning the lab repo)

```bash
python scripts/check_env.py
```

This loads the SO-101 model, steps the environment for a few hundred steps with random actions, and renders one frame. It should print `All checks passed`. *(Script is provided in the lab repo.)*

### Optional: interactive viewer (local machines only)

```bash
python -m mujoco.viewer        # macOS: mjpython -m mujoco.viewer
```

A window should open. This does not work on Colab or on remote/headless machines, and that is fine for this lab.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Your venv isn't active. Run `source ~/so101-rl/bin/activate`. |
| `gladLoadGL error` / `EGL` errors | Install the libs in step A4, or switch to `export MUJOCO_GL=osmesa`. |
| `Cannot initialize a EGL device display` (Colab) | Set `MUJOCO_GL` **before** importing mujoco, or use `osmesa`. |
| Rendering is black or very slow | `osmesa` is software rendering. Use `egl` if available, and lower the render resolution. |
| `pip` dependency conflict with Gymnasium | Use a fresh virtual environment. Don't mix with other course environments. |
| `ffmpeg` not found when saving video | `sudo apt-get install -y ffmpeg` (Linux/Colab), `brew install ffmpeg` (macOS). |
| macOS: viewer crashes with `launch_passive` error | Run with `mjpython` instead of `python`. |
| Torch download very large or slow | Use the CPU index URL shown in step A2. |
| Colab loses everything on restart | Save checkpoints to Google Drive (see the Colab tips). |

Still stuck? Post the **full error message** plus the output of Check 1 on the course forum.
