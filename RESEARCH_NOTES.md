# Research Notes: DPVO VIO + VTOL Fixed-Wing in OpenGRiD

## 1. OpenGRiD Interface

**Status:** No public documentation for "OpenGRiD" was found in indexed sources as of March 2026.

The most likely underlying interface is **Project AirSim** (iamaisim/ProjectAirSim) or the legacy
**Microsoft AirSim**, both of which support:
- VTOL aircraft (quad-x tailsitter and quad tiltrotor)
- Python API client
- PX4 SITL integration for fixed-wing transition

**Action needed from you:**
- Confirm whether OpenGRiD uses the AirSim Python client (`import airsim`) or a custom SDK
- Provide the connection endpoint (host/port)
- Confirm the vehicle name and camera name in the scene
- Confirm the coordinate system (NED vs ENU vs world-frame)

### VTOL Fixed-Wing Mode — How to Transition

| Interface | Transition command |
|---|---|
| Project AirSim Python API | `drone.transition_to_fixed_wing_async().join()` |
| PX4 SITL via MAVLink | `MAV_CMD_DO_VTOL_TRANSITION` with param1=4 (fixed-wing) |
| ArduPilot SITL | `vehicle.mode = VehicleMode("FBWA")` (fixed-wing mode) |

### Waypoint / Velocity Control in Fixed-Wing Mode

Fixed-wing aircraft cannot hover — minimum speed must stay above stall speed (~15 m/s for most
simulation models). Use **velocity commands** rather than position-hold:

```python
# AirSim velocity command (NED frame)
drone.move_by_velocity_async(vx, vy, vz, duration_s,
    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=heading_deg)
).join()
```

For PX4 SITL use MAVLink `SET_POSITION_TARGET_LOCAL_NED` with velocity fields.

---

## 2. DPVO vs DPVO-SLAM — Key Differences

| Feature | DPVO (pure VO) | DPVO-SLAM |
|---|---|---|
| Loop closure | **No** | Yes (NetVLAD retrieval) |
| Classic loop closure | **No** | Optional (BoW) |
| Global map optimization | **No** | Yes (full BA on loop) |
| Memory growth | Bounded (sliding window) | Unbounded (global map) |
| Latency | Lower | Higher on loop events |
| Config flags | `LOOP_CLOSURE=False` (default) | `LOOP_CLOSURE=True` |
| Python class | `dpvo.dpvo.DPVO` | Same class, different cfg |

**The class is the same** — only the configuration differs. To disable SLAM:

```python
cfg.LOOP_CLOSURE         = False
cfg.CLASSIC_LOOP_CLOSURE = False
```

These are already the defaults in `dpvo/config.py`, so unless your original script explicitly
set them to `True`, removing those lines is sufficient.

---

## 3. DPVO Python API Cheat Sheet

```python
from dpvo.config import cfg
from dpvo.dpvo   import DPVO

# 1. Build instance
dpvo = DPVO(cfg, "models/dpvo.pth", ht=480, wd=640, viz=False)

# 2. Per-frame call (inside loop)
#    timestamp : monotonically increasing int (frame index)
#    image     : torch.Tensor shape (3, H, W), dtype uint8, on CUDA
#    intrinsics: torch.Tensor([fx, fy, cx, cy], dtype=float32).cuda()  ← MUST be CUDA tensor
dpvo(timestamp, image_tensor, intrinsics)

# 3. Finalize after all frames processed
# IMPORTANT: terminate() returns (poses, tstamps) — poses FIRST, tstamps SECOND
poses, tstamps = dpvo.terminate()
# poses  shape: (N, 7) — [x, y, z, qx, qy, qz, qw] per frame (world-from-camera, interpolated)
# tstamps shape: (N,)   — float64 frame indices
```

### Converting AirSim frame to DPVO tensor

```python
import airsim, cv2, numpy as np, torch

responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
])
img1d      = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
img        = img1d.reshape(responses[0].height, responses[0].width, 3)  # RGB
tensor     = torch.from_numpy(img).permute(2,0,1).cuda()                # (3,H,W) uint8
intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).cuda() # MUST be CUDA tensor
```

---

## 4. Visual Inertial Odometry — DPVO + IMU Note

DPVO is a **monocular Visual Odometry** system — it does not natively consume IMU data.
Options for true VIO:

| Approach | Complexity | Notes |
|---|---|---|
| DPVO alone | Low | Sufficient for short flights; drifts over time |
| DPVO + IMU complementary filter | Medium | IMU corrects roll/pitch; use accel for gravity align |
| DPVO + pre-integrated IMU factor | High | Feed IMU deltas as motion prior into DPVO's `MOTION_MODEL` |
| MSCKFT / VINS-Mono + DPVO | High | Run DPVO for appearance-based loop closure only |

For a perimeter circuit of ~800 m × 800 m at 20 m/s the flight is ~5–6 min per lap.
Pure DPVO drift should be manageable for 1–2 laps; add IMU fusion for longer missions.

---

## 5. Map Edge Waypoints

Adjust `WAYPOINTS` in `opengrid_dpvo_vio_vtol.py` to match your map. The defaults use
a symmetric ±400 m square. You can query the OpenGRiD/AirSim world bounds with:

```python
# AirSim — get world extent from scene object list or environment config
env = client.simGetEnvironmentState()
print(env)
```

---

## 6. References

- DPVO GitHub: https://github.com/princeton-vl/DPVO
- DPV-SLAM paper (ECCV 2024): https://arxiv.org/abs/2408.01654
- Project AirSim (VTOL): https://iamaisim.github.io/ProjectAirSim/
- PX4 VTOL SITL in AirSim: https://iamaisim.github.io/ProjectAirSim/controllers/px4/px4_sitl.html
- MAV_CMD_DO_VTOL_TRANSITION: https://mavlink.io/en/messages/common.html#MAV_CMD_DO_VTOL_TRANSITION
- VTOL VIO landmark paper (MDPI Sensors 2022): https://www.mdpi.com/1424-8220/22/24/9654
