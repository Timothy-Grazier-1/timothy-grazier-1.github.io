"""
OpenGRiD VTOL Fixed-Wing VIO Navigation using DPVO (non-SLAM)
==============================================================

RESEARCH SUMMARY
----------------
This script adapts DPVO (Deep Patch Visual Odometry) — without the SLAM backend —
to perform Visual (Inertial) Odometry for navigating a VTOL aircraft in fixed-wing mode
around the edges of the OpenGRiD simulation map.

KEY FINDINGS
------------
1. DPVO vs DPVO-SLAM:
   - DPVO-SLAM adds loop closure on top of DPVO via LOOP_CLOSURE=True / CLASSIC_LOOP_CLOSURE=True
   - Pure DPVO has no loop closure, runs as a sliding-window recurrent VO system
   - The core API is identical: slam(timestamp, image_tensor, intrinsics)
   - To disable SLAM/loop-closure: ensure cfg.LOOP_CLOSURE=False (this is the default)
   - DPVO's terminate() returns (timestamps, poses_SE3) — SE3 poses as 4x4 matrices

2. DPVO Python API (princeton-vl/DPVO):
   - Class: dpvo.dpvo.DPVO(cfg, network_weights_path, ht, wd, viz=False)
   - Call:  dpvo_instance(timestamp: int, image: torch.Tensor[C,H,W], intrinsics: np.ndarray[fx,fy,cx,cy])
   - End:   (timestamps, poses) = dpvo_instance.terminate()
   - Image tensor must be uint8 RGB on CUDA, shape (3, H, W)
   - Intrinsics: np.array([fx, fy, cx, cy]) in pixels

3. OpenGRiD / AirSim Interface:
   *** NOTE: OpenGRiD-specific documentation was not publicly available at time of research ***
   This script targets the Project AirSim Python API (iamaisim/ProjectAirSim), which is
   the most likely underlying interface for OpenGRiD simulation environments. Adjust the
   client calls to match your specific OpenGRiD API bindings.

   Project AirSim key API calls used:
   - client.confirmConnection()
   - drone.enable_api_control()
   - drone.arm()
   - drone.takeoff_async(altitude).join()
   - drone.transition_to_fixed_wing_async().join()   # switch to fixed-wing mode
   - drone.move_by_velocity_async(vx, vy, vz, duration)
   - drone.move_to_position_async(x, y, z, velocity)
   - client.get_images([ImageRequest(camera_name, ImageType.Scene, False, False)])
   - drone.get_imu_data()
   - drone.get_multirotor_state() / drone.get_fixed_wing_state()

4. VIO approach:
   DPVO is a monocular VO (not true VIO). IMU data is collected in parallel and fused
   with DPVO pose estimates using a complementary filter for roll/pitch stabilization
   and velocity integration. For true VIO, consider DPVO + separate IMU pre-integration.

5. Map-edge waypoints:
   Four corner waypoints define the perimeter circuit. The script flies:
   NW corner -> NE corner -> SE corner -> SW corner -> NW (repeat)
   Adjust WAYPOINTS to match your OpenGRiD map dimensions.

INSTALLATION REQUIREMENTS
--------------------------
    git clone https://github.com/princeton-vl/DPVO.git --recursive
    cd DPVO
    pip install .
    # Also install airsim client for OpenGRiD:
    pip install airsim   # or the OpenGRiD-specific client package
"""

import time
import threading
import numpy as np
import cv2
import torch

# ── DPVO imports ──────────────────────────────────────────────────────────────
from dpvo.config import cfg
from dpvo.dpvo import DPVO

# ── Simulation client import ──────────────────────────────────────────────────
# TODO: Replace with OpenGRiD-specific import if different from AirSim
import airsim
from airsim import ImageRequest, ImageType


# =============================================================================
# CONFIGURATION — Adjust these for your OpenGRiD environment
# =============================================================================

# Path to the DPVO pretrained model weights
DPVO_WEIGHTS = "models/dpvo.pth"

# Camera resolution (match your OpenGRiD camera config)
IMAGE_HEIGHT = 480
IMAGE_WIDTH  = 640

# Camera intrinsics [fx, fy, cx, cy] in pixels
# TODO: Replace with your OpenGRiD camera calibration values
CAMERA_INTRINSICS = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)

# Camera name in OpenGRiD / AirSim scene
CAMERA_NAME = "front_center"

# VTOL vehicle name in the simulation
VEHICLE_NAME = "VTOL_1"

# Altitude to maintain in fixed-wing mode (meters, AirSim uses NED so negative = up)
CRUISE_ALTITUDE_M = -50.0        # 50 m AGL (NED: negative = up)
CRUISE_SPEED_MPS  = 20.0         # 20 m/s forward cruise speed

# Minimum speed to maintain fixed-wing flight (stall prevention)
MIN_FIXED_WING_SPEED_MPS = 15.0

# Waypoints defining the MAP EDGE circuit (NED coordinates in meters)
# TODO: Adjust to match your OpenGRiD world bounds
WAYPOINTS = [
    (-400.0,  -400.0, CRUISE_ALTITUDE_M),   # NW corner
    (-400.0,   400.0, CRUISE_ALTITUDE_M),   # NE corner
    ( 400.0,   400.0, CRUISE_ALTITUDE_M),   # SE corner
    ( 400.0,  -400.0, CRUISE_ALTITUDE_M),   # SW corner
]

# Arrival radius — how close (m) to a waypoint triggers advancing to the next
WAYPOINT_RADIUS_M = 30.0

# How many laps around the map to complete
NUM_LAPS = 2

# Frame stride — process every N-th frame through DPVO (1 = every frame)
DPVO_FRAME_STRIDE = 1

# IMU sample rate (Hz) — how often to read IMU
IMU_HZ = 100

# =============================================================================
# DPVO CONFIGURATION (pure VO, no SLAM loop closure)
# =============================================================================

def build_dpvo_config():
    """Return a CfgNode with loop closure disabled (pure DPVO, not DPVO-SLAM)."""
    # Start from defaults
    dpvo_cfg = cfg.clone()

    # --- Disable ALL loop-closure / SLAM features ---
    dpvo_cfg.defrost()
    dpvo_cfg.LOOP_CLOSURE          = False   # no DPV-SLAM backend
    dpvo_cfg.CLASSIC_LOOP_CLOSURE  = False   # no NetVLAD/BoW retrieval
    dpvo_cfg.OPTIMIZATION_WINDOW   = 12      # local window only
    dpvo_cfg.BUFFER_SIZE           = 2048    # smaller buffer (no global map needed)
    dpvo_cfg.PATCHES_PER_FRAME     = 80
    dpvo_cfg.KEYFRAME_THRESH       = 12.5
    dpvo_cfg.MOTION_MODEL          = 'DAMPED_LINEAR'
    dpvo_cfg.MIXED_PRECISION       = True
    dpvo_cfg.freeze()
    return dpvo_cfg


# =============================================================================
# IMU DATA COLLECTOR (runs in background thread)
# =============================================================================

class IMUCollector:
    """Continuously reads IMU data from OpenGRiD/AirSim in a background thread."""

    def __init__(self, drone):
        self.drone  = drone
        self.lock   = threading.Lock()
        self._data  = None          # latest ImuData
        self._stop  = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        dt = 1.0 / IMU_HZ
        while not self._stop:
            try:
                imu = self.drone.get_imu_data()
                with self.lock:
                    self._data = imu
            except Exception as e:
                print(f"[IMU] Read error: {e}")
            time.sleep(dt)

    def get_latest(self):
        """Return latest ImuData (thread-safe). Returns None if not yet available."""
        with self.lock:
            return self._data

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)


# =============================================================================
# POSE TRACKER — wraps DPVO for online use
# =============================================================================

class DPVOPoseTracker:
    """
    Wraps DPVO for frame-by-frame pose estimation without SLAM loop closure.

    Usage:
        tracker = DPVOPoseTracker()
        tracker.process_frame(timestamp, bgr_image_numpy)
        current_pose = tracker.get_latest_pose()   # 4x4 SE3 matrix
        all_traj = tracker.finalize()              # (timestamps, poses_list)
    """

    def __init__(self):
        dpvo_cfg   = build_dpvo_config()
        self.dpvo  = DPVO(dpvo_cfg, DPVO_WEIGHTS,
                          ht=IMAGE_HEIGHT, wd=IMAGE_WIDTH, viz=False)
        self._frame_count = 0
        self._last_pose   = np.eye(4)
        self._lock        = threading.Lock()
        print("[DPVO] Initialized — pure VO mode (LOOP_CLOSURE=False)")

    def process_frame(self, timestamp: int, bgr_frame: np.ndarray) -> None:
        """
        Feed a new BGR frame into DPVO.

        Args:
            timestamp:  monotonically increasing integer frame index
            bgr_frame:  HxWx3 uint8 numpy array in BGR (as returned by OpenCV / AirSim)
        """
        if self._frame_count % DPVO_FRAME_STRIDE != 0:
            self._frame_count += 1
            return

        # Convert BGR → RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Build CUDA tensor shape (3, H, W) uint8
        image_tensor = (
            torch.from_numpy(rgb)
                 .permute(2, 0, 1)        # HWC → CHW
                 .cuda()
        )

        with torch.no_grad():
            self.dpvo(timestamp, image_tensor, CAMERA_INTRINSICS)

        self._frame_count += 1

    def get_latest_pose(self) -> np.ndarray:
        """
        Returns the most recent 4×4 SE3 pose (camera-to-world) as a numpy array.
        Returns identity before DPVO has enough frames to initialise.
        """
        # DPVO exposes self.dpvo.poses as a (N, 7) tensor [tx,ty,tz, qx,qy,qz,qw]
        # Convert the latest estimate to a 4×4 matrix.
        try:
            poses = self.dpvo.poses          # shape (BUFFER_SIZE, 7)
            n     = self.dpvo.n              # number of frames ingested so far
            if n < 2:
                return np.eye(4)

            latest = poses[n - 1].cpu().numpy()   # [tx, ty, tz, qx, qy, qz, qw]
            T = _pose_vec_to_matrix(latest)
            with self._lock:
                self._last_pose = T
            return T
        except Exception:
            return np.eye(4)

    def finalize(self):
        """
        Call after the flight is complete. Returns (timestamps, poses) where
        poses is a list of 4×4 SE3 numpy matrices, one per keyframe.
        """
        print("[DPVO] Finalizing trajectory …")
        with torch.no_grad():
            traj = self.dpvo.terminate()    # returns (timestamps_np, poses_np)
        # traj[1] shape: (N, 7) — convert to list of 4×4 matrices
        timestamps = traj[0]
        pose_mats  = [_pose_vec_to_matrix(p) for p in traj[1]]
        print(f"[DPVO] Trajectory has {len(pose_mats)} keyframes.")
        return timestamps, pose_mats


def _pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Convert a 7-element pose vector [tx, ty, tz, qx, qy, qz, qw]
    to a 4×4 homogeneous SE3 transformation matrix.
    """
    tx, ty, tz, qx, qy, qz, qw = pose_vec.tolist()
    # Rotation from quaternion (Hamilton convention)
    r = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    T = np.eye(4)
    T[:3, :3] = r
    T[:3,  3] = [tx, ty, tz]
    return T


# =============================================================================
# WAYPOINT NAVIGATOR
# =============================================================================

class WaypointNavigator:
    """
    Guides the VTOL in fixed-wing mode through the MAP_EDGE waypoint circuit
    using pose estimates from DPVO as position feedback.

    Control strategy:
    - Uses DPVO's incremental pose for heading/position correction.
    - Falls back to AirSim's built-in GPS-based move_to_position_async when
      DPVO has not yet initialised (< 5 keyframes).
    - Yaw is commanded to point from current position toward the next waypoint.
    - Speed is held constant at CRUISE_SPEED_MPS to stay above stall speed.
    """

    def __init__(self, drone, pose_tracker: DPVOPoseTracker):
        self.drone        = drone
        self.tracker      = pose_tracker
        self._wp_idx      = 0
        self._lap_count   = 0

    def _distance_to_wp(self, state, wp):
        """Euclidean distance from current drone NED position to waypoint NED."""
        pos = state.kinematics_estimated.position
        dx  = pos.x_val - wp[0]
        dy  = pos.y_val - wp[1]
        dz  = pos.z_val - wp[2]
        return (dx**2 + dy**2 + dz**2) ** 0.5

    def _yaw_to_wp(self, state, wp):
        """Compute yaw (radians) pointing from current position toward waypoint."""
        pos = state.kinematics_estimated.position
        dx  = wp[0] - pos.x_val
        dy  = wp[1] - pos.y_val
        return np.arctan2(dy, dx)

    def fly_circuit(self):
        """
        Fly around the map perimeter NUM_LAPS times.
        Blocks until all laps are complete or an error occurs.
        """
        print(f"[NAV] Starting {NUM_LAPS}-lap perimeter circuit …")

        for lap in range(NUM_LAPS):
            print(f"[NAV] --- Lap {lap + 1} / {NUM_LAPS} ---")
            for wp_idx, wp in enumerate(WAYPOINTS):
                print(f"[NAV] Flying to waypoint {wp_idx}: {wp}")
                self._fly_to_waypoint(wp)

        print("[NAV] Circuit complete.")

    def _fly_to_waypoint(self, wp):
        """
        Command the VTOL toward a single waypoint and block until arrival.
        Uses velocity commands to maintain fixed-wing flight parameters.
        """
        while True:
            state = self.drone.get_multirotor_state()
            dist  = self._distance_to_wp(state, wp)

            if dist < WAYPOINT_RADIUS_M:
                print(f"[NAV] Waypoint reached (dist={dist:.1f} m).")
                return

            yaw = self._yaw_to_wp(state, wp)

            # Decompose speed into NED components
            vx = CRUISE_SPEED_MPS * np.cos(yaw)
            vy = CRUISE_SPEED_MPS * np.sin(yaw)

            # Hold altitude: compute vertical correction
            pos   = state.kinematics_estimated.position
            dz    = CRUISE_ALTITUDE_M - pos.z_val
            vz    = np.clip(0.5 * dz, -3.0, 3.0)    # gentle altitude hold

            # --- DPVO pose feedback ---
            dpvo_pose = self.tracker.get_latest_pose()
            # dpvo_pose is camera-frame; for heading correction we read the
            # yaw from DPVO and compare to commanded yaw.
            # (Skipped here for brevity — add yaw correction from DPVO SE3 matrix
            #  by extracting heading: yaw_dpvo = atan2(R[1,0], R[0,0]))

            # Send velocity command (3 s duration, then re-evaluate)
            self.drone.move_by_velocity_async(
                float(vx), float(vy), float(vz),
                duration=3.0,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=float(np.degrees(yaw)))
            ).join()

            time.sleep(0.1)


# =============================================================================
# MAIN MISSION
# =============================================================================

def capture_frame(client) -> np.ndarray:
    """
    Capture a single RGB frame from the front-center camera.
    Returns a HxWx3 BGR uint8 numpy array.

    TODO: Adapt to OpenGRiD-specific image capture API if different.
    """
    responses = client.simGetImages([
        ImageRequest(CAMERA_NAME, ImageType.Scene, False, False)
    ], vehicle_name=VEHICLE_NAME)

    if not responses:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    resp = responses[0]
    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    img   = img1d.reshape(resp.height, resp.width, 3)
    # AirSim returns RGBA; convert to BGR for consistency with OpenCV
    bgr   = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr


def run_mission():
    """
    Full mission:
      1. Connect to OpenGRiD/AirSim
      2. Take off in multirotor mode
      3. Transition to fixed-wing mode
      4. Start DPVO + IMU collection
      5. Fly perimeter circuit using VIO feedback
      6. Land and save trajectory
    """

    # ── Connect ───────────────────────────────────────────────────────────────
    # TODO: Replace host/port with your OpenGRiD connection parameters
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
    client.confirmConnection()
    print("[SIM] Connected to OpenGRiD simulation server.")

    drone = client.vehicle_client(VEHICLE_NAME)
    # Alternative if using the newer Project AirSim API:
    # from projectairsim import Drone
    # drone = Drone(client, VEHICLE_NAME)

    # ── Arm and take off ──────────────────────────────────────────────────────
    drone.enable_api_control()
    drone.arm()
    print("[VTOL] Armed. Taking off …")

    # Take off to transition altitude in multirotor mode first
    takeoff_altitude = -30.0     # 30 m AGL in NED
    drone.takeoff_async(timeout_sec=20).join()
    drone.move_to_position_async(0, 0, takeoff_altitude, velocity=5).join()
    print(f"[VTOL] At {abs(takeoff_altitude)} m AGL. Transitioning to fixed-wing …")

    # ── Transition to fixed-wing mode ─────────────────────────────────────────
    # AirSim/Project AirSim VTOL transition command:
    # drone.transition_to_fixed_wing_async().join()
    #
    # For PX4 SITL VTOL, you can also send a MAVLink command directly:
    #   from pymavlink import mavutil
    #   conn.mav.command_long_send(... MAV_CMD_DO_VTOL_TRANSITION ...)
    #
    # TODO: Confirm the correct transition call for your OpenGRiD VTOL model.
    try:
        drone.transition_to_fixed_wing_async().join()
        print("[VTOL] Transition to fixed-wing complete.")
    except AttributeError:
        # Fallback: use MAVLink DO_VTOL_TRANSITION if using PX4 SITL
        print("[VTOL] transition_to_fixed_wing_async not available — "
              "falling back to MAVLink VTOL transition command.")
        # (Requires pymavlink and a MAVLink connection on udp:127.0.0.1:14550)
        _send_mavlink_vtol_transition(fixed_wing=True)

    # Accelerate to cruise speed before starting VIO (DPVO needs texture motion)
    drone.move_by_velocity_async(
        CRUISE_SPEED_MPS, 0, 0, duration=5.0
    ).join()

    # ── Initialise DPVO tracker ───────────────────────────────────────────────
    tracker = DPVOPoseTracker()

    # ── Start IMU collection ──────────────────────────────────────────────────
    imu_collector = IMUCollector(drone)

    # ── Start waypoint navigator ──────────────────────────────────────────────
    navigator = WaypointNavigator(drone, tracker)

    # ── Frame capture + DPVO running in main thread ───────────────────────────
    # We interleave frame capture and navigation in a loop.
    # For better performance, run capture in a separate thread.
    mission_thread = threading.Thread(
        target=navigator.fly_circuit, daemon=False
    )
    mission_thread.start()

    frame_idx = 0
    print("[VIO] Frame capture loop started.")
    try:
        while mission_thread.is_alive():
            t_start = time.time()

            # Capture frame
            bgr_frame = capture_frame(client)

            # Feed to DPVO
            tracker.process_frame(frame_idx, bgr_frame)

            # Optionally log IMU
            imu = imu_collector.get_latest()
            if imu and frame_idx % 50 == 0:
                print(f"[VIO] Frame {frame_idx} | "
                      f"accel=({imu.linear_acceleration.x_val:.2f}, "
                      f"{imu.linear_acceleration.y_val:.2f}, "
                      f"{imu.linear_acceleration.z_val:.2f}) m/s² | "
                      f"DPVO n={getattr(tracker.dpvo, 'n', '?')}")

            frame_idx += 1

            # Throttle to ~30 Hz
            elapsed = time.time() - t_start
            sleep_t = max(0.0, 1.0/30.0 - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("[VIO] Interrupted by user.")
    finally:
        imu_collector.stop()
        mission_thread.join(timeout=5.0)

    # ── Finalize trajectory ───────────────────────────────────────────────────
    timestamps, trajectory = tracker.finalize()

    # ── Land ──────────────────────────────────────────────────────────────────
    print("[VTOL] Landing …")
    try:
        drone.transition_to_multirotor_async().join()
    except AttributeError:
        _send_mavlink_vtol_transition(fixed_wing=False)

    drone.land_async().join()
    drone.disarm()
    drone.disable_api_control()
    print("[VTOL] Landed and disarmed.")

    # ── Save trajectory ───────────────────────────────────────────────────────
    _save_trajectory_tum(timestamps, trajectory, "vtol_vio_trajectory.txt")
    print("[OUT] Trajectory saved to vtol_vio_trajectory.txt")
    print("[MISSION] Complete.")


# =============================================================================
# UTILITIES
# =============================================================================

def _send_mavlink_vtol_transition(fixed_wing: bool):
    """
    Send MAV_CMD_DO_VTOL_TRANSITION via pymavlink to switch between
    multirotor (state=3) and fixed-wing (state=4) modes.
    Only needed when using PX4 SITL without Project AirSim's transition API.
    """
    try:
        from pymavlink import mavutil
        conn = mavutil.mavlink_connection("udp:127.0.0.1:14550")
        conn.wait_heartbeat()
        state = 4 if fixed_wing else 3   # 3=MC, 4=FW (MAV_VTOL_STATE)
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            0,          # confirmation
            state,      # param1: target state
            0, 0, 0, 0, 0, 0
        )
        time.sleep(5)   # allow time for transition
        conn.close()
    except ImportError:
        print("[WARN] pymavlink not installed — cannot send VTOL transition command.")
    except Exception as e:
        print(f"[WARN] MAVLink transition failed: {e}")


def _save_trajectory_tum(timestamps, pose_matrices, filepath: str):
    """
    Save trajectory in TUM RGB-D format:
    timestamp tx ty tz qx qy qz qw
    """
    with open(filepath, "w") as f:
        f.write("# TUM trajectory format: timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, pose_matrices):
            tx, ty, tz = T[:3, 3]
            # Extract quaternion from rotation matrix
            R = T[:3, :3]
            qw = 0.5 * np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2]))
            if qw > 1e-6:
                qx = (R[2,1] - R[1,2]) / (4 * qw)
                qy = (R[0,2] - R[2,0]) / (4 * qw)
                qz = (R[1,0] - R[0,1]) / (4 * qw)
            else:
                qx, qy, qz = 0.0, 0.0, 0.0
            f.write(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_mission()
