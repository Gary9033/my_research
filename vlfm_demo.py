#!/usr/bin/env python3
"""
VLFM standalone demo:
  - 用 habitat-sim 載入 HM3D glb 場景
  - agent 從指定位置出發，直線往前走
  - 每步輸出：RGB frame、value map、frontier map、當前 cosine score
"""

import os
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Habitat-Sim ──────────────────────────────────────────────────────────────
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_angle_axis

# ── VLFM ─────────────────────────────────────────────────────────────────────
from depth_camera_filtering import filter_depth
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.mapping.value_map import ValueMap
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix

# ═══════════════════════════════════════════════════════════════════════════════
# 設定（與 VLFM / HabitatMixin 相同）
# ═══════════════════════════════════════════════════════════════════════════════
SCENE_PATH    = "/home/gary/vlfm/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
START_POS     = np.array([-3.0, 0.0, 0.0])   # world space (x, height, z)
IMG_W, IMG_H  = 640, 480
HFOV_DEG      = 79.0
CAMERA_HEIGHT = 0.88    # rgb_sensor.position[1]，與 VLFM config 相同
MIN_DEPTH     = 0.5
MAX_DEPTH     = 5.0
N_STEPS       = 30
STEP_SIZE     = 0.25    # 每步 0.25m（Habitat 預設）
TEXT_PROMPT   = "Seems like there is a clock ahead."
OUTPUT_DIR    = "vlfm_demo_output"
BLIP2_PORT    = 12182

os.makedirs(OUTPUT_DIR, exist_ok=True)

# focal length（與 HabitatMixin.__init__ 相同算法）
hfov_rad = np.deg2rad(HFOV_DEG)
FX = FY = IMG_W / (2 * np.tan(hfov_rad / 2))


# ═══════════════════════════════════════════════════════════════════════════════
# 建立 Habitat-Sim
# ═══════════════════════════════════════════════════════════════════════════════
def make_sim() -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.allow_sliding = True

    # RGB sensor
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [IMG_H, IMG_W]
    rgb_spec.hfov = HFOV_DEG
    rgb_spec.position = [0.0, CAMERA_HEIGHT, 0.0]

    # Depth sensor
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [IMG_H, IMG_W]
    depth_spec.hfov = HFOV_DEG
    depth_spec.position = [0.0, CAMERA_HEIGHT, 0.0]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.ActuationSpec(amount=STEP_SIZE),
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=30.0),
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


# ═══════════════════════════════════════════════════════════════════════════════
# 取得 agent 的 yaw（從 quaternion）
# ═══════════════════════════════════════════════════════════════════════════════
def get_yaw(agent_state: habitat_sim.AgentState) -> float:
    """Habitat rotation quaternion → yaw (radians, CCW from above)"""
    rot = agent_state.rotation          # np.quaternion
    angle, axis = quat_to_angle_axis(rot)
    yaw = angle * np.sign(axis[1]) if abs(axis[1]) > 0.5 else 0.0
    return float(yaw)


# ═══════════════════════════════════════════════════════════════════════════════
# 儲存每步的圖（3-panel: RGB / ValueMap / FrontierMap）
# ═══════════════════════════════════════════════════════════════════════════════
def save_frame(step, rgb, value_map, cosine_score,
               robot_xy, robot_heading, frontiers,
               sorted_frontiers, sorted_values):

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── 1. RGB ────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(rgb)
    ax1.set_title(f"Step {step:03d}  |  BLIP2 cosine: {cosine_score:.4f}", fontsize=11)
    ax1.axis("off")

    # ── 2. Value Map ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    vm_bgr = value_map.visualize()
    vm_rgb = cv2.cvtColor(vm_bgr, cv2.COLOR_BGR2RGB)
    ax2.imshow(vm_rgb)
    ax2.set_title("Value Map", fontsize=11)
    ax2.axis("off")

    # ── 3. Frontier Map（在 value map 上疊加 robot + frontiers）─────────────
    ax3 = fig.add_subplot(gs[2])
    fm_img = vm_bgr.copy()
    map_size = fm_img.shape[0]
    meters_per_pixel = 0.05   # ValueMap 預設值

    def world_to_px(xy):
        cx = cy = map_size // 2
        px = int(cx + xy[0] / meters_per_pixel)
        py = int(cy - xy[1] / meters_per_pixel)
        return (np.clip(px, 0, map_size-1), np.clip(py, 0, map_size-1))

    # robot 位置 + heading 箭頭
    rx, ry = world_to_px(robot_xy)
    cv2.circle(fm_img, (rx, ry), 8, (0, 255, 0), -1)
    arr_len = 20
    ax_end = (int(rx + arr_len * np.cos(robot_heading)),
              int(ry - arr_len * np.sin(robot_heading)))
    cv2.arrowedLine(fm_img, (rx, ry), ax_end, (0, 255, 0), 2, tipLength=0.4)

    # 所有 frontiers
    for f in frontiers:
        cv2.circle(fm_img, world_to_px(f[:2]), 5, (0, 255, 255), 2)

    # 最佳 frontier（橘色）
    if len(sorted_frontiers) > 0:
        bpt = world_to_px(sorted_frontiers[0][:2])
        cv2.circle(fm_img, bpt, 7, (0, 128, 255), -1)
        cv2.putText(fm_img, f"{sorted_values[0]:.3f}",
                    (bpt[0]+8, bpt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,128,255), 1)

    ax3.imshow(cv2.cvtColor(fm_img, cv2.COLOR_BGR2RGB))
    ax3.set_title("Frontier Map  (● robot  ○ frontier  ● best)", fontsize=10)
    ax3.axis("off")

    fig.suptitle(
        f"Step {step:03d}  |  robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f})  "
        f"yaw={np.rad2deg(robot_heading):.1f}°",
        fontsize=12, y=1.01
    )
    out_path = os.path.join(OUTPUT_DIR, f"step_{step:03d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 簡易 frontier 產生器（沿前進方向 + 左右，取代 frontier_exploration lib）
# ═══════════════════════════════════════════════════════════════════════════════
def make_frontiers(robot_xy, robot_heading, n=5, spacing=1.0):
    pts = []
    for i in range(1, n + 1):
        pts.append([
            robot_xy[0] + spacing * i * np.cos(robot_heading),
            robot_xy[1] + spacing * i * np.sin(robot_heading),
        ])
    for off in [-np.pi/4, np.pi/4, -np.pi/2, np.pi/2]:
        a = robot_heading + off
        pts.append([
            robot_xy[0] + spacing * 2 * np.cos(a),
            robot_xy[1] + spacing * 2 * np.sin(a),
        ])
    return np.array(pts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=== 初始化 Habitat-Sim ===")
    sim = make_sim()
    agent = sim.initialize_agent(0)

    state = habitat_sim.AgentState()
    state.position = START_POS.copy()
    state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)

    print(f"=== 初始化 BLIP2ITM (port={BLIP2_PORT}) ===")
    itm = BLIP2ITMClient(port=BLIP2_PORT)

    print("=== 初始化 ValueMap ===")
    value_map = ValueMap(value_channels=1, use_max_confidence=False)

    print(f"=== 開始走 {N_STEPS} 步 ===")
    for step in range(N_STEPS):
        # ── observations ─────────────────────────────────────────────────────
        obs       = sim.get_sensor_observations()
        rgb       = obs["rgb"][:, :, :3].astype(np.uint8)
        depth_raw = obs["depth"].astype(np.float32)          # meters

        # depth 正規化（同 HabitatMixin）
        depth_norm     = np.clip((depth_raw - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0.0, 1.0)
        depth_filtered = filter_depth(depth_norm, blur_type=None)

        # pose（同 _cache_observations）
        cur_state = agent.get_state()
        pos = cur_state.position                   # [x, height, z]
        gps_x, gps_y = pos[0], -pos[2]            # Habitat GPS flip y
        camera_yaw   = get_yaw(cur_state)

        camera_position = np.array([gps_x, gps_y, CAMERA_HEIGHT])
        robot_xy        = camera_position[:2]
        tf_cam_to_ep    = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # BLIP2 cosine
        cosine_score = itm.cosine(rgb, TEXT_PROMPT)
        print(f"Step {step:03d} | pos=({gps_x:.2f},{gps_y:.2f}) "
              f"yaw={np.rad2deg(camera_yaw):.1f}° | cosine={cosine_score:.4f}")

        # 更新 ValueMap（同 _update_value_map）
        value_map.update_map(
            values=np.array([cosine_score]),
            depth=depth_filtered,
            tf_camera_to_episodic=tf_cam_to_ep,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            fov=hfov_rad,
        )
        value_map.update_agent_traj(robot_xy, camera_yaw)

        # frontiers + value 排序
        frontiers = make_frontiers(robot_xy, camera_yaw)
        sorted_frontiers, sorted_values = value_map.sort_waypoints(frontiers, radius=0.5)

        # 輸出圖片
        save_frame(step, rgb, value_map, cosine_score,
                   robot_xy, camera_yaw, frontiers,
                   sorted_frontiers, sorted_values)

        # 往前走
        sim.step("move_forward")

    print(f"\n完成！圖片存在 ./{OUTPUT_DIR}/")
    sim.close()


if __name__ == "__main__":
    main()
