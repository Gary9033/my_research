#!/usr/bin/env python3
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_angle_axis

from depth_camera_filtering import filter_depth
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.mapping.value_map import ValueMap
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix

# ═══ 設定 ═══════════════════════════════════════════════════════════════════
SCENE_PATH    = "/home/gary/vlfm/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
START_POS     = np.array([-3.0, 0.0, 0.0])
IMG_W, IMG_H  = 640, 480
HFOV_DEG      = 79.0
CAMERA_HEIGHT = 0.88
MIN_DEPTH     = 0.5
MAX_DEPTH     = 5.0
N_STEPS       = 30
STEP_SIZE     = 0.25
TEXT_PROMPT   = "Seems like there is a clock ahead."
OUTPUT_DIR    = "vlfm_demo_output"
BLIP2_PORT    = 12182

os.makedirs(OUTPUT_DIR, exist_ok=True)
hfov_rad = np.deg2rad(HFOV_DEG)
FX = FY = IMG_W / (2 * np.tan(hfov_rad / 2))

# ═══ Habitat-Sim ═════════════════════════════════════════════════════════════
def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.allow_sliding = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [IMG_H, IMG_W]
    rgb_spec.hfov = HFOV_DEG
    rgb_spec.position = [0.0, CAMERA_HEIGHT, 0.0]

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
            "move_forward", habitat_sim.ActuationSpec(amount=STEP_SIZE)),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=30.0)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

def get_yaw(agent_state):
    rot = agent_state.rotation
    angle, axis = quat_to_angle_axis(rot)
    return float(angle * np.sign(axis[1]) if abs(axis[1]) > 0.5 else 0.0)

# ═══ 重疊像素分析（新增）════════════════════════════════════════════════════
def print_overlap_analysis(value_map: ValueMap, conf_before: np.ndarray,
                           conf_after: np.ndarray, step: int):
    """
    比較 update_map 前後的信心矩陣，找出：
      1. 本步新觀測到的像素（conf_before==0, conf_after>0）
      2. 重疊像素（conf_before>0 AND conf_after>0）
         → 信心變高 or 變低，並換算回 world XY 座標
    ValueMap 座標系：
      px = int(cam_x * ppm) + origin[0]    (行，對應 x)
      py = int(-cam_y * ppm) + origin[1]   (列，對應 y)
    → world_x = (px - origin[0]) / ppm
    → world_y = -(py - origin[1]) / ppm
    """
    ppm    = value_map.pixels_per_meter           # 預設 20 px/m
    origin = value_map._episode_pixel_origin      # (ox, oy)
    size   = conf_before.shape[0]

    diff = conf_after - conf_before               # 正：信心上升，負：下降

    # 新觀測（之前是 0，現在有值）
    new_mask      = (conf_before == 0) & (conf_after > 0)
    # 重疊且信心上升
    overlap_up    = (conf_before > 0) & (diff > 1e-4)
    # 重疊且信心下降
    overlap_down  = (conf_before > 0) & (diff < -1e-4)

    def px_to_world(rows, cols):
        """rows/cols → world (x, y) in meters"""
        # ValueMap 存的 row=px (x方向), col=py (y方向)
        wx =  (rows - origin[0]) / ppm
        wy = -(cols - origin[1]) / ppm
        return wx, wy

    n_new  = np.sum(new_mask)
    n_up   = np.sum(overlap_up)
    n_down = np.sum(overlap_down)

    print(f"\n{'='*60}")
    print(f"[Step {step:03d}] 重疊分析：")
    print(f"  新觀測像素：{n_new} px  ({n_new/ppm**2:.2f} m²)")
    print(f"  重疊像素（信心↑）：{n_up} px")
    print(f"  重疊像素（信心↓）：{n_down} px")

    # 信心變化最大的前 5 個重疊像素，印出 world XY
    all_overlap = (conf_before > 0) & (conf_after > 0)
    if np.any(all_overlap):
        rows_ov, cols_ov = np.where(all_overlap)
        diffs_ov = diff[rows_ov, cols_ov]
        conf_b   = conf_before[rows_ov, cols_ov]
        conf_a   = conf_after[rows_ov, cols_ov]

        # 按 |diff| 降序取前5
        top_idx = np.argsort(-np.abs(diffs_ov))[:5]
        print(f"  前5大信心變化像素（world XY）：")
        for i in top_idx:
            wx, wy = px_to_world(rows_ov[i], cols_ov[i])
            direction = "↑" if diffs_ov[i] > 0 else "↓"
            print(f"    ({wx:+.2f}m, {wy:+.2f}m)  "
                  f"conf: {conf_b[i]:.3f} → {conf_a[i]:.3f}  "
                  f"Δ={diffs_ov[i]:+.4f} {direction}")

        # 統計信心上升 vs 下降的平均 Δ
        if n_up > 0:
            mean_up = diffs_ov[diffs_ov > 1e-4].mean()
            print(f"  重疊像素平均信心↑：+{mean_up:.4f}")
        if n_down > 0:
            mean_down = diffs_ov[diffs_ov < -1e-4].mean()
            print(f"  重疊像素平均信心↓：{mean_down:.4f}")
    print(f"{'='*60}")

# ═══ 存圖 ════════════════════════════════════════════════════════════════════
def save_frame(step, rgb, value_map, cosine_score,
               robot_xy, robot_heading, frontiers,
               sorted_frontiers, sorted_values):

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(rgb)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    vm_bgr = value_map.visualize()
    ax2.imshow(cv2.cvtColor(vm_bgr, cv2.COLOR_BGR2RGB))
    ax2.set_title("Value Map", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    fm_img = vm_bgr.copy()
    map_size = fm_img.shape[0]
    mpp = 0.05

    def world_to_px(xy):
        cx = cy = map_size // 2
        return (np.clip(int(cx + xy[0]/mpp), 0, map_size-1),
                np.clip(int(cy - xy[1]/mpp), 0, map_size-1))

    rx, ry = world_to_px(robot_xy)
    cv2.circle(fm_img, (rx, ry), 8, (0, 255, 0), -1)
    ax_end = (int(rx + 20*np.cos(robot_heading)),
              int(ry - 20*np.sin(robot_heading)))
    cv2.arrowedLine(fm_img, (rx, ry), ax_end, (0,255,0), 2, tipLength=0.4)
    for f in frontiers:
        cv2.circle(fm_img, world_to_px(f[:2]), 5, (0,255,255), 2)
    if len(sorted_frontiers) > 0:
        bpt = world_to_px(sorted_frontiers[0][:2])
        cv2.circle(fm_img, bpt, 7, (0,128,255), -1)
        cv2.putText(fm_img, f"{sorted_values[0]:.3f}", (bpt[0]+8, bpt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,128,255), 1)

    ax3.imshow(cv2.cvtColor(fm_img, cv2.COLOR_BGR2RGB))
    ax3.set_title("Frontier Map  (● robot  ○ frontier  ● best)", fontsize=10)
    ax3.axis("off")

    curr_value = value_map.sort_waypoints(np.array([robot_xy]), radius=0.3)[1]
    curr_value_scalar = curr_value[0] if len(curr_value) > 0 else 0.0

    fig.suptitle(
        f"Step {step:03d}  |  "
        f"Robot: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})  |  "
        f"Yaw: {np.rad2deg(robot_heading):.1f}°  |  "
        f"Raw Cosine: {cosine_score:.4f}  |  "
        f"Map Value: {curr_value_scalar:.4f}",
        fontsize=12, y=1.02
    )
    out_path = os.path.join(OUTPUT_DIR, f"step_{step:03d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")

# ═══ frontier 產生器 ═════════════════════════════════════════════════════════
def make_frontiers(robot_xy, robot_heading, n=5, spacing=1.0):
    pts = []
    for i in range(1, n+1):
        pts.append([robot_xy[0] + spacing*i*np.cos(robot_heading),
                    robot_xy[1] + spacing*i*np.sin(robot_heading)])
    for off in [-np.pi/4, np.pi/4, -np.pi/2, np.pi/2]:
        a = robot_heading + off
        pts.append([robot_xy[0] + spacing*2*np.cos(a),
                    robot_xy[1] + spacing*2*np.sin(a)])
    return np.array(pts)

# ═══ MAIN ════════════════════════════════════════════════════════════════════
def main():
    print("=== 初始化 Habitat-Sim ===")
    sim   = make_sim()
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
        obs       = sim.get_sensor_observations()
        rgb       = obs["rgb"][:, :, :3].astype(np.uint8)
        depth_raw = obs["depth"].astype(np.float32)

        depth_norm     = np.clip((depth_raw - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0.0, 1.0)
        depth_filtered = filter_depth(depth_norm, blur_type=None)

        cur_state = agent.get_state()
        pos = cur_state.position
        gps_x, gps_y = pos[0], -pos[2]
        camera_yaw   = get_yaw(cur_state)

        camera_position = np.array([gps_x, gps_y, CAMERA_HEIGHT])
        robot_xy        = camera_position[:2]
        tf_cam_to_ep    = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        cosine_score = itm.cosine(rgb, TEXT_PROMPT)
        print(f"Step {step:03d} | pos=({gps_x:.2f},{gps_y:.2f}) "
              f"yaw={np.rad2deg(camera_yaw):.1f}° | cosine={cosine_score:.4f}")

        # ── 重疊分析：update_map 前後快照 ──────────────────────────────────
        conf_before = value_map._map.copy()   # update_map 之前的信心矩陣

        value_map.update_map(
            values=np.array([cosine_score]),
            depth=depth_filtered,
            tf_camera_to_episodic=tf_cam_to_ep,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            fov=hfov_rad,
        )

        conf_after = value_map._map.copy()    # update_map 之後的信心矩陣

        # 印出重疊像素分析
        print_overlap_analysis(value_map, conf_before, conf_after, step)
        # ────────────────────────────────────────────────────────────────────

        value_map.update_agent_traj(robot_xy, camera_yaw)

        frontiers = make_frontiers(robot_xy, camera_yaw)
        sorted_frontiers, sorted_values = value_map.sort_waypoints(frontiers, radius=0.5)

        save_frame(step, rgb, value_map, cosine_score,
                   robot_xy, camera_yaw, frontiers,
                   sorted_frontiers, sorted_values)

        sim.step("move_forward")

    print(f"\n完成！圖片存在 ./{OUTPUT_DIR}/")
    sim.close()

if __name__ == "__main__":
    main()
