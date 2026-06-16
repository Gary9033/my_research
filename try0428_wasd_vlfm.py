import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from IPython.display import clear_output, display

from depth_camera_filtering import filter_depth
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.blip2itm import BLIP2ITMClient


DEFAULT_SCENE_PATH = "/home/gary/vlfm/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
START_POS = np.array([-3.0, 0.0, 0.0])
IMG_W, IMG_H = 640, 480
HFOV_DEG = 79.0
CAMERA_HEIGHT = 0.88
MIN_DEPTH = 0.5
MAX_DEPTH = 5.0
STEP_SIZE = 0.25
TURN_DEG = 30.0
TEXT_PROMPT = "Seems like there is a clock ahead."
BLIP2_PORT = 12182

hfov_rad = np.deg2rad(HFOV_DEG)
FX = FY = IMG_W / (2 * np.tan(hfov_rad / 2))


def make_vlfm_sim(scene_path: str) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
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
            "move_forward",
            habitat_sim.ActuationSpec(amount=STEP_SIZE),
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=TURN_DEG),
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=TURN_DEG),
        ),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def get_yaw(agent_state: habitat_sim.AgentState) -> float:
    angle, axis = quat_to_angle_axis(agent_state.rotation)
    return float(angle * np.sign(axis[1]) if abs(axis[1]) > 0.5 else 0.0)


def move_backward(agent: habitat_sim.Agent, step_size: float) -> None:
    state = agent.get_state()
    yaw = get_yaw(state)
    state.position = state.position + np.array([-np.cos(yaw), 0.0, -np.sin(yaw)]) * step_size
    agent.set_state(state)


def capture_step(sim: habitat_sim.Simulator, agent: habitat_sim.Agent, itm: BLIP2ITMClient,
                 value_map: ValueMap, obstacle_map: ObstacleMap):
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3].astype(np.uint8)
    depth_raw = obs["depth"].astype(np.float32)
    depth_norm = np.clip((depth_raw - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0.0, 1.0)
    depth_filtered = filter_depth(depth_norm, blur_type=None)

    state = agent.get_state()
    pos = state.position
    robot_xy = np.array([pos[0], -pos[2]])
    robot_heading = get_yaw(state)
    tf_cam_to_ep = xyz_yaw_to_tf_matrix(np.array([robot_xy[0], robot_xy[1], CAMERA_HEIGHT]), robot_heading)

    cosine_score = itm.cosine(rgb, TEXT_PROMPT)

    value_map.update_map(
        values=np.array([cosine_score]),
        depth=depth_filtered,
        tf_camera_to_episodic=tf_cam_to_ep,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        fov=hfov_rad,
    )
    value_map.update_agent_traj(robot_xy, robot_heading)

    obstacle_map.update_map(
        depth_filtered,
        tf_cam_to_ep,
        MIN_DEPTH,
        MAX_DEPTH,
        FX,
        FY,
        hfov_rad,
    )
    obstacle_map.update_agent_traj(robot_xy, robot_heading)

    return rgb, cosine_score, robot_xy, robot_heading


def render_state(step_idx: int, action_name: str, rgb: np.ndarray, cosine_score: float,
                 robot_xy: np.ndarray, robot_heading: float, value_map: ValueMap,
                 obstacle_map: ObstacleMap) -> None:
    vm_bgr = value_map.visualize(obstacle_map=obstacle_map)
    fm_bgr = obstacle_map.visualize().copy()

    def to_px(xy: np.ndarray) -> tuple[int, int]:
        return tuple(obstacle_map._xy_to_px(np.array([xy]))[0])

    robot_px = to_px(robot_xy)
    cv2.circle(fm_bgr, robot_px, 6, (0, 255, 0), -1)
    arrow_len = 20
    end_px = (
        int(robot_px[0] + arrow_len * np.cos(robot_heading)),
        int(robot_px[1] - arrow_len * np.sin(robot_heading)),
    )
    cv2.arrowedLine(fm_bgr, robot_px, end_px, (0, 255, 0), 2, tipLength=0.35)

    frontiers = obstacle_map.frontiers
    if len(frontiers) > 0:
        sorted_frontiers, sorted_values = value_map.sort_waypoints(frontiers, radius=0.5)
    else:
        sorted_frontiers = np.array([])
        sorted_values = []

    for frontier in frontiers:
        cv2.circle(fm_bgr, to_px(frontier[:2]), 4, (0, 255, 255), 2)

    if len(sorted_frontiers) > 0:
        best_px = to_px(sorted_frontiers[0][:2])
        cv2.circle(fm_bgr, best_px, 7, (0, 128, 255), -1)
        cv2.putText(
            fm_bgr,
            f"{sorted_values[0]:.3f}",
            (best_px[0] + 8, best_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 128, 255),
            1,
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Step {step_idx:03d} | action={action_name}")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vm_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Value Map")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(fm_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("VLFM Frontier Map")
    axes[2].axis("off")

    fig.suptitle(
        f"prompt={TEXT_PROMPT} | cosine={cosine_score:.4f} | "
        f"robot=({robot_xy[0]:.2f}, {robot_xy[1]:.2f}) | "
        f"yaw={np.rad2deg(robot_heading):.1f}°",
        y=1.02,
    )
    clear_output(wait=True)
    display(fig)
    plt.close(fig)


def main() -> None:
    scene_path = globals().get("test_scene", DEFAULT_SCENE_PATH)
    print(f"Loading Habitat scene: {scene_path}")

    sim = make_vlfm_sim(scene_path)
    agent = sim.initialize_agent(0)

    state = habitat_sim.AgentState()
    state.position = START_POS.copy()
    state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)

    print(f"Loading BLIP2ITM client on port {BLIP2_PORT} ...")
    itm = BLIP2ITMClient(port=BLIP2_PORT)

    value_map = ValueMap(value_channels=1, use_max_confidence=False)
    obstacle_map = ObstacleMap(min_height=0.15, max_height=0.88, agent_radius=0.18)

    step_idx = 0
    rgb, cosine_score, robot_xy, robot_heading = capture_step(sim, agent, itm, value_map, obstacle_map)
    render_state(step_idx, "init", rgb, cosine_score, robot_xy, robot_heading, value_map, obstacle_map)
    print("Controls: w=forward, s=backward, a=turn_left, d=turn_right, q=quit")

    while True:
        key = input("WASD> ").strip().lower()
        if key == "q":
            break
        if key == "w":
            action_name = "move_forward"
            sim.step("move_forward")
        elif key == "s":
            action_name = "move_backward"
            move_backward(agent, STEP_SIZE)
        elif key == "a":
            action_name = "turn_left"
            sim.step("turn_left")
        elif key == "d":
            action_name = "turn_right"
            sim.step("turn_right")
        else:
            print("Please press one of: w/a/s/d/q")
            continue

        step_idx += 1
        rgb, cosine_score, robot_xy, robot_heading = capture_step(
            sim, agent, itm, value_map, obstacle_map
        )
        render_state(step_idx, action_name, rgb, cosine_score, robot_xy, robot_heading, value_map, obstacle_map)
        print("Controls: w=forward, s=backward, a=turn_left, d=turn_right, q=quit")


main()
