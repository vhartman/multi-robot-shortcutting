"""Render certified RAI trajectories with their native Viser appearance."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import imageio.v3 as iio
import numpy as np
import robotic as ry
import viser
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "all_scaling"):
    sys.path.insert(0, str(directory))

import study_lib  # noqa: E402
from common import restore_mrbench_path  # noqa: E402
from render_3d_video import records, sampled_states  # noqa: E402
from mr_shortcutting.api import Proposal  # noqa: E402
from mr_shortcutting.shortcutters import (  # noqa: E402
    _candidate,
    _interpolate_mode_safe,
)


def read(path: Path):
    with path.open() as stream:
        return json.load(stream)


def restored_path(env, result, which):
    """Restore either a materialized planner path or an offline lineage."""
    if "selected_path" in result:
        return restore_mrbench_path(env, records(result, which))

    source = read(Path(result["source_path"]))
    raw = restore_mrbench_path(env, source["states"])
    if which == "initial":
        return raw
    if result["interpolation_fallback_to_raw"]:
        path = raw
    else:
        path, _ = _interpolate_mode_safe(env, raw, 0.5, validate_input=False)
    for event in result["stats"]["accepted_lineage"]:
        proposal = Proposal(
            int(event["first"]),
            int(event["last"]),
            tuple(int(robot) for robot in event["robots"]),
            source=event["source"],
        )
        candidate = _candidate(env, path, proposal)
        for offset, state in enumerate(candidate):
            path[proposal.first + offset].q = state.q
    return path


def apply_state(env, state, config, base):
    config.clear()
    config.addConfigurationCopy(base)
    env.set_to_mode(
        state.mode, config=config, use_cached=False, place_in_cache=False,
    )
    from multi_robot_multi_goal_planning.problems.rai.rai_config import get_robot_joints
    for index, robot in enumerate(env.robots):
        config.setJointState(state.q[index], get_robot_joints(config, robot))


def geometry_names(config):
    output = set()
    for frame in config.getFrames():
        vertices = np.asarray(frame.getMeshPoints())
        faces = np.asarray(frame.getMeshTriangles())
        if vertices.ndim == 2 and faces.ndim == 2 and len(vertices) and len(faces):
            output.add(frame.name)
    return output


def frame_style(frame):
    colors = np.asarray(frame.getMeshColors())
    if colors.ndim == 2 and len(colors):
        rgba = colors[0]
        color = tuple(int(value) for value in rgba[:3])
        opacity = float(rgba[3]) / 255.0 if len(rgba) > 3 and rgba[3] < 255 else None
    else:
        raw = frame.info().get("color", [0.7, 0.7, 0.7])
        color = tuple(int(255 * value) for value in raw[:3])
        opacity = float(raw[3]) if len(raw) > 3 and raw[3] < 1.0 else None
    return color, opacity


def rigid_anchor_name(frame):
    """Return the nearest joint frame that rigidly carries this geometry."""
    current = frame.getParent()
    while current is not None:
        info = current.info()
        if "joint" in info:
            return current.name
        current = current.getParent()
    return frame.name


def presentation_geometry(frame):
    """Hide collision-only proxy shapes while retaining scene geometry."""
    info = frame.info()
    return not (
        float(info.get("contact", 0)) < 0
        and not bool(info.get("visual", False))
    )


def dynamic_geometry_names(env, states, config, base, samples=32):
    """Identify meshes whose pose or presence changes anywhere on the path."""
    indices = np.unique(np.linspace(0, len(states) - 1, min(samples, len(states))).astype(int))
    reference = {}
    dynamic = set()
    for sample_index, state_index in enumerate(indices):
        apply_state(env, states[state_index], config, base)
        frames = {frame.name: frame for frame in config.getFrames()}
        names = set(reference) | set(frames)
        for name in names:
            frame = frames.get(name)
            present = False
            if frame is not None:
                vertices = np.asarray(frame.getMeshPoints())
                faces = np.asarray(frame.getMeshTriangles())
                present = (
                    vertices.ndim == 2 and faces.ndim == 2
                    and len(vertices) > 0 and len(faces) > 0
                )
            signature = None if not present else (
                np.asarray(frame.getPosition(), dtype=float),
                np.asarray(frame.getQuaternion(), dtype=float),
            )
            if sample_index == 0:
                reference[name] = signature
                continue
            initial = reference.get(name)
            if (initial is None) != (signature is None):
                dynamic.add(name)
            elif initial is not None:
                position_changed = not np.allclose(
                    initial[0], signature[0], rtol=0, atol=1e-7
                )
                quaternion_changed = min(
                    np.linalg.norm(initial[1] - signature[1]),
                    np.linalg.norm(initial[1] + signature[1]),
                ) > 1e-7
                if position_changed or quaternion_changed:
                    dynamic.add(name)
    return dynamic


def build_scene(server, config, dynamic_names):
    handles = []
    static_groups = {}
    dynamic_groups = {}
    for frame in config.getFrames():
        if not presentation_geometry(frame):
            continue
        vertices = np.asarray(frame.getMeshPoints(), dtype=np.float32)
        faces = np.asarray(frame.getMeshTriangles(), dtype=np.uint32)
        if vertices.ndim != 2 or faces.ndim != 2 or not len(vertices) or not len(faces):
            continue
        color, opacity = frame_style(frame)
        if frame.name not in dynamic_names:
            wxyz = np.asarray(frame.getQuaternion(), dtype=float)
            rotation = Rotation.from_quat((wxyz[1], wxyz[2], wxyz[3], wxyz[0]))
            world_vertices = rotation.apply(vertices) + np.asarray(frame.getPosition())
            group = static_groups.setdefault((opacity, color), [[], [], 0])
            group[0].append(world_vertices.astype(np.float32))
            group[1].append(faces + group[2])
            group[2] += len(vertices)
            continue
        anchor_name = rigid_anchor_name(frame)
        anchor = config.getFrame(anchor_name)
        frame_wxyz = np.asarray(frame.getQuaternion(), dtype=float)
        frame_rotation = Rotation.from_quat(
            (frame_wxyz[1], frame_wxyz[2], frame_wxyz[3], frame_wxyz[0])
        )
        anchor_wxyz = np.asarray(anchor.getQuaternion(), dtype=float)
        anchor_rotation = Rotation.from_quat(
            (anchor_wxyz[1], anchor_wxyz[2], anchor_wxyz[3], anchor_wxyz[0])
        )
        world_vertices = (
            frame_rotation.apply(vertices) + np.asarray(frame.getPosition())
        )
        local_vertices = anchor_rotation.inv().apply(
            world_vertices - np.asarray(anchor.getPosition())
        )
        group = dynamic_groups.setdefault(
            (anchor_name, opacity, color), [[], [], 0]
        )
        group[0].append(local_vertices.astype(np.float32))
        group[1].append(faces + group[2])
        group[2] += len(vertices)
    for index, ((opacity, color), (vertices, faces, _)) in enumerate(
        static_groups.items()
    ):
        server.scene.add_mesh_simple(
            f"static/{index}",
            vertices=np.concatenate(vertices),
            faces=np.concatenate(faces),
            color=color,
            opacity=opacity,
            material="standard",
            flat_shading=False,
            side="double",
            cast_shadow=True,
            receive_shadow=True,
        )
    for index, ((anchor_name, opacity, color), (vertices, faces, _)) in enumerate(
        dynamic_groups.items()
    ):
        handle = server.scene.add_mesh_simple(
            f"dynamic/{index}",
            vertices=np.concatenate(vertices),
            faces=np.concatenate(faces),
            color=color,
            opacity=opacity,
            material="standard",
            flat_shading=False,
            side="double",
            cast_shadow=True,
            receive_shadow=True,
        )
        handles.append((anchor_name, handle))
    return handles


def push_state(env, state, config, base, handles, transform_cache):
    apply_state(env, state, config, base)
    for handle_index, (anchor_name, handle) in enumerate(handles):
        frame = config.getFrame(anchor_name)
        visible = frame is not None
        previous = transform_cache.get(handle_index)
        if previous is None or previous[0] != visible:
            handle.visible = visible
        if not visible:
            transform_cache[handle_index] = (False, None, None)
            continue
        position = np.asarray(frame.getPosition(), dtype=np.float32)
        wxyz = np.asarray(frame.getQuaternion(), dtype=np.float32)
        if previous is None or not np.array_equal(previous[1], position):
            handle.position = position
        if previous is None or not np.array_equal(previous[2], wxyz):
            handle.wxyz = wxyz
        transform_cache[handle_index] = (True, position, wxyz)


def camera_bounds(env, states, config, base):
    points = []
    stride = max(1, len(states) // 16)
    for state in states[::stride] + [states[-1]]:
        apply_state(env, state, config, base)
        for frame in config.getFrames():
            name = frame.name.lower()
            if any(token in name for token in ("table", "floor", "world", "origin", "goal")):
                continue
            vertices = np.asarray(frame.getMeshPoints())
            faces = np.asarray(frame.getMeshTriangles())
            if vertices.ndim != 2 or faces.ndim != 2 or not len(vertices) or not len(faces):
                continue
            points.append(np.asarray(frame.getPosition(), dtype=float))
    if not points:
        raise RuntimeError("could not determine camera bounds")
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (low + high)
    span = np.maximum(high - low, 0.5)
    radius = max(float(np.linalg.norm(span)), 1.0)
    return center, span, radius, float(low[2])


def wait_for_client(server, chrome, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        clients = server.get_clients()
        if clients:
            return next(iter(clients.values()))
        if chrome.poll() is not None:
            raise RuntimeError("headless Chrome exited before connecting to Viser")
        time.sleep(0.1)
    raise TimeoutError("headless Chrome did not connect to Viser")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--which", choices=("initial", "final"), default="final")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    case = study_lib.cases()[args.case]
    with contextlib.redirect_stdout(io.StringIO()):
        env = study_lib.build_environment(case)
    result = read(args.result)
    path = restored_path(env, result, args.which)
    states = sampled_states(env, path, args.frames)

    base = ry.Config()
    base.addConfigurationCopy(env.C_orig)
    display = ry.Config()
    display.addConfigurationCopy(env.C_orig)
    center, span, radius, ground_z = camera_bounds(env, states, display, base)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(
        hdri="studio", background=False, environment_intensity=0.8,
    )
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    dynamic_names = dynamic_geometry_names(env, states, display, base)
    handles = build_scene(server, base, dynamic_names)
    transform_cache = {}
    grid_size = max(float(span[0]), float(span[1]), 2.0) * 1.35
    server.scene.add_grid(
        "ground", width=grid_size, height=grid_size, plane="xy",
        position=(float(center[0]), float(center[1]), ground_z - 0.01),
        cell_size=max(grid_size / 12.0, 0.25), section_size=max(grid_size / 3.0, 1.0),
        cell_color=(220, 222, 225), section_color=(205, 208, 212),
        plane_color=(244, 245, 247), plane_opacity=1.0, shadow_opacity=0.35,
    )
    push_state(env, states[0], display, base, handles, transform_cache)

    direction = np.asarray((1.30, -1.45, 1.10), dtype=float)
    direction /= np.linalg.norm(direction)
    camera_position = center + direction * radius * 1.35
    server.initial_camera.position = tuple(camera_position)
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.fov = math.radians(48.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="viser-render-") as profile:
        chrome = subprocess.Popen(
            [
                "google-chrome", "--headless=new", "--no-sandbox",
                "--disable-dev-shm-usage", "--hide-scrollbars",
                "--enable-webgl", "--use-angle=swiftshader",
                f"--window-size={args.width},{args.height}",
                f"--user-data-dir={profile}", f"http://127.0.0.1:{args.port}",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ffmpeg = None
        try:
            client = wait_for_client(server, chrome)
            client.camera.position = tuple(camera_position)
            client.camera.look_at = tuple(center)
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.fov = math.radians(48.0)
            server.flush()
            time.sleep(1.0)
            # Warm up shaders before the first encoded frame.
            client.get_render(args.height, args.width, transport_format="jpeg")
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
                    "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.output),
                ],
                stdin=subprocess.PIPE,
            )
            for index, state in enumerate(states):
                push_state(env, state, display, base, handles, transform_cache)
                server.flush()
                frame = client.get_render(
                    args.height, args.width, transport_format="jpeg",
                )[..., :3]
                if index == 0 and args.preview is not None:
                    args.preview.parent.mkdir(parents=True, exist_ok=True)
                    iio.imwrite(args.preview, frame)
                ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
                if (index + 1) % max(1, args.fps) == 0 or index + 1 == len(states):
                    print(f"rendered {index + 1}/{len(states)} frames", flush=True)
            ffmpeg.stdin.close()
            if ffmpeg.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        finally:
            if ffmpeg is not None and ffmpeg.poll() is None:
                ffmpeg.kill()
            chrome.terminate()
            try:
                chrome.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome.kill()
            server.stop()
    print(args.output)


if __name__ == "__main__":
    main()
