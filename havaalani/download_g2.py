"""Agibot Genie Sim G2 MuJoCo modelini indirir.

Kaynak: https://huggingface.co/datasets/agibot-world/GenieSimAssets
Lisans: CC BY-NC-SA 4.0. Model yeniden çizilmez; yalnızca resmi MJCF alınır.
"""

from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "vendor" / "mujoco"
BASE = "https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/main/mujoco"

FILES = [
    "G2_place_workpiece.xml",
    "robot/parts/G2/chassis/asset.xml",
    "robot/parts/G2/chassis/link.xml",
    "robot/parts/G2/chassis/actuator_sensor.xml",
    "robot/parts/G2/body_t2_crs_pw/asset.xml",
    "robot/parts/G2/body_t2_crs_pw/link.xml",
    "robot/parts/G2/body_t2_crs_pw/actuator_sensor.xml",
    "robot/parts/G2/arm_crs_pw/asset.xml",
    "robot/parts/G2/arm_crs_pw/link.xml",
    "robot/parts/G2/arm_crs_pw/actuator_sensor.xml",
    "robot/parts/G2/hand_pw/asset.xml",
    "robot/parts/G2/hand_pw/gripper_left.xml",
    "robot/parts/G2/hand_pw/gripper_right.xml",
    "robot/parts/G2/hand_pw/actuator_left_sensor.xml",
    "robot/parts/G2/hand_pw/actuator_right_sensor.xml",
    "robot/parts/meshes/G2/T2/base_link.STL",
    "robot/parts/meshes/G2/T2/body_link1.STL",
    "robot/parts/meshes/G2/T2/body_link2.STL",
    "robot/parts/meshes/G2/T2/body_link3.STL",
    "robot/parts/meshes/G2/T2/body_link4.STL",
    "robot/parts/meshes/G2/T2/body_link5.STL",
    "robot/parts/meshes/G2/T2/head_link1.STL",
    "robot/parts/meshes/G2/T2/head_link2.STL",
    "robot/parts/meshes/G2/T2/head_link3.STL",
    "robot/parts/meshes/G2/T2/chassis_lwheel_front_link1.STL",
    "robot/parts/meshes/G2/T2/chassis_lwheel_front_link2.STL",
    "robot/parts/meshes/G2/T2/chassis_lwheel_rear_link1.STL",
    "robot/parts/meshes/G2/T2/chassis_lwheel_rear_link2.STL",
    "robot/parts/meshes/G2/T2/chassis_rwheel_front_link1.STL",
    "robot/parts/meshes/G2/T2/chassis_rwheel_front_link2.STL",
    "robot/parts/meshes/G2/T2/chassis_rwheel_rear_link1.STL",
    "robot/parts/meshes/G2/T2/chassis_rwheel_rear_link2.STL",
    "robot/parts/meshes/arm/crs/arm_base_link.STL",
    "robot/parts/meshes/arm/crs/arm_link1.STL",
    "robot/parts/meshes/arm/crs/arm_link2.STL",
    "robot/parts/meshes/arm/crs/arm_link3.STL",
    "robot/parts/meshes/arm/crs/arm_link4.STL",
    "robot/parts/meshes/arm/crs/arm_link5.STL",
    "robot/parts/meshes/arm/crs/arm_link6.STL",
    "robot/parts/meshes/arm/crs/arm_link7.STL",
    "robot/parts/meshes/gripper/workpiece/base_link.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_01.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_02.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_03.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_04.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_05.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_06.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_07.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_08.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_09.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_10.STL",
    "robot/parts/meshes/gripper/workpiece/convex/wuliao_Collision_11.STL",
]

def _strip_pushers():
    """Eski mavi itici kutuları gripper XML'inden çıkarır."""
    import re

    for relative in (
        "robot/parts/G2/hand_pw/gripper_left.xml",
        "robot/parts/G2/hand_pw/gripper_right.xml",
    ):
        path = MODEL / relative
        text = path.read_text()
        cleaned = re.sub(r"\n\s*<geom name=\"pusher_[lr]\"[^>]*/>", "", text)
        if cleaned != text:
            path.write_text(cleaned)


def ensure_model():
    marker = MODEL / "robot/parts/meshes/arm/crs/arm_link7.STL"
    if marker.exists() and (MODEL / "G2_place_workpiece.xml").exists():
        _strip_pushers()
        return
    print("G2 MuJoCo modeli indiriliyor (yaklaşık 35 MB)...")
    for relative in FILES:
        destination = MODEL / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(f"{BASE}/{relative}", destination)
        print(" ", relative)
    _strip_pushers()
    print("G2 modeli hazır.")
