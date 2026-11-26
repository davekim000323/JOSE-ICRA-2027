"""Write ``usd/g1_29dof_gripper.urdf``, the G1 29-DOF with a 1-DOF jaw gripper on the right wrist.

The right rubber hand (visual only, no collider) is replaced by

* ``right_gripper_base``    fixed to ``right_wrist_yaw_link`` at x = 0.05 m, palm block
* ``right_gripper_fixed_finger``  fixed jaw, inner face at y = -0.034 m
* ``right_gripper_finger``  moving jaw on prismatic ``right_gripper_joint`` along -y,
  0 (open, 68 mm gap) to 0.06 m, 20 N effort, 0.2 m/s

Every other link and joint is byte-identical to ``g1_29dof_rev_1_0.urdf``, so the
29 original joints keep their names, limits and the actuator groups in
``ppo_walk/g1_asset.py``. Masses add to 0.16 kg against the 0.17 kg hand removed.
Fingertip pads are boxes so that the PhysX contact patch is flat.

Run ``python tools/make_gripper_urdf.py`` from the repository root.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "usd", "g1_29dof_rev_1_0.urdf")
DST = os.path.join(ROOT, "usd", "g1_29dof_gripper.urdf")


def box_link(name, size, mass, origin="0 0 0", rgba="0.2 0.2 0.2 1"):
    sx, sy, sz = size
    ixx = mass * (sy * sy + sz * sz) / 12
    iyy = mass * (sx * sx + sz * sz) / 12
    izz = mass * (sx * sx + sy * sy) / 12
    geom = f'<box size="{sx} {sy} {sz}"/>'
    return f"""  <link name="{name}">
    <inertial>
      <origin xyz="{origin}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}"/>
    </inertial>
    <visual>
      <origin xyz="{origin}" rpy="0 0 0"/>
      <geometry>{geom}</geometry>
      <material name="{name}_mat"><color rgba="{rgba}"/></material>
    </visual>
    <collision>
      <origin xyz="{origin}" rpy="0 0 0"/>
      <geometry>{geom}</geometry>
    </collision>
  </link>
"""


GRIPPER = (
    box_link("right_gripper_base", (0.03, 0.10, 0.04), 0.10)
    + """  <joint name="right_gripper_base_joint" type="fixed">
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <parent link="right_wrist_yaw_link"/>
    <child link="right_gripper_base"/>
  </joint>
"""
    + box_link("right_gripper_fixed_finger", (0.07, 0.012, 0.03), 0.03, rgba="0.8 0.3 0.1 1")
    + """  <joint name="right_gripper_fixed_finger_joint" type="fixed">
    <origin xyz="0.05 -0.04 0" rpy="0 0 0"/>
    <parent link="right_gripper_base"/>
    <child link="right_gripper_fixed_finger"/>
  </joint>
"""
    + box_link("right_gripper_finger", (0.07, 0.012, 0.03), 0.03, rgba="0.8 0.3 0.1 1")
    + """  <joint name="right_gripper_joint" type="prismatic">
    <origin xyz="0.05 0.04 0" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <parent link="right_gripper_base"/>
    <child link="right_gripper_finger"/>
    <limit effort="20" velocity="0.2" lower="0.0" upper="0.06"/>
  </joint>
"""
)


def main():
    text = open(SRC).read()
    # Drop the right rubber hand link and its fixed joint.
    text, n_joint = re.subn(r'\s*<joint name="right_hand_palm_joint".*?</joint>', "", text, flags=re.S)
    text, n_link = re.subn(r'\s*<link name="right_rubber_hand">.*?</link>', "", text, flags=re.S)
    assert n_joint == 1 and n_link == 1, (n_joint, n_link)
    text = text.replace('<robot name="g1_29dof_rev_1_0">', '<robot name="g1_29dof_gripper">')
    text = text.rstrip()
    assert text.endswith("</robot>")
    text = text[: -len("</robot>")] + "\n" + GRIPPER + "</robot>\n"
    open(DST, "w").write(text)
    print("wrote", DST)


if __name__ == "__main__":
    main()
