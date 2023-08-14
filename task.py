 # Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# [setup]

import os

import magnum as mn
from magnum import Vector3, Matrix4
import numpy as np

import habitat_sim
import pickle
import random
import habitat
from habitat.config import read_write

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from habitat.tasks.rearrange.articulated_agent_manager import ArticulatedAgentManager
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.config.default import get_agent_config, get_config
from habitat.articulated_agents.mobile_manipulator import ArticulatedAgentCameraParams
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")


urdf_files = {
    "hab_spot": os.path.join(data_path, "versioned_data/hab_spot_arm_1.0/urdf/spot_arm.urdf"),
}


def init_sim():
    config  = get_config("hm3d_robot_nav.yaml")
    if not os.path.exists(config.habitat.simulator.scene):
        print("Please download Habitat test data to data folder.")
    sim = habitat.sims.make_sim(
        "RearrangeSim-v0", config=config.habitat.simulator
    )
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        config.habitat.simulator.agents.main_agent.height
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return config


def simulate(sim, dt=1.0, get_frames=True, boxes = None, env = None):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    count = 0  
    while sim.get_world_time() < start_time + dt:
        if count % 15 == 0:
            ac = env.action_space.sample()
            while(ac['action'] == 'arm_action'):
                ac = env.action_space.sample()
            print(ac)
            ac['action_args']['base_vel'][1] = 0
            env.step(ac)
        if boxes is not None:
            for box in boxes:
                anti_grav_force = -1.0 * sim.get_gravity() * box.mass
                box.apply_force(anti_grav_force, [0.0, 0.0, 0.0])
                box.apply_torque([0.0, 0.01, 0.0])
        count+=1
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

def place_robot_from_agent(
    sim,
    robot_id,
    angle_correction=-1.56,
    local_base_pos=None,
):
    if local_base_pos is None:
        local_base_pos = np.array([0.0, -0.1, -2.0])
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    base_transform = mn.Matrix4.rotation(
        mn.Rad(angle_correction), mn.Vector3(1.0, 0, 0)
    )
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    robot_id.transformation = base_transform


def add_objaverse_objects(sim, dir_path = 'data/object_configs'):
    # get the physics object attributes manager
    obj_templates_mgr = sim.get_object_template_manager()
    # get the rigid object manager
    rigid_obj_mgr = sim.get_rigid_object_manager()
    obj_template_handles = obj_templates_mgr.get_template_handles(os.path.join(dir_path))
    boxes = []
    for obj_template_handle in obj_template_handles:
        random_pos = Vector3(np.random.uniform(-10,10,[1,3])[0])
        box_orientation = mn.Quaternion.rotation(mn.Deg(90.0), [-1.0, 0.0, 0.0])
        # instance and place the boxes
        new_box = rigid_obj_mgr.add_object_by_template_handle(obj_template_handle)
        new_box.translation = random_pos
        new_box.rotation = box_orientation
        boxes.append(
            new_box
        )
    return boxes, obj_template_handles
    


# This is wrapped such that it can be added to a unit test
def main(make_video=True, show_video=True):

    config = init_sim()
    with habitat.Env(config=config) as env:
        env.reset()
        sim = env._sim
        print("sim num objects ",  sim.get_rigid_object_manager().get_num_objects())
        agent = sim.articulated_agent
        agent.reconfigure()
        boxes, temp_handles = add_objaverse_objects(sim)
        place_robot_from_agent(sim, robot_id = agent.sim_obj)
        agent.params.cameras["articulated_agent_gripper_rgb"] = ArticulatedAgentCameraParams(
                        cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
                        cam_orientation=mn.Vector3(0, -1.571, 0.0),
                        attached_link_id=7,
                        relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                    )
        agent.params.cameras["articulated_agent_gripper_depth"] = ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=7,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                )
        agent.params.cameras["articulated_agent_gripper_semantic"] = ArticulatedAgentCameraParams(
                cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
                cam_orientation=mn.Vector3(0, -1.571, 0.0),
                attached_link_id=7,
                relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
            )
        agent.update()
        observations = []
        observations += simulate(sim, dt=3, boxes = boxes, get_frames=make_video, env = env)
        
        vut.make_video(
            observations,
            "articulated_agent_gripper_rgb",
            "color",
            output_path + "dynamic_control_gripper",
            open_vid=show_video,
        )
        vut.make_video(
            observations,
            "articulated_agent_arm_rgb",
            "color",
            output_path + "dynamic_control_arm",
            open_vid=show_video,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    main(make_video, show_video)
