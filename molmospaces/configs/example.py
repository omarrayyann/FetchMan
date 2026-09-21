"""Minimal example task: pick up a bowl in one fixed house.

This is the config the README walks through. It deliberately pins the scene and
turns scene resampling off, so every reset gives the same house and the run is
reproducible on a single GPU. `bowl_mixed.py` instead points `scene` at a
scene list and randomizes across tens of thousands of houses.

Domain randomization is left on but mild: textures, lighting, object placement
and camera jitter still vary per episode, which is what makes the collected
data transfer. Set the `randomize_*` flags to False to see the same frame twice.
"""
import ml_collections

# Where the robot starts relative to the object. `nav` spawns it a short walk
# away so the episode includes locomotion; `grasp` starts it already in front
# of the object. Weights are the sampling probabilities.
NAV_PROFILE = dict(
    spawn_at_grasp=0,
    spawn_radius_min=0.45,
    spawn_radius_max=2.5,
    arm_init_radius=0.3,
    pregrasp_xyz_noise=0.15,
    pregrasp_rot_noise=0.15,
    action_noise_std=0.006,
    face_yaw_offset=0.3,
    spawn_along_line=True,
    walk_dist_min=0.2,
)

GRASP_PROFILE = dict(
    spawn_at_grasp=True,
    arm_init_radius=0.5,
    pregrasp_xyz_noise=0.15,
    pregrasp_rot_noise=0.15,
    action_noise_std=0.002,
    face_yaw_offset=0.3,
)


def get_config():
    """Example task config: bowls in one fixed house."""
    return ml_collections.ConfigDict(dict(
        # One fixed house. Swap for "scene_lists/bowl_scenes.txt" to train
        # across every house that contains a bowl.
        scene="sample_scenes/procthor-10k-val/val_1.xml",
        objects="bowl",
        seed=0,

        # Fixed scene: never resample the house between episodes.
        randomize_scene=False,

        # Still vary which bowl, where it sits, and how the room looks.
        randomize_object=True,
        randomize_placement=True,
        randomize_textures=True,
        max_textures=5,
        randomize_lighting=True,
        randomize_height=True,
        randomize_height_min=0.1,
        randomize_height_max=0.9,
        randomize_height_favored=0.75,
        randomize_robot_height=True,
        randomize_robot_height_min=0.74,
        randomize_robot_height_max=0.77,

        # Camera jitter, applied to both the head and the wrist cameras.
        wrist_camera_pos_noise=0.01,
        wrist_camera_rot_noise=0.0349,
        wrist_camera_fovy_noise=2.0,
        head_camera_pos_noise=0.01,
        head_camera_rot_noise=0.0349,
        head_camera_fovy_noise=2.0,
        head_camera_distortion_noise=0.2,

        object_noise=0.0,
        spawn_visibility_check=True,
        grasp_spawn_radius_min=0.2,
        grasp_spawn_radius_max=0.5,
        terminate_before_grasp_collision=True,
        terminate_on_grasp_collision=True,

        launch_viewer=False,
        camera_size=(224, 384),
        skill_profiles=[
            ("nav",   0.9, NAV_PROFILE),
            ("grasp", 0.1, GRASP_PROFILE),
        ],
    ))
