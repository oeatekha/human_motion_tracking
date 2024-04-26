import numpy as np
import json

def project(xyz_c, center, focal, eps=1e-5):
    """
    :param xyz_c (*, 3) 3d point in camera coordinates
    :param focal (1)
    :param center (*, 2)
    return (*, 2)
    """
    return focal * xyz_c[..., :2] / (xyz_c[..., 2:3] + eps) + center  # (N, *, 2)


def process_camera_json(file_path):
    def get_camera_matrix(intrinsic_props):
        fx, fy, cx, cy = intrinsic_props
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_camera_pose(camera_translation_props, camera_rotation_props):
        camera_translation = np.array(camera_translation_props)
        camera_rotation = np.array([[camera_rotation_props[0], camera_rotation_props[1], camera_rotation_props[2]], 
                                    [camera_rotation_props[3], camera_rotation_props[4], camera_rotation_props[5]], 
                                    [camera_rotation_props[6], camera_rotation_props[7], camera_rotation_props[8]]])
        
        # make the matrix 

    def get_camera_pose_inv(camera_translation_props, camera_rotation_props):
        camera_translation = np.array(camera_translation_props)
        camera_rotation = np.array(camera_rotation_props).reshape(3, 3)
        return np.hstack((camera_rotation.T, -camera_rotation.T @ camera_translation.reshape(3, 1)))

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Assuming the data structure is a list of dictionaries with 'intrinsics', 'translation', and 'rotation' keys

    intrinsic_props = data['intrinsics'][-1] 
    camera_translation_props = data['translation'][-1]
    camera_rotation_props = data['rotation'][-1]

    intrinsic = get_camera_matrix(intrinsic_props)
    pose = get_camera_pose(camera_translation_props, camera_rotation_props)
    pose_inv = get_camera_pose_inv(camera_translation_props, camera_rotation_props)

    return intrinsic, pose, pose_inv

def world_to_pixel(camera_point, intrinsic):
    # camera_point: 3x1 numpy array
    # intrinsic: 3x3 numpy array
    # Returns: pixel_point: 2x1 numpy array

    # Ensure camera_point is in homogeneous coordinates for projection
    camera_point_homogeneous = camera_point
    # Apply the intrinsic matrix to project the 3D camera point onto the 2D image plane
    pixel_point_projected = intrinsic @ camera_point_homogeneous.T
    # Normalize by the third component to get the pixel coordinates
    pixel_point = pixel_point_projected / pixel_point_projected[2]
    
    return pixel_point[:2].flatten()



