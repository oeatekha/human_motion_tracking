import numpy as np
import pdb
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

last_error = None

X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])
RIGHT_FOOT_INDEX = [792, 770,771,772, 786, 787]
LEFT_FOOT_INDEX = [393, 384, 394, 395, 382, 380]
INTRINSICS = np.array([[1.5e+03, 0.0e+00, 9.6e+02],
 [0.0e+00, 1.5e+03, 5.4e+02],
 [0.0e+00, 0.0e+00, 1.0e+00]])

class PedestrianPoint:
    def __init__(self, uv, pose, c2w, ped_id):
        self.uv = np.array(uv)
        self.pose = np.array(pose)
        self.c2w = c2w
        self.ped_id = ped_id


def normalize(v):
    return v / np.linalg.norm(v)

def get_pedestrian_points(pedestrian2d3d, t, n):
    """
    Get all pedestrian points at a given time frame that match a given pedestrian ID.

    Parameters:
    - pedestrian2d3d: The main dictionary containing all pedestrian points.
    - t: The time frame to consider.
    - n: The pedestrian ID to match.

    Returns:
    - A list containing only the pedestrian points that match the given ID at the given time frame.
    """
    # Initialize an empty list to store the matching pedestrian points
    matching_points = []

    # Check if the time frame exists in the main dictionary
    if t in pedestrian2d3d:
        # Iterate over all pedestrian points at this time frame
        for uv_tuple, pedestrian_point in pedestrian2d3d[t].items():
            # Check if the pedestrian ID matches the given ID
            if pedestrian_point.ped_id == n:
                # Add the pedestrian point to the matching points list
                matching_points.append(pedestrian_point)

    # Return the list of matching pedestrian points
    return matching_points



def apply_transformation_to_points(points, R, distance, normal):
    # Rotate each point
    rotated_points = points @ R.T  # Assuming points are in rows
    
    # Calculate the translation amount
    translation_distance = distance
    translation_vector = normalize(normal) * translation_distance
    
    # Translate each point
    translated_points = rotated_points + translation_vector
    
    return translated_points

# using a uv find the x,y in the world
# using the intrinsics matrix
def world_to_uv(world_coords, intrin):

    x, y, z = world_coords
    fx, fy = intrin[0, 0], intrin[1, 1]
    cx, cy = intrin[0, 2], intrin[1, 2]
    
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    
    return (int(u), int(v))

def world_2_uv(world_coords, intrin, R, t):
    # Convert the world coordinates to a numpy array if not already
    world_coords = np.array(world_coords).reshape(3, 1)

    # Apply rotation and translation to transform world coordinates into camera coordinates
    camera_coords = R @ world_coords + t.reshape(3, 1)

    # Extract x, y, z from camera coordinates
    x, y, z = camera_coords.flatten()

    # Extract intrinsic parameters
    fx, fy = intrin[0, 0], intrin[1, 1]
    cx, cy = intrin[0, 2], intrin[1, 2]

    # Project coordinates using the intrinsic matrix
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    
    return (int(u), int(v))

    # Return integer pixel co

def uv_to_world(uv, z, intrin):

    u, v = uv
    fx, fy = intrin[0, 0], intrin[1, 1]
    cx, cy = intrin[0, 2], intrin[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.array([x, y, z]) 

def dist_point_to_plane(plane, point):
    a, b, c, d = plane
    x, y, z = point.flatten()
    return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)


def compute_shadow_point(plane, point):
    a, b, c, d = plane
    x, y, z = point.flatten()
    t = -d / (a*x + b*y + c*z)
    shadow_point = t * np.array([x, y, z])
    return shadow_point

def calculate_shadow_points(pedestrian, pred_depth, INTRINSICS, trR, plane_offset, trplane, FINAL_WIDTH, FINAL_HEIGHT, translation=np.array([0, 0, 0])):
    """
    Calculate the shadow points of 3D points on a given plane.

    Parameters:
    - plane_params (tuple): The coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0.
    - points_3d (numpy.ndarray): An array of 3D points where each point is represented as [x, y, z].

    Returns:
    - numpy.ndarray: An array of shadow points on the plane.
    """
    print("Calculating shadow points...")

    print("Translation: ", translation)
    p_shadows = []
    h_shadows = []
    human_poses_corrected = []
    pred_poses_corrected = []

    normal = trplane[:3]
    X_PRIME_AXIS = normalize(np.cross(normal, X_AXIS))
    Y_PRIME_AXIS = normalize(np.cross(normal, X_PRIME_AXIS))



    for point in pedestrian:

        #print(point)

        # Get the uv
        u, v = point.uv
        u = int(u)
        v = int(v)
        pose_human = point.pose

        #print("uv, pose", u, v, pose_human)

        if u < 0 or u >= FINAL_WIDTH or v < 0 or v >= FINAL_HEIGHT:
            continue

        Z_pred = np.array(pred_depth)[v, u]
        if Z_pred == 0:
            continue
        
        pose_pred = uv_to_world((u, v), Z_pred, INTRINSICS)
        


        pose_human = np.array(pose_human).reshape(-1, 3) + translation.flatten()
        
        human_poses_corrected.append(pose_human)
        pose_pred = np.array(pose_pred).reshape(-1, 3) 
        pred_poses_corrected.append(pose_pred)
        # Calculate the shadow points
        human_shadow_points = compute_shadow_point(trplane, pose_human)
        pred_shadow_points = compute_shadow_point(trplane, pose_pred)

        h_shadows.append(human_shadow_points)
        p_shadows.append(pred_shadow_points)


    return p_shadows, h_shadows, np.array(human_poses_corrected), np.array(pred_poses_corrected)


def calculate_normal_translation(pedestrian, pred_depth, INTRINSICS, FINAL_WIDTH, FINAL_HEIGHT):


    ph_norms = []

    for point in pedestrian:

        #print(point)

        # Get the uv
        u, v = point.uv
        u = int(u)
        v = int(v)
        pose_human = point.pose

        #print("uv, pose", u, v, pose_human)

        if u < 0 or u >= FINAL_WIDTH or v < 0 or v >= FINAL_HEIGHT:
            continue

        Z_pred = np.array(pred_depth)[v, u]
        if Z_pred == 0:
            continue
        
        pose_pred = uv_to_world((u, v), Z_pred, INTRINSICS)
        pose_pred = np.array(pose_pred).reshape(-1, 3)
        pose_human = np.array(pose_human).reshape(-1, 3) 

        ph_vector = np.array([0,0,0]) - pose_human
        # Make UnitV    
        ph_norm = ph_vector / np.linalg.norm(ph_vector)

        ph_norms.append(ph_norm)

    # Find the norm that is most common what is the line that fits the most common direction, like random sample consensus
    
    # Find the most common direction
    ph_norms = np.array(ph_norms)
    ph_final = np.mean(ph_norms, axis=0)
    # Print Error

    print("ph_final: ", ph_final)
    #print("ph_norms: ", ph_norms)
    print("Final Error: ", np.sum((ph_norms - ph_final) ** 2) / len(ph_norms))
    
       
    return ph_final





def objective_function(alpha, Vt, pedestrian, pred_depth, gt_points, INTRINSICS, trplane, FINAL_WIDTH, FINAL_HEIGHT):
    global last_error

    print("gtpoint  shape: ", gt_points.shape)
    print("Alpha: ", alpha)

    # Get Feet GT POINTS, Vt is translaion vector
    gt_points = np.array(gt_points)
    gt_points = gt_points.reshape(-1, 3)
    left_points = gt_points[LEFT_FOOT_INDEX]
    right_points = gt_points[RIGHT_FOOT_INDEX]

    #print("Left Points: ", left_points)
    #print("Right Points: ", right_points)

    # Get Distance between feet and plane for every point in left and right
    left_distances = []
    right_distances = []

    for point in left_points:
        left_distances.append(dist_point_to_plane(trplane, point))

    for point in right_points:
        right_distances.append(dist_point_to_plane(trplane, point))

    left_distances = np.array(left_distances)
    right_distances = np.array(right_distances)

    lfavg = np.mean(left_distances)
    rfavg = np.mean(right_distances)

    body_points = left_points if lfavg < rfavg else right_points

    # Use Translation Vector * alpha and try to minimize the distance to the plane
    t_a = Vt * alpha

    # Calculate the error
    error = 0
    for point in body_points:
        error += dist_point_to_plane(trplane, (point + t_a))

    error = error / len(body_points)
    

    print("Error: ", error)
    print("Distance: ", dist_point_to_plane(trplane, (body_points[0] + t_a)))
    last_error = error
    return error

    

def optimize_translation(Vt, pedestrian, pred_depth, gt_points, INTRINSICS, trplane, FINAL_WIDTH, FINAL_HEIGHT):
    initial_alpha = 10  # Starting guess
    result = minimize(
        objective_function, 
        initial_alpha, 
        args=(Vt, pedestrian, pred_depth, gt_points, INTRINSICS, trplane, FINAL_WIDTH, FINAL_HEIGHT),
    )

    return result.x[0] * Vt
    


    

