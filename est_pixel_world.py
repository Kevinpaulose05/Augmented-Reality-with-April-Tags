import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordinates of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    
    Author: Kevin Paulose, 63031616
    """

    ##### STUDENT CODE START #####

    # Append a column of ones to the pixels to convert them to homogeneous coordinates
    Pc_homogeneous = np.column_stack((pixels, np.ones(pixels.shape[0])))

    Rw_inv = np.linalg.inv(R_wc)

    # Camera translation in world frame
    tw_inv = -Rw_inv @ t_wc

    # Create the [R | t] matrix by taking the first two columns of Rw_inv and appending tw_inv
    Rt = np.column_stack((Rw_inv[:, :2], tw_inv))

    Pw = np.zeros((Pc_homogeneous.shape[0], 3))

    for i in range(Pc_homogeneous.shape[0]):
        # Calculate the 3D point in world coordinates by transforming Pc to Pw
        Pw[i, :] = np.dot(np.linalg.inv(K @ Rt), Pc_homogeneous[i, :].reshape(3, 1)).flatten()

        # Normalize the 3D point by dividing by its last element (homogeneous coordinate)
        Pw[i, :-1] /= Pw[i, -1]

        # Set the last element to 0 to represent a 3D point in the world frame
        Pw[i, -1] = 0

    ##### STUDENT CODE END #####
    return Pw
