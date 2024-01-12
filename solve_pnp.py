import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic.

    Input:
        Pc: 4x2 numpy array of pixel coordinates of the April tag corners in (x, y) format.
        Pw: 4x3 numpy array of world coordinates of the April tag corners in (x, y, z) format.
        K: 3x3 intrinsic camera matrix (default is the identity matrix).

    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc).
        t: (3, ) numpy array describing camera translation in the world (t_wc).
        
    Author: Kevin Paulose, 63031616
    """

    ##### STUDENT CODE START #####

    # Homography Approach: Pose from Projective Transformation

    # Remove the 3rd column of zeroes in Pw
    Pw = Pw[:, :2]

    H = est_homography(Pw, Pc)
    
    # Compute H_prime
    H_prime = np.linalg.inv(K) @ H
    #print(H)
    #print(np.linalg.inv(K))
    #print(H_prime)

    # Calculate indivudla h1, h2, h3 vectors (prime here since H_prime)
    h1_prime = H_prime[:, 0]
    h2_prime = H_prime[:, 1]
    h3_prime = H_prime[:, 2]
    h12_prime = np.cross(h1_prime, h2_prime)
    #print(h1prime)

    # Stack the columns of H_prime and h12_prime
    h = np.column_stack((H_prime[:, :2], h12_prime))

    # Taking SVD of h
    U, S, Vt = np.linalg.svd(h)
    V = Vt.T
    
    S[-1] = np.linalg.det(U @ Vt)

    # Calculate A matrix
    A = np.eye(3)
    A[2,2] = np.linalg.det(U @ Vt)

    # Calculate R and t
    R = U @ A @ V.T
    t = h3_prime/ np.linalg.norm(h1_prime)

    # Convert from camera to world -> world to camera
    R = np.linalg.inv(R)
    t = -R @ t

    ##### STUDENT CODE END #####

    return R, t

