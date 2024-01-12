import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    Author: Kevin Paulose, 63031616
    """

    ##### STUDENT CODE START #####

    #Implentation of Grunert's solution (1841)

    R = np.eye(3)
    t = np.zeros([3])

    # Extract points from world coordinates
    p1 = Pw[1,:]
    p2 = Pw[2,:]
    p3 = Pw[3,:]

    # Calcaulte camera parameters
    f = (K[0][0] + K[1][1]) / 2
    offset = np.array([K[0][2], K[1][2]])
    uv1 = Pc[1,:] - offset
    uv2 = Pc[2,:] - offset
    uv3 = Pc[3,:] - offset
    uv1 = np.append(uv1, f)
    uv2 = np.append(uv2, f)
    uv3 = np.append(uv3, f)

    # define a,b,c, and cos of alpha,beta,gamma
    a = np.linalg.norm(p2-p3)
    b = np.linalg.norm(p1-p3)
    c = np.linalg.norm(p1-p2)

    # unit vectors j1, j2, j3
    j1 = uv1 / np.linalg.norm(uv1)
    j2 = uv2 / np.linalg.norm(uv2)
    j3 = uv3 / np.linalg.norm(uv3)

    calpha = np.dot(j2, j3)
    cbeta = np.dot(j1, j3)
    cgamma = np.dot(j1, j2)

    # Calcaulate coefficients of the 4th degree polynomial in v
    co1 = (a ** 2 - c ** 2) / b ** 2
    co2 = (a ** 2 + c ** 2) / b ** 2
    co3 = (b ** 2 - c ** 2) / b ** 2
    co4 = (b ** 2 - a ** 2) / b ** 2
    A4 = ((co1 - 1) ** 2) - 4 * ((c / b) ** 2) * (calpha ** 2)
    A3 = 4 * (co1 * (1 - co1) * cbeta - (1 - co2) * calpha * cgamma + 2 * ((c / b) ** 2) * (calpha ** 2) * cbeta)
    A2 = 2 * ((co1 ** 2) - 1 + 2 * (co1 ** 2) * (cbeta ** 2) + 2 * co3 * (calpha ** 2) - 4 * co2 * calpha * cbeta * cgamma + 2 * co4 * (cgamma ** 2))
    A1 = 4 * (-co1 * (1 + co1) * cbeta + 2 * ((a / b) ** 2) * (cgamma ** 2) * cbeta - (1 - co2) * calpha * cgamma)
    A0 = ((1 + co1) ** 2) - 4 * ((a / b) ** 2) * (cgamma ** 2)
    
    # A is a array of the coefficients
    A = np.array([A4,A3,A2,A1,A0])
    A = np.ravel(A)
    # print(A)
    
    # Calculate the roots of above polynomial
    sols = np.roots(A.T)
    sols = np.array(sols)
    sols = np.real(sols)
    sols = sols[sols>0]

    R_w = None
    t_w = None
    least_error = 10^5  # To reduce error gap and because 5 is my lucky number :)
    for i in range(len(sols)):

        # Calculate u
        u = ((-1 + (a ** 2 - c ** 2) / b ** 2) * np.square(sols[i]) - 2 * ((a ** 2 - c ** 2) / b ** 2) * cbeta * sols[i] + 1 + ((a ** 2 - c ** 2) / b ** 2)) / (2 * (cgamma - sols[i] * calpha))

        # Calculate s1, s2, and s3
        s1 = np.sqrt((b ** 2) / (1 + sols[i] ** 2 - 2 * sols[i] * cbeta))
        s2 = u * s1
        s3 = sols[i] * s1

        # Stack the s1, s2, s3 arrays along the second axis to form the p matrix
        p = np.vstack((s1 * j1, s2 * j2, s3 * j3))

        # Calculate R and t for Procrustes transformation
        R, t = Procrustes(Pw[1:], p)

        # Calculate P
        P = np.dot(K, np.dot(R, Pw[0, :].T) + t)
        
        P_last = P[-1]
        P_Normal = P/P_last
        P_Normal = P_Normal[:-1]

        #calculate error wrt  the unused coordinate in gurnert's equation?
        diff = (P_Normal - Pc[0])
        error = np.linalg.norm(diff)

        if error < least_error:
            least_error = error
            R_w = R
            t_w = t

    Rw_inv = np.linalg.inv(R_w)

    # Return the camera orientation and translation
    return (Rw_inv, - Rw_inv @ t_w)

def Procrustes(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    
    # Calculate and remove mean of the points in X and Y (centering the data)
    X_prime = np.mean(X, axis = 0)
    Y_prime = np.mean(Y, axis = 0)
    X = (X - X_prime).T
    Y = (Y - Y_prime).T
    
    # SVD
    U, S, Vt = np.linalg.svd(X @ Y.T)
    V = np.transpose(Vt)
    # print(V @ np.transpose(U))
    
    # Ensure that S is a 3x3 diagonal matrix with the determinant as the last element
    S = np.eye(3)
    S[2][2] = np.linalg.det(V @ np.transpose(U))

    # Calculate the rotation matrix using the SVD
    R = V @ S @ np.transpose(U)
    # print(np.linalg.det(R))

    # Translation vector t by applying the rotation to X_prime and subtracting it from Y_prime.
    t = Y_prime - R @ X_prime
    t = np.array(t).squeeze() #Ensure t is a 1D array
    
    ##### STUDENT CODE END #####
    
    return R, t