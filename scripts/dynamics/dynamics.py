from __future__ import print_function
import pinocchio as pin
import numpy as np


# computation of gravity terms
def getg(q, model, data):
    qd = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    qdd = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
   
    g = pin.rnea(model, data, q,qd ,qdd)     
    return g


# computation of generalized mass matrix
def getM(q, model, data ):
    n = len(q)
    M = np.zeros((n,n))
    qd = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(n):
        ei = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ei[i] = 1
        
        g = getg(q, model, data)
        tau = pin.rnea(model, data, q, qd ,ei) -g

        M[:5,i] = tau

    return M


# computation of Coriolis and centrifugal terms
def getC(q, qd, model, data):
    qdd = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
   
    g = getg(q,model, data)
    C = pin.rnea(model, data, q, qd, qdd) - g    
    
    return C      


# compute the forward dynamics
def forwardDynamics(q, qd, tau, model, data):
    qdd = np.zeros_like(q)
    
    # Compute the inverse dynamics
    g = getg(q, model, data)
    C = getC(q, qd, model, data)
    M = getM(q, model, data)

    # compute the bias term h
    h = C +g

    # Solve for joint accelerations
    qdd = np.linalg.solve(M, tau - h)
    
    return qdd





