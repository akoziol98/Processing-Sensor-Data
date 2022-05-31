# function quaternionAngle = calculateQuatAngle(q1,q2)
'''
This snipplet is going to calculate the absolute distance between
quaternions in degrees of orientation

 Input:
     q1 -> contains the quaternion at time point 1
     q2 -> contains the quaternion at time point 2

 Output:
     quaternionAngle -> differences in orientation betweem two quaternions

V1.0 creation of the document by David Lopez 05.04.2020
'''
import numpy as np
from numpy.linalg import norm
from math import atan2

def quatConj(q):
    """Return the conjugate of quaternion `q`.
    Copied from https://github.com/yxiong/xy_python_utils/blob/master/xy_python_utils/quaternion.py
    """
    return np.append(q[0], -q[1:])

def quaternion_multiply(Q0,Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)
    Copied from https://automaticaddison.com/how-to-multiply-two-quaternions-together-using-python/
    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]

    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion

def calculateQuatAngle(q1,q2):
# Calculate the difference between quaternions
    q12 = quaternion_multiply(quatConj(q1), q2)
    quaternionAngle = 2 * atan2(norm(q12[1:3]), q12[0])
    return quaternionAngle
