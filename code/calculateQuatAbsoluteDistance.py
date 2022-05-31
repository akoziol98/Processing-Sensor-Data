'''
This snipplet is going to calculate the absolute distance between
quaternions accounting for the sign ambiguity.

Based on: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py

 Input:
     q1 -> contains the quaternion at time point 1
     q2 -> contains the quaternion at time point 2

 Output:
     quaternionDist -> absolute distance between two quaternions

V1.0 creation of the document by David Lopez 05.04.2020
'''
from numpy.linalg import norm
def calculateQuatAbsoluteDistance(q1, q2):
# Calculate the distance between quaternions
    q1_minus_q2 = q1 - q2
    q1_plus_q2  = q1 + q2
    d_minus = norm(q1_minus_q2)
    d_plus = norm(q1_plus_q2)
    if d_minus < d_plus:
        quaternionDist = d_minus
    else:
        quaternionDist = d_plus

    return quaternionDist