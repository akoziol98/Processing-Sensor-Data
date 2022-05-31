'''
This function is going to calculate two different distances between two quaternions:
- Absolute distance: Find the distance between two quaternions accounting for the sign ambiguity.
                     This function does not measure the distance on the hypersphere, but it takes into
                     account the fact that q and -q encode the same rotation. It is thus a good indicator for rotation similarities.

- Distance in angles:

 Input:
     q1 -> contains an array of quaternions or an individual value
     q2 -> is not mandatory. This is included to

 Output:
     quat_Abs_Dist -> absolute distance betweem two quaternions
     quat_angles -> distance in degrees between two consecutive quaternions


V 1.0 Creation of the document by David Lopez Perez 02.04.2020
29.03.2022 Python translation Agata Kozioł
'''
import numpy as np

from calculateQuatAbsoluteDistance import calculateQuatAbsoluteDistance
from calculateQuatAngle import calculateQuatAngle


def quat_distances(q1):
    q2 = -1
    # Check if they are the same size
    if q2 == -1:
        uniqueArray = 1
    else:
        if not q1.size and not q2.size:
            if q1.size != q2.size:
                raise ValueError('The sizes of both arrays are different')
            else:
                uniqueArray = 0

    # Process the data
    quat_Abs_Dist = np.zeros(shape=(len(q1)-1, 1), dtype=float)
    quat_angles = np.zeros(shape=(len(q1)-1, 1), dtype=float)
    if uniqueArray:
        for iQuat in range(len(q1)-1):
            quat_Abs_Dist[iQuat] = calculateQuatAbsoluteDistance(q1[iQuat], q1[iQuat + 1])
            quat_angles[iQuat] = calculateQuatAngle(q1[iQuat], q1[iQuat+1])

    else:
        raise ValueError('Use only one quaterion. quat_distance')
        for iQuat in range(len(q1)):
            quat_Abs_Dist[iQuat] =calculateQuatAbsoluteDistance(q1[iQuat], q2[iQuat])
            quat_angles[iQuat] = calculateQuatAngle(q1[iQuat], q2[iQuat])
    assert len(q1)-1 == len(quat_Abs_Dist) == len(quat_angles)

    return quat_Abs_Dist, quat_angles
