import cv2
import numpy as np


def computeH(src_points, dst_points):
    assert(len(src_points) == len(dst_points))
    assert(len(src_points) > 3)
    A = []
    for i in range(0, len(src_points)):
        img_point = dst_points[i][0]
        world_point = src_points[i][0]
        u = img_point[0];
        v = img_point[1];
        x_w = world_point[0];
        y_w = world_point[1];
        A.append([
            -x_w, -y_w, -1.0, 0.0, 0.0, 0.0, u*x_w, u*y_w, u 
        ]);
        A.append([
            0.0, 0.0, 0.0, -x_w, -y_w, -1.0, v*x_w, v*y_w, v
        ]);
    U, S, Vh = np.linalg.svd(A)
    ret = Vh[len(Vh) - 1]
    ret = ret / ret[len(ret) - 1]
    return np.array(ret).reshape(3, 3)




def computeB(hs):
    assert(len(hs) > 2)

    def get_v(h, i, j):
        h1 = h[:, 0]
        h2 = h[:, 1]
        h3 = h[:, 2]
        i = i - 1;
        j = j - 1;
        return [
            h1[i] * h1[j], h1[i] * h2[j] + h2[i] * h1[j], h3[i] * h1[j] + h1[i] * h3[j],
            h2[i] * h2[j], h3[i] * h2[j] + h2[i] * h3[j], h3[i] * h3[j],
        ]

    A = []

    for h in hs:
        col1 = get_v(h, 1, 2);
        col2 = [a - b for a, b in zip(get_v(h, 1, 1) , get_v(h, 2, 2))];
        A.append(col1)
        A.append(col2)

    U, S, Vh = np.linalg.svd(A)
    ret = Vh[len(Vh) - 1]
    ret = ret / ret[len(ret) - 1]  

    return np.array([[ret[0], ret[1], ret[2]],
                     [ret[1], ret[3], ret[4]],
                     [ret[2], ret[4], ret[5]],
                     ])

    