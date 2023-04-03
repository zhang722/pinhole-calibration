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
    return ret
    


def generate_wp(pattern, length):
    width = pattern[0]
    height = pattern[1]
    ret = [[] for i in range(width * height)]

    for i in range(0, height):
        for j in range(0, width):
            ret[i * width + j].append([j * length, i  * length])
    return np.asarray(ret)
    

img = cv2.imread("../cali.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
if ret == True:
    for corner in enumerate(corners):
        # if corner[0] > 7:
        #     break
        img = cv2.circle(img, corner[1].astype(np.int64)[0], 50, (0, 255, 0), 5)

wp = generate_wp((7, 5), 30.5)
print(wp)
img_w = np.zeros((500, 500), np.uint8)
for p in enumerate(wp):
    idx = p[0]
    # if idx > 6: 
    #     break
    p_i = p[1].astype(np.int64)
    img_w = cv2.circle(img_w, (p_i[0][0], p_i[0][1]), 10, (255, 0, 0), 1)
# cv2.imshow("real img", img_w)



M, mask = cv2.findHomography(wp, corners)
vh = computeH(wp, corners)
print(vh)

img_w_warp = cv2.warpPerspective(img_w, M, (5000, 5000))
resize_img = cv2.resize(img_w_warp, None, fx=0.1, fy=0.1)
cv2.imshow("real img", resize_img)

for p in wp:
    p_homo = [p[0][0], p[0][1], 1.0]
    p_i = M.dot(p_homo)
    p_i_norm = [p_i[0] / p_i[2], p_i[1] / p_i[2], 1.0]
    img = cv2.circle(img, (int(p_i_norm[0]), int(p_i_norm[1])), 50, (255, 0, 0), 5)

resize_img = cv2.resize(img, None, fx=0.3, fy=0.3)

cv2.imshow("img", resize_img)

cv2.waitKey(0)
cv2.destroyAllWindows() 