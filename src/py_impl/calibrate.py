import cv2
import numpy as np
import computeM as cm
import os

def generate_wp(pattern, length):
    width = pattern[0]
    height = pattern[1]
    ret = [[] for i in range(width * height)]

    for i in range(0, height):
        for j in range(0, width):
            ret[i * width + j].append([j * length, i  * length])
    return np.asarray(ret)
    
    

def main(paths):
    wp = generate_wp((11, 8), 20)
    hs = []

    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == False:
            continue

        M, mask = cv2.findHomography(wp, corners)
        vh = cm.computeH(wp, corners)
        print(vh)
        hs.append(vh)
    
    b = cm.computeB(hs)
    print(b)

    # resize_img = cv2.resize(img, None, fx=0.3, fy=0.3)

    # cv2.imshow("img", resize_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 



if __name__=="__main__":
    paths = list(range(0, 10))
    str_paths = ["../../cali/10000" + str(p) + ".png" for p in paths]
    print(str_paths)
    main(str_paths)
    
