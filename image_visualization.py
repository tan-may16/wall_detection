import cv2
import numpy as np
from queue import Queue
from realsense_depth import *
def Image_cleaning(img):
    result=cv2.GaussianBlur(img, (3, 3), cv2.BORDER_WRAP)

    return result


# img=cv2.imread('depth_image0.png')
# cv2.imshow('depth_image',img[:,:,0])
# print(np.unique(img[:,:,0]))
# print('*******')
# print(np.unique(img[:,:,1]))
# print('*******')
# print(np.unique(img[:,:,2]))
# print(img.shape)

# img[np.where(img[:,:,1]>=20)]=255


# cv2.imshow('depth_image_2',img)
# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()

dc = DepthCamera()
while True:

    ret, depth_frame, color_frame = dc.get_frame()
    # depth_frame=Image_cleaning(depth_frame)
    # depth_frame[:,200:300]=10000
    cv2.imshow("depth frame", depth_frame)
    print(np.unique(depth_frame))
    print('*******')
    print(np.unique(color_frame))
    print('###########')
    key = cv2.waitKey(1000)

    if key == 27:
        break