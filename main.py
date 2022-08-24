# Code released by Kang Tae wook (Kookmin Univ)



import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


from cv2 import inRange
global lower_white
lower_white = np.array([150,150,150])
global upper_white
upper_white = np.array([255,255,255])
global lower_yellow
lower_yellow = np.array([10,100,100])
global upper_yellow
upper_yellow = np.array([40,255,255])


def grayscale(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img_gray

def HSVscale(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return img_hsv


def color_filter(img):

    mask_white = cv2.inRange(img,lower_white,upper_white)
    white_image = cv2.bitwise_and(img,img,mask = mask_white)
    HSVscale(img)
    mask_yellow = cv2.inRange(img,lower_yellow,upper_yellow)
    yellow_image = cv2.bitwise_and(img,img,mask = mask_yellow)
    img_combine = cv2.addWeighted(white_image,1.0,yellow_image,1.0,0)
    return img_combine

def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x=", x ,"y= ", y )

'''ROI '''
def region_of_interest(img):
    mask = np.zeros((height,width),dtype="uint8")

    pts = np.array([[432,50],[640,50],[700,717],[452,717]])

    mask= cv2.fillPoly(mask,[pts],(255,255,255),cv2.LINE_AA)


    img_masked = cv2.bitwise_and(img,mask)
    return img_masked

''' 이미지 합치기 '''
def img_blending(img_src, img_dst):
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    ret, img_mask = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
    img_mask_inv = cv2.bitwise_not(img_mask)

    ''' 차선 ROI SIZE'''
    height,width = 270,800
    ''' 원본이미지에 덮어씌울 ROI SIZE만큼 SLICING'''
    img_roi = img_dst[436:706,311:1111]

    img1 = cv2.bitwise_and(img_src,img_src,mask= img_mask_inv)
    img2 = cv2.bitwise_and(img_roi,img_roi,mask = img_mask)

    dst = cv2.add(img1,img2)

    img_dst[436:706 , 311:1111] = dst

    return img_dst


'''곡률반경 계산 함수 '''
def Radius(rx, ry):
    a=0
    b=0
    c=0
    R=0
    h=0
    w=0
    if (rx[-1] - rx[0] != 0):
        a = (ry[-1] - ry[0]) / (rx[-1] - rx[0])
        b = -1
        c = ry[0] - rx[0] * (ry[-1] - ry[0]) / (rx[10] - rx[0])
        h = abs(a * np.mean(rx) + b * np.mean(ry) + c) / math.sqrt(pow(a, 2) + pow(b, 2))
        w = math.sqrt(pow((ry[-1] - ry[0]), 2) + pow((rx[-1] - rx[0]), 2))


    if h != 0:
        R = h / 2 + pow(w, 2) / h * 8

    return R*3/800

def steering_Angle(R):

    if R != 0:
        angle = np.arctan(1.04/R)
        return angle*180/np.pi





'조사창의 갯수'
nwindows = 11
'조사창의 높이 '
window_height = 40
'조사창 너비'
margin = 8
minpix= 5

'''축거 = 1.04M'''
wheel_base = 1.04
'MAIN'

cv2.namedWindow('Color')
cv2.setMouseCallback('Color',mouse_callback)

video = cv2.VideoCapture("project_video.mp4")

ret, img_frame = video.read()

height,width,channels = img_frame.shape

src = np.float32([[0,0],
                  [800,0],
                  [0,270],
                  [800,270]])

dst = np.float32([[0,0],
                  [width,0],
                  [500,height],
                  [580,height]])

M = cv2.getPerspectiveTransform(src,dst)
M_inv = cv2.getPerspectiveTransform(dst,src)



if video is None:
    print("x")
    exit(1)
if ret == False:
    print("No video")
    exit(1)



while(1):
    mask = np.zeros((height, width), dtype="uint8")
    ret, img_frame = video.read()

    img_roi = img_frame[436:706, 311:1111]

    img_warped = cv2.warpPerspective(img_roi,M,(width,height))


    img_blurred = cv2.GaussianBlur(img_warped, (5, 5), 0)

    _, L, _ = cv2.split(cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HLS))

    _, img_binary = cv2.threshold(L, 130, 255, cv2.THRESH_BINARY)

    img_masked = region_of_interest(img_binary)

    out_img = np.dstack((img_masked, img_masked, img_masked)) * 255



    ' 이하 sliding window '
    histogram = np.sum(img_masked[img_masked.shape[0] // 2:, :], axis=0)

    midpoint = 520
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = leftx_current + 90

    nz = img_masked.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []



    for window in range(nwindows):

        win_yl = img_masked.shape[0] - (window + 1) * window_height
        win_yh = img_masked.shape[0] - window * window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin



        good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh) / 2)

        rx.append(rightx_current)
        ry.append((win_yl + win_yh) / 2)

        cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
    # right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)

    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]



    img_blended = cv2.addWeighted(out_img, 1, img_warped, 0.6, 0)
    img_revwarped = cv2.warpPerspective(img_blended,M_inv,(width,height))
    img_detected = img_revwarped[0:270, 0:800]


    '이상 sliding window'

    R = Radius(lx,ly)

    angle = steering_Angle(R)


    print(angle)
    img_result = img_blending(img_detected,img_frame)
    cv2.imshow('Color', img_masked)

    key= cv2.waitKey(1)
    if key == 27:
        break




cv2.destroyAllWindows()
