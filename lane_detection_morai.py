#Code Released by Kang Tae wook

#! /usr/bin/env python

from email.mime import image
import rospy
import math
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge,CvBridgeError
from morai_msgs.msg import CtrlCmd
#yelllow_filtering parameter

global lower_white
lower_white = np.array([150,150,150])
global upper_white
upper_white = np.array([255,255,255])
global lower_yellow
lower_yellow = np.array([10,100,100])
global upper_yellow
upper_yellow = np.array([130,255,255])

class camera_sim():

	def __init__(self):
		rospy.init_node('image_to_receiver', anonymous=False)
		self.ackermann_pub = rospy.Publisher("ctrl_cmd", CtrlCmd, queue_size=1)
		self.subcam = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)
		self.bridge = CvBridge()
		rospy.on_shutdown(self.cam_shutdown)
		rospy.spin()
		
	def mouse_callback(self,event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			print("x=", x ,"y= ", y )
	
	
	def Radius(self,rx, ry):
		a=0
		b=0
		c=0
		R=0
		h=0
		w=0
		if (rx[-1] - rx[0] != 0):
			a = (ry[-1] - ry[0]) / (rx[-1] - rx[0])
			b = -1
			c = ry[0] - rx[0] * (ry[-1] - ry[0]) / (rx[-1] - rx[0])
			h = abs(a * np.mean(rx) + b * np.mean(ry) + c) / math.sqrt(pow(a, 2) + pow(b, 2))
			w = math.sqrt(pow((ry[-1] - ry[0]), 2) + pow((rx[-1] - rx[0]), 2))


		if h != 0:
			R = h / 2 + pow(w, 2) / h * 8

		return R*3/160
		
	def steeringAngle(self,R,lx,rx):
		
		
		angle = np.arctan(1.04/R)
		if rx[0]>730 : 
			while not rx[0]<730:
				return -0.008
		#elif rx[0]<720:
		#	while not [rx]>720:
		#		return 0.008
		else :
			if lx[-1]>lx[0]:
				return angle * -15.0
			elif lx[-1]<lx[0]:
				return angle * 15.0
			elif lx[-1] == lx[0]:
				return 0	
		
			
			
		
		
	def callback(self, data):
        # simulation cam -> cv2
		try:
			img_frame = self.bridge.compressed_imgmsg_to_cv2(data)
		except CvBridgeError as e:
			print("converting error")
			print(e)
			
		height, width, channel = img_frame.shape
		
		ackermann_msg = CtrlCmd()
		
		ackermann_msg.velocity = 40
		ackermann_msg.longlCmdType = 2
		
		cv2.namedWindow('frame')
		cv2.setMouseCallback('frame',self.mouse_callback)
		
  
		
		# perspective_transform, binary
		src = np.float32([[0,0],
                  [width,0],
                  [0,height],
                  [width,height]])

		dst = np.float32([[0,0],
                  [width,0],
                  [565,height],
                  [715,height]])

		M = cv2.getPerspectiveTransform(src,dst)
		M_inv = cv2.getPerspectiveTransform(dst,src)
        
		img_roi = img_frame[0:height, 0:width]

		img_warped = cv2.warpPerspective(img_roi,M,(width,height))


		img_blurred = cv2.GaussianBlur(img_warped, (5, 5), 0)

		_, L, _ = cv2.split(cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HLS))

		_, img_binary = cv2.threshold(L, 170, 255, cv2.THRESH_BINARY)
		
		# yellow_filtering
		mask_white = cv2.inRange(img_blurred, lower_white, upper_white)
		white_image = cv2.bitwise_and(img_blurred, img_blurred, mask=mask_white)
		white_gray = cv2.cvtColor(white_image,cv2.COLOR_BGR2GRAY)
		_, white_binary = cv2.threshold(white_gray, 170, 255, cv2.THRESH_BINARY)
		
		#FILTERING
		hsv_image = cv2.cvtColor(img_blurred,cv2.COLOR_BGR2HSV)
		mask_yellow = cv2.inRange(img_blurred,lower_yellow,upper_yellow)
		yellow_image = cv2.bitwise_and(img_blurred,img_blurred,mask = mask_yellow)

		img_bgr = cv2.cvtColor(yellow_image,cv2.COLOR_HSV2BGR)
		img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
		_, yellow_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

		img_combined = cv2.addWeighted(white_binary,1,yellow_binary,1,0)
		
		img_binary = cv2.addWeighted(img_combined,1,img_binary,1,0)
		
        # ROI(region of Interest) 
        
		mask = np.zeros((height,width),dtype="uint8")

		pts = np.array([[450,0],[840,0],[780,720],[500,720]])
		mask= cv2.fillPoly(mask,[pts],(255,255,255),cv2.LINE_AA)

		img_masked = cv2.bitwise_and(img_binary,mask)
        
        # sliding window
        
		nwindows = 15 

		window_height = 25

		margin = 15
		minpix= 5
		
		wheel_base = 1.04
		
		# out_img = 
		out_img = np.dstack((img_masked, img_masked, img_masked)) * 255

		histogram = np.sum(img_masked[img_masked.shape[0] // 2:, :], axis=0)

		midpoint = width/2

		leftx_current = np.argmax(histogram[:midpoint])

		rightx_current = np.argmax(histogram[midpoint+30:])+midpoint+30

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

		# lfit = np.polyfit(np.array(ly), np.array(lx), 2)

		# rfit = np.polyfit(np.array(ry), np.array(rx), 2)

		out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]

		out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

		img_blended = cv2.addWeighted(out_img, 1, img_warped, 0.6, 0)
		
		img_revwarped = cv2.warpPerspective(img_blended,M_inv,(width,height))
	
		
        # Calculate Curved
		R = self.Radius(rx,ry)
		ackermann_msg.steering = self.steeringAngle(R,lx,rx)
        
		
		
		
		self.ackermann_pub.publish(ackermann_msg)
		
		cv2.imshow("frame",img_blended)
		
		key = cv2.waitKey(1)
		
		

	def cam_shutdown(self):
		print("I'm dead!")

if __name__=="__main__":
	
	
	cs = camera_sim()
		
