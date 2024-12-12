import cv2
import numpy as np
import os

from matplotlib import pyplot as plt


def segment_and_recognize(plate_images):
	"""
	In this file, you will define your own segment_and_recognize function.
	To do:
		1. Segment the plates character by character
		2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
		3. Recognize the character by comparing the distances
	Inputs:(One)
		1. plate_imgs: cropped plate images by Localization.plate_detection function
		type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
	Outputs:(One)
		1. recognized_plates: recognized plate characters
		type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
	Hints:
		You may need to define other functions.
	"""
	#1
	#k1 parameter is percentage of max size of a letter/digit from the image/k2 for the smallest
	#ratioStandard - percentage of the image that is a letter/digit
	ratioStandard = 0.3
	processEachDL(plate_images,0.1,0.2,0.01,ratioStandard)

	#visualization
	# plt.imshow(cv2.cvtColor(imageInverted,cv2.COLOR_GRAY2RGB))
	# plt.title('Green')
	# plt.axis('off')
	# plt.show()

	# recognized_plates = [None, None, None]
	# return recognized_plates



def processEachDL(image, epsilon,k1,k2,ratioStandard):
	#1

	#TODO: max digit size, min digit size have to be set, minimum ration
	#TODO: so 3 things

	#We convert RGB->Gray
	imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#separatin into a binary image
	image = isodata_thresholding(imgGray,epsilon)
	# inverting so letter/digits are white the plate is black
	imageInverted = 255-image

	plt.imshow(cv2.cvtColor(imageInverted,cv2.COLOR_GRAY2RGB))
	plt.title('Green')
	plt.axis('off')
	plt.show()

	# Calculate an area of the image so then we can use that to discard not valid contours
	height, width = image.shape[:2]
	MAX_AREA = height*width

	#Now if we have find countours for each digit
	contours,_ = cv2.findContours(imageInverted,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		area = cv2.contourArea(contour)
		#if this contour is too big we discard it
		if(MAX_AREA*k1<area or MAX_AREA*k2>area):
			continue


		x, y, w, h = cv2.boundingRect(contour)


		#ratio
		roi = imageInverted[y:y + h, x:x + w]

		nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
		allPixelsInTheRectangle = w * h
		ratio = nonZeroPixelsInTheRectangle/allPixelsInTheRectangle

		if(ratio<ratioStandard):
			continue



		imageCropped = imageInverted[y:y + h, x:x + w]

		# visualization
		plt.imshow(cv2.cvtColor(imageCropped, cv2.COLOR_GRAY2RGB))

		plt.title('Green')
		plt.axis('off')
		plt.show()




def isodata_thresholding(image, epsilon = 2):
	# Compute the histogram and set up variables
	hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
	tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
	old_tau = -2*epsilon
	# Iterations of the isodata thresholding algorithm
	while(abs(tau - old_tau) >= epsilon):
		ForegroundMask = image >= tau
		#TODO Calculate m1
		m1 = image[ForegroundMask]
		#this so there is no division by zero problem
		m1Len = np.count_nonzero(m1) if np.count_nonzero(m1)!=0 else 0.0001
		m1Sum = np.sum(m1)
		m1 = m1Sum/m1Len
		#TODO Calculate m2
		BackgroundMask = image < tau
		m2 = image[BackgroundMask]
		m2Len = np.count_nonzero(m2) if np.count_nonzero(m2) != 0 else 0.0001
		m2Sum = np.sum(m2)
		m2 = m2Sum / m2Len
		# TODO Calculate new tau
		old_tau = tau
		tau = (m1 + m2) / 2
	# TODO Threshold the image based on last tau
	# background = np.where(image < tau, image, 0)
	# foreground = np.where(image >= tau, image, 0)
	image = np.where(image>=tau,255,0).astype(np.uint8)
	return image


plate_image_path = "recognitionTestPlate.png"
plate_image = cv2.imread(plate_image_path)
if plate_image is None:
    print(f"Error: Could not load image from path {plate_image_path}")
else:
    segment_and_recognize(plate_image)