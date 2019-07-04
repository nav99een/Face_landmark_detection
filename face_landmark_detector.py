#run python3 face_landmark_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat

#importing essential python libraries and modules
import argparse
import cv2
import imutils
import dlib
from imutils import face_utils
import numpy as np

#Reading arguments from command line or bash
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

#Defining model to extract face feature from image
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(args['shape_predictor'])

if __name__=='__main__':
	
	a = int(input("Form Image:1\nFrom webcam:2\nEnter the choice :"))
	if a==1:
		input_path = input("Enter path to the image :")
		img = cv2.imread(input_path)
		img = imutils.resize(img, width=600)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		rects= detect(gray,1)

		for (i,rect) in enumerate(rects):
			shape=predict(gray,rect)
			shape = face_utils.shape_to_np(shape)
			
			(x,y,w,h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 25, 255), 1)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(img, (x, y), 1, (197, 67, 255), -1)

		# show the output image with the face detections + facial landmarks
		cv2.imshow("Output", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		vs = 0

	elif a==2:
		vs = cv2.VideoCapture(0)

	if vs:
		while(True):
			img = vs.read()[1]
			img = imutils.resize(img, width=600)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			rects= detect(gray,1)

			for (i,rect) in enumerate(rects):
				shape=predict(gray,rect)
				shape = face_utils.shape_to_np(shape)
				
				(x,y,w,h) = face_utils.rect_to_bb(rect)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

				cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 25, 255), 1)

				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				for (x, y) in shape:
					cv2.circle(img, (x, y), 1, (197, 67, 255), -1)

			# show the output image with the face detections + facial landmarks
			cv2.imshow("Output", img)
			key = cv2.waitKey(1)
			if key ==27:
				break

		vs.release()
		cv2.destroyAllWindows()



#================*****============*****============****=============END==========*****============****==========***===========*****==============#
