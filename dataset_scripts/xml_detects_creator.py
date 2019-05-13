import imutils
# import dlib
import cv2
import datetime
import glob
import sys

def main():
	# construct the argument parser and parse the arguments

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	#detector = dlib.get_frontal_face_detector()
	#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	#predictor = dlib.shape_predictor("predictor.dat")

	# images_folder = 'fotos_oldSpice/'
	# images_folder = 'train_llavero/'
	# images_folder = 'test_llavero/'
	images_folder = sys.argv[1]

	files = glob.glob(images_folder + "*")
	print('%d images for detection' % (len(files)))
	font = cv2.FONT_HERSHEY_SIMPLEX

	file = open(images_folder + "training.xml","w")

	file.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
	file.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
	file.write("<dataset>\n")
	file.write("<name>Training examples</name>\n")
	# file.write("<comment>CPS Images.\n")
	# file.write("   This images are from CPS Dataset\n")
	# file.write("</comment>\n")
	file.write("<images>\n")
	
	n = len(files)
	for i,f in enumerate(files):

		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(f)

		
		if image is not(None):
			image = imutils.resize(image, width=700)
			
			if i % 1 == 0:
				
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				# detect faces in the grayscale image
				
				fromCenter = False
				# Select multiple rectangles
				rects = cv2.selectROIs(str(i+1) + ' of ' + str(n), image, fromCenter)
				cv2.destroyAllWindows()
				#rects = cv2.selectROI("Output", image, False, fromCenter)

				if len(rects) > 0:
					filename = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+str(i))+".jpg" 
					cv2.imwrite(images_folder + str(filename),image)
					file.write("  <image file='"+str(filename)+"'>\n")
					
				

					# loop over the object detections
					for rect in rects:
						# determine the facial landmarks for the face region, then
						# convert the facial landmark (x, y)-coordinates to a NumPy
						# array
						x,y,w,h = rect
						
						#x,y,w,h = rect
						file.write("    <box top='"+str(y)+"' left='"+str(x)+"' width='"+str(w)+"' height='"+str(h)+"'>\n")
						# show the face number
							
						
						#cv2.waitKey(10000)
						file.write("    </box>\n")


					file.write("  </image>\n")

	file.write("</images>\n")
	file.write("</dataset>\n")
	

if __name__ == '__main__':
	main()