# run this code in the directory where the folders with the jpgs are located

import cv2 as cv
import os

codec = cv.VideoWriter_fourcc('m','p','4','v')
fps = 12.0
folders = os.listdir()  # or put in here the absolute path, preferably using os.path.join()

for folder in folders:
	files = sorted(os.listdir(folder))
	images = []
	
	for file in files:
		if file.endswith("jpg"):
			images.append(file)

	if not images:
		print("Folder " + folder + " does not contain jpg images")
		continue
 
	frame = cv.imread(os.path.join(folder, images[0]))
	height, width, channels = frame.shape
	video = cv.VideoWriter(os.path.join("0_videos", folder) + ".mp4", codec, fps, (width, height))
    
	for image in images:
		frame = cv.imread(os.path.join(folder, image))
		video.write(frame)

	video.release()
