# -*- coding: utf-8 -*-
# run this code in the directory where target folder is located

import cv2
import numpy as np
import random
import os
import shutil

source_folder_name = os.path.join(os.sep, 'home', 'em165153', 'Downloads', '20bn-jester-v1', '0_videos')
target_folder_name = "randomly_selected_subset"
sample_counter = 1
number_of_target_samples_per_class = 100

for class_id in range(1, 6):
    classes = np.genfromtxt("class" + str(class_id) + "-video-ids.csv", delimiter=',', dtype=(int, int))
    encoded_videos = []

    while True:
        video_id, class_id2 = classes[random.randrange(len(classes))]
        
        if class_id != class_id2:
            print("class IDs do not match")
            break
        
        if video_id in encoded_videos:
            continue

        capture = cv2.VideoCapture(os.path.join(source_folder_name, str(video_id)) + ".mp4")
        w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        capture.release()
        
        if w != 176 or h != 100:
            continue
        
        shutil.copyfile(os.path.join(source_folder_name, str(video_id)) + ".mp4", os.path.join(target_folder_name, "sam") + str(sample_counter) + "_" + str(video_id) + "_" + str(class_id) + ".mp4")

        encoded_videos.append(video_id)
        sample_counter += 1
        
        if sample_counter % number_of_target_samples_per_class == 1:
            print("Finished class " + str(class_id))
            break

