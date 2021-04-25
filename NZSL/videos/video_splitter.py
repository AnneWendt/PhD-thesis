import cv2
import os
import time

FILENAME = 'up2.mp4'
CLASS = 'up'
SRC_DIR = 'videos'
TAR_DIR = 'samples'
SAM_COUNT_START = 247
CODEC = cv2.VideoWriter_fourcc('m','p','4','v')
FPS = 20

def run(filename):
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    capture = cv2.VideoCapture(filename)

    fps = capture.get(cv2.CAP_PROP_FPS)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("Frames per second:", fps)
    print("Frame width:", width)
    print("Frame height:", height)

    cnt = SAM_COUNT_START
    video_name = os.path.join(TAR_DIR, "sam") + str(cnt) + '_' + CLASS + ".mp4"
    video = cv2.VideoWriter(video_name, CODEC, FPS, (width, height))

    success, frame = capture.read()

    while success:
        cv2.imshow("Video", frame)
        video.write(frame)

        if cv2.waitKey(50) == 32:
            cv2.rectangle(frame, (1, 1), (width-1, height-1), (0, 0, 255), 4)
            cv2.imshow("Video", frame)
            cv2.waitKey(5)  # otherwise, rectangle is not shown
            
            video.release()
            time.sleep(1)

            cnt += 1
            video_name = os.path.join(TAR_DIR, "sam") + str(cnt) + '_' + CLASS + ".mp4"
            video = cv2.VideoWriter(video_name, CODEC, FPS, (width, height))

        success, frame = capture.read()

    video.release()
    capture.release()

run(os.path.join(SRC_DIR, FILENAME))

