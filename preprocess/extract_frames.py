import cv2
import glob
from tqdm import tqdm
import os

VIDEO_PATH = './data/YouTubeClips/*.avi'

def extract_frames(video_name):
    if not os.path.exists("./Images"):
        os.mkdir("./Images")
    vidcap = cv2.VideoCapture(video_name)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite("./Images/" + video_name.split('\\')[-1][:-4] + "image{count}.jpg".format(count=count), image)
        return hasFrames
    sec = 0
    frameRate = 1 
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

if __name__ == "__main__":

    files = glob.glob(VIDEO_PATH)

    for file in tqdm(files):
        extract_frames(file)
