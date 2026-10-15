import os
import urllib.request
import cv2
import numpy as np
import open3d as o3d
URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
for name in["aloeL.jpg","aloeR.jpg"]:
    if not os.path.exists(name):
        print("downloading",name)
        urllib.request.urlretrieve(URL + name, name)