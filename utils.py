import cv2 as cv
import argparse
import glob
import numpy as np

def calibrate_cam():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, .001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpeg')
    iter = 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,8), None, flags=cv.CALIB_CB_FAST_CHECK)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (8,8), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        iter += 1
        print(iter)
    cv.destroyAllWindows()
    return

def video_to_mp4(input, output, fps: int = 0, frame_size: tuple = (), fourcc: str = "H264"):
    vidcap = cv.VideoCapture(input)
    if not fps:
        fps = round(vidcap.get(cv.CAP_PROP_FPS))
    success, arr = vidcap.read()
    if not frame_size:
        height, width, _ = arr.shape
        frame_size = width, height
    writer = cv.VideoWriter(
        output,
        apiPreference=0,
        fourcc=cv.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=frame_size
    )
    while True:
        if not success:
            break
        writer.write(arr)
        success, arr = vidcap.read()
    writer.release()
    vidcap.release()

def open_vid(file_path:str, algo:str):
    if algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(file_path))
    if not capture.isOpened():
        print('Unable to open: ' + file_path)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        fgMask = backSub.apply(frame)
        
        
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        # cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break




# open_vid('test.mp4', 'MOG2')
calibrate_cam()