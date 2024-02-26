import time

import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly,updateTrajectoryDrawing1
from coke_features import coke_create,coke_detect
from keynet_features import keynet_detect,keynet_create
import os
if __name__ == "__main__":
    tracker = FeatureTracker()
    # detector = cv2.GFTTDetector_create()

    detector = cv2.ORB_create(nfeatures=1000)

    # detector1 = cv2.xfeatures2d.SIFT_create(1000)
    model0 = keynet_create()

    model1 = coke_create()  # coke

    # detector1 = cv2.xfeatures2d.SIFT_create(1000)
    datapath = "/data3/kitti_VO/kitti/sequences/00"
    imagesPath = os.path.join(datapath, "image_2")
    dataset_reader = DatasetReaderKITTI(datapath)

    K = dataset_reader.readCameraMatrix()

    prev_points = np.empty(0)
    prev_points1 = np.empty(0)
    prev_points3 = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    track_positions1=[]## 1
    track_positions3 = []  ## 3


    camera_rot, camera_pos = np.eye(3), np.zeros((3,1))
    camera_rot1, camera_pos1 = np.eye(3), np.zeros((3, 1)) #1
    camera_rot3, camera_pos3 = np.eye(3), np.zeros((3, 1))  # 1

    plt.show()

    # Process next frames
    for frame_no in range(1, 1101):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        st = time.time()
        prev_points = detector.detect(prev_frame)
        inter = time.time() - st
        print(f"orb time: {inter}")

        st = time.time()
        # prev_points1 = detector1.detect(prev_frame) ## 1
        prev_points1 = keynet_detect(model0,frame_no,None,imagesPath)
        inter = time.time() - st
        print(f"keynet time: {inter}")

        st = time.time()
        if frame_no == 1:
            prev_points3,cooperate_features = coke_detect(model1,frame_no,None,imagesPath)  #2
        else:
            prev_points3, cooperate_features = coke_detect(model1, frame_no, cooperate_features,imagesPath)  # 2
        inter = time.time() - st
        print(f"coke time: {inter}")


        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key = lambda p: p.response, reverse=True))
        # prev_points1 = cv2.KeyPoint_convert(sorted(prev_points1, key=lambda p: p.response, reverse=True)) ## 1
        prev_points1 = prev_points1[:, :2]
        prev_points1 = prev_points1.astype(np.float32)
        # prev_points3 = cv2.KeyPoint_convert(sorted(prev_points3, key=lambda p: p.response, reverse=True))  ## 3
        prev_points3 = prev_points3[:,:2]
        prev_points3 = prev_points3.astype(np.float32)
    
        # Feature tracking (optical flow)
        prev_points, curr_points = tracker.trackFeatures(prev_frame, curr_frame, prev_points, removeOutliers=True)
        prev_points1,curr_points1 = tracker.trackFeatures(prev_frame, curr_frame, prev_points1, removeOutliers=True) #1
        prev_points3, curr_points3 = tracker.trackFeatures(prev_frame, curr_frame, prev_points3,
                                                           removeOutliers=True)  # 3
        print (f"{len(curr_points)} ORB features left after feature tracking.")
        print(f"{len(curr_points1)} Key.Net features left after feature tracking.")
        print(f"{len(curr_points3)} CoKE features left after feature tracking.")

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        E1,mask1 =  cv2.findEssentialMat(curr_points1, prev_points1, K, cv2.RANSAC, 0.99, 1.0, None) #1
        E3,mask3 =  cv2.findEssentialMat(curr_points3, prev_points3, K, cv2.RANSAC, 0.99, 1.0, None) #3

        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])

        prev_points1 = np.array([pt for (idx, pt) in enumerate(prev_points1) if mask1[idx] == 1]) #1
        curr_points1 = np.array([pt for (idx, pt) in enumerate(curr_points1) if mask1[idx] == 1]) #1

        prev_points3 = np.array([pt for (idx, pt) in enumerate(prev_points3) if mask3[idx] == 1])  # 3
        curr_points3 = np.array([pt for (idx, pt) in enumerate(curr_points3) if mask3[idx] == 1])  # 3

        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        _, R1, T1, _ = cv2.recoverPose(E1, curr_points1, prev_points1, K) #1
        _, R3, T3, _ = cv2.recoverPose(E3, curr_points3, prev_points3, K)  # 3
        # print(f"{len(curr_points)} features left after pose estimation.")
        print (f"{len(curr_points)} ORB features left after pose estimation.")
        print(f"{len(curr_points1)} Key.Net features left after pose estimation.")
        print(f"{len(curr_points3)} CoKE features left after pose estimation.")

        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos = camera_pos + kitti_scale * camera_rot.dot(T)
        camera_pos1 = camera_pos1 + kitti_scale * camera_rot1.dot(T1) #1
        camera_pos3 = camera_pos3 + kitti_scale * camera_rot1.dot(T3)  # 3

        camera_rot = R.dot(camera_rot)
        camera_rot1 = R1.dot(camera_rot1) #1
        camera_rot3 = R3.dot(camera_rot3)  # 3

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)
        track_positions1.append(camera_pos1) #1
        track_positions3.append(camera_pos3)  # 3

        # updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        updateTrajectoryDrawing1(np.array(track_positions), np.array(kitti_positions),np.array(track_positions1),np.array(track_positions3))
        drawFrameFeatures(curr_frame, prev_points3, curr_points3, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR
        prev_points1, prev_frame_BGR = curr_points1, curr_frame_BGR
        prev_points3, prev_frame_BGR = curr_points3, curr_frame_BGR
    # cv2.destroyAllWindows()

