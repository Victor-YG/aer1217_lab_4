"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv


class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)


    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)

        return frame

    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera

        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt
                temp_feature[:, 4:6] = kp_l[pt_idx].pt
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) &
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) &
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor


    def pose_estimation(self, features_coor):
        # dummy C and r
        C = np.eye(3)
        r = np.array([0, 0, 0])
        # feature in right img (without filtering)
        f_r_prev, f_r_cur = features_coor[:, 2:4], features_coor[:, 6:8]

        # ------------- start your code here -------------- #

        # reproject to 3D - pa, pb ~ (3 x N)
        pa, pb = self.reprojection(features_coor)

        # outlier rejection with RANSAC
        pa_in, pb_in = self.ransac(pa, pb)

        # final point cloud alignment with all inliers (pa --> pb)
        C, r = self.point_cloud_alignment(pa_in, pb_in)

        # replace (1) the dummy C and r to the estimated C and r.
        #         (2) the original features to the filtered features
        return C, r, f_r_prev, f_r_cur


    def reprojection(self, features_coor):
        '''Project 2D keypoint correspondences into 3D point clouds with stereo camera info'''

        # upack features
        kp_l_prev = features_coor[:, 0:2]
        kp_r_prev = features_coor[:, 2:4]
        kp_l_curr = features_coor[:, 4:6]
        kp_r_curr = features_coor[:, 6:8]

        # 3D reprojection
        projection_l = np.zeros([3, 4])
        projection_l[0, 0] = self.cam.fx
        projection_l[0, 2] = self.cam.cu
        projection_l[1, 1] = self.cam.fy
        projection_l[1, 2] = self.cam.cv
        projection_l[2, 2] = 1.0

        projection_r = np.zeros([3, 4])
        projection_r[0, 0] = self.cam.fx
        projection_r[0, 2] = self.cam.cu
        projection_r[0, 3] =-self.cam.fx * self.cam.baseline
        projection_r[1, 1] = self.cam.fy
        projection_r[1, 2] = self.cam.cv
        projection_r[2, 2] = 1.0

        pa = cv.triangulatePoints(projection_l, projection_r, np.transpose(kp_l_prev), np.transpose(kp_r_prev))
        pb = cv.triangulatePoints(projection_l, projection_r, np.transpose(kp_l_curr), np.transpose(kp_r_curr))
        pa = pa / pa[3:]
        pb = pb / pb[3:]

        return pa[0:3, :], pb[0:3, :]


    def ransac(self, pa, pb):
        pass


    def point_cloud_alignment(self, pa, pb):
        pass


    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right

    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME

        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking

    def update(self, img_left, img_right, frame_id):

        self.new_frame_left = img_left
        self.new_frame_right = img_right

        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)

        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)

        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)

        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right

        return frame_left, frame_right


