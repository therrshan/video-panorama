
import cv2
import numpy as np
import cv2 as cv
import imutils

class StitcherLeft:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.sift = cv2.SIFT_create()

    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        windowsize = 50

        if version == 'left':
            offset = int(windowsize / 2)
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img1))
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1


        else:
            offset = int(windowsize / 2)
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img2))

            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)
        (kp2, des2) = self.detectAndDescribe(img2)
        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

        if R is None:
            return img2
        (good, M, mask) = R

        result = cv.warpAffine(img1, M, (img2.shape[1], img1.shape[0]))

        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")
        img2 = img2 * mask_pano
        result = result * mask_newImg

        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]

        return result



    def detectAndDescribe(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])

        return (kps, des)

    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)

            return (good, M, mask)

        return None

    def crop_result(self, result_img):
        h = result_img.shape[0]
        w = result_img.shape[1]
        cut_off_percent = 0.1
        w_new = int(w * cut_off_percent)
        result_img = result_img[:, w_new:]
        return result_img


class StitcherRight:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.sift = cv2.SIFT_create()

    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        windowsize = 50

        if version == 'left':
            offset = int(windowsize / 2)
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img1))
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1

        else:
            offset = int(windowsize / 2)
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img2))
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)
        (kp2, des2) = self.detectAndDescribe(img2)

        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

        if R is None:
            return None
        (good, M, mask) = R
        result = cv.warpAffine(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")

        img2 = img2 * mask_pano
        result = result * mask_newImg

        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]
        return result

    def detectAndDescribe(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, des)

    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])
            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)
            return (good, M, mask)

        return None

    def crop_result(self, result_img):
        h = result_img.shape[0]
        w = result_img.shape[1]
        cut_off_percent = 0.9
        w_new = int(w * cut_off_percent)
        result_img = result_img[:, :w_new]
        return result_img







