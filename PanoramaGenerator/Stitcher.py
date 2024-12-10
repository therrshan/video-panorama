
import cv2
import numpy as np
import cv2 as cv
import imutils

class StitcherLeft:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.sift = cv2.SIFT_create()

    # create a mask for blending so that the edge of two blended images will be smooth
    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # after fine-tuning 50 works well
        windowsize = 50

        # for the image on the left, which is new image
        if version == 'left':
            offset = int(windowsize / 2)
            # here the feathering only happens on the inner left side of panorama, in case the newly appended image
            # does not have new information on the left
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img1))
            # generate a linear mask for the left so that the opacity will gradually reduce to 0 from left to right
            # the opacity gets 0 when it is the right-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1

        # for the image on the right, which is panorama
        else:
            offset = int(windowsize / 2)
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img2))
            # generate a linear mask for the right so that the opacity will gradually increase to 0 from left to right
            # the opacity gets 0 when it is the left-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    # stitch two images
    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)  # new left image for stitching
        (kp2, des2) = self.detectAndDescribe(img2)  # existing pano
        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh) # get a mask of the key points

        if R is None:
            # no matching, directly return the original pano img
            return img2
        (good, M, mask) = R

        result = cv.warpAffine(img1, M, (img2.shape[1], img1.shape[0]))

        # generate a mask for both panorama and the new image for a smooth blending
        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")

        # apply the mask
        img2 = img2 * mask_pano
        result = result * mask_newImg

        # blend two images, by adding the new image to the panorama
        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]

        return result



    def detectAndDescribe(self, img):
        # get the key points and description of the image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # use sift for feature description
        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])

        return (kps, des)


    # match the points for projection or transformation
    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        # use brute force KNN matching method to match two features
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        # stores points that can be used for warpping processing
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            # as mentioned above, the warpPerspective + findHomography does not work well, so changed to affine warpping
            # (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)

            return (good, M, mask)

        # if not enough points for matching, return None
        return None

    # crop left 10% and right 10% of the input new frame to avoid too much distortion near the edge
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

    # create a mask for blending so that the edge of two blended images will be smooth
    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # after fine-tuning 50 works well
        windowsize = 50

        # for the image on the left, which is panorama
        if version == 'left':
            offset = int(windowsize / 2)
            # here the feathering only happens on the inner right side of panorama, in case the newly appended image
            # does not have new information on the right
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img1))
            # generate a linear mask for the left so that the opacity will gradually reduce to 0 from left to right
            # the opacity gets 0 when it is the right-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1

        # for the image on the right, which is new image
        else:
            offset = int(windowsize / 2)
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img2))
            # generate a linear mask for the right so that the opacity will gradually increase to 0 from left to right
            # the opacity gets 0 when it is the left-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    # stitch two images
    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)  # new right image for stitching
        (kp2, des2) = self.detectAndDescribe(img2)  # pano

        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh) # get a mask of the key points

        if R is None:
            # no matching, directly return the original pano img
            return None
        (good, M, mask) = R

        # I tried to use warpPerspective + findHomography, but the image will stretch to very long when it keeps appending
        # affine transformation will keep the parallelism and ratios of distances, it seems to be a better tool
        # result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # warp the new image to fit it to a large canvas on which it is projected to a place
        # where its cooresponding place existing in the panorama will be overlapping
        # here is different to stitching on left, because stitching on right only needs a larger canvas, does not need
        # to move the existing panorama to right as stitching on left
        result = cv.warpAffine(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # generate a mask for both panorama and the new image for a smooth blending
        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")

        # apply the mask
        img2 = img2 * mask_pano
        result = result * mask_newImg

        # blend two images, by adding the new image to the panorama
        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]
        return result

    def detectAndDescribe(self, img):
        # get the key points and description of the image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # use sift for feature description
        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, des)

    # match the points for projection or transformation
    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        # use brute force KNN matching method to match two features
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        # stores points that can be used for warpping processing
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))


        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            # as mentioned above, the warpPerspective + findHomography does not work well, so changed to affine warpping
            # (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)

            return (good, M, mask)
        # if not enough points for matching, return None
        return None

    # crop left 10% and right 10% of the input new frame to avoid too much distortion near the edge
    def crop_result(self, result_img):
        h = result_img.shape[0]
        w = result_img.shape[1]
        cut_off_percent = 0.9
        w_new = int(w * cut_off_percent)
        result_img = result_img[:, :w_new]
        return result_img







