import cv2
import numpy as np
import dlib
from utils.image_utilities import ImageUtilities
from collections import namedtuple
import os


class SWAP:

    visualize = False
    img_path = '/home/viraj-uk/HUSTLE/FACE_SWAPPING/images'     # replace image path accordingly

    def __init__(self, img1, img2):
        self.img1, self.img2 = self.init_images(img1, img2)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
        self.imgStack = ImageUtilities()
        self.landmarks_points = ()
        self.ordered_triangles = {}
        self.img2_copy = self.img2.copy()

    @staticmethod
    def init_images(img1, img2):

        if img1 is None or img2 is None:
            raise FileNotFoundError

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        return img1, img2

    def show_images(self):

        row_1 = [self.img1, self.img2_copy, self.img2]

        stacked_image = self.imgStack.stack_images(0.8, row_1)
        cv2.imshow('Stacked Image', stacked_image)
        cv2.imwrite(os.path.join(self.img_path, 'result.jpg'), stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def face_landmarks(self, gray):
        faces = self.detector(gray)
        landmarks_points = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

        return landmarks_points

    def get_triangles(self, which):

        rect = cv2.boundingRect(np.array(getattr(self.landmarks_points, which), np.int32))
        subdivision = cv2.Subdiv2D(rect)
        subdivision.insert(getattr(self.landmarks_points, which))

        return np.array(subdivision.getTriangleList(), np.int32)

    def first_order_triangles(self):

        # get delaunay triangles for face landmarks
        first_triangles = self.get_triangles('first')

        # compare delaunay triangles of both images
        minimum_triangles = min(len(self.get_triangles('first')), len(self.get_triangles('second')))

        landmark_array = np.array(self.landmarks_points.first, np.int32)

        for i in range(0, minimum_triangles):

            pt1 = (first_triangles[i][0], first_triangles[i][1])
            pt2 = (first_triangles[i][2], first_triangles[i][3])
            pt3 = (first_triangles[i][4], first_triangles[i][5])

            triangle_pt1 = np.where((landmark_array == pt1).all(axis=1))
            triangle_pt2 = np.where((landmark_array == pt2).all(axis=1))
            triangle_pt3 = np.where((landmark_array == pt3).all(axis=1))

            self.ordered_triangles[i] = [triangle_pt1[0][0], triangle_pt2[0][0], triangle_pt3[0][0]]

    def swap(self):

        # landmark points for faces
        LandmarkPoints = namedtuple('LandmarkPoints', ['first', 'second'])
        self.landmarks_points = LandmarkPoints(self.face_landmarks(self.img1_gray), self.face_landmarks(self.img2_gray))

        # order triangles on image 1, based on detected landmark points
        self.first_order_triangles()

        accumulated_mask = np.zeros((self.img2.shape[0], self.img2.shape[1]), np.uint8)

        for key in self.ordered_triangles:

            pt1, pt2, pt3 = self.ordered_triangles[key]
            points = np.array([[self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], self.landmarks_points.first[pt3]]], np.int32)
            points2 = np.array([[self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], self.landmarks_points.second[pt3]]], np.int32)

            M = cv2.getAffineTransform(np.float32(points), np.float32(points2))

            step_warped_image = cv2.warpAffine(self.img1, M, (self.img1.shape[1], self.img1.shape[0]))

            step_warped_mask_triangle = np.zeros((step_warped_image.shape[0], step_warped_image.shape[1]), np.uint8)
            step_warped_mask_triangle = cv2.fillPoly(step_warped_mask_triangle, points2, (255))

            self.img2 = cv2.subtract(self.img2, cv2.cvtColor(step_warped_mask_triangle, cv2.COLOR_GRAY2BGR))

            step_warped_triangle = cv2.bitwise_and(step_warped_image, step_warped_image, mask=step_warped_mask_triangle)
            self.img2 = cv2.add(self.img2, step_warped_triangle)

            accumulated_mask = cv2.add(accumulated_mask, step_warped_mask_triangle)

        (x, y, w, h) = cv2.boundingRect(accumulated_mask)
        center = (int(x + w / 2), int(y + h / 2))

        self.img2 = cv2.seamlessClone(self.img2, self.img2_copy, accumulated_mask, center, cv2.NORMAL_CLONE)
