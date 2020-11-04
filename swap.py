import cv2
import numpy as np
import dlib
from utils.image_utilities import ImageUtilities


class SWAP:

    visualize = False

    def __init__(self, img1, img2):
        self.img1, self.img2 = self.init_images(img1, img2)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.detector = detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
        self.imgStack = ImageUtilities()
        self.landmarks_points = []
        self.landmarks_points2 = []
        self.mask1 = np.zeros_like(self.img1_gray)
        self.mask2 = np.zeros_like(self.img2_gray)
        self.convex_hull_1 = None
        self.convex_hull_2 = None
        self.index_triangles = []
        self.destination_img = np.zeros_like(self.img2)
        self.result = None

    @classmethod
    def enable_visualization(cls, show=True):
        if show:
            cls.visualize = True
            return
        cls.visualize = False

    @staticmethod
    def init_images(img1, img2):
        # img1; path to img1
        #  img2; path to img2
        if img1 is None or img2 is None:
            raise FileNotFoundError

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        return img1, img2

    def show_images(self):

        stacked_image = self.imgStack.stack_images(0.6, ([self.img1, self.img2, self.destination_img, self.result]))
        cv2.imshow('Stacked Image', stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def first_face_landmarks(self):

        faces = self.detector(self.img1_gray)
        for face in faces:
            landmarks = self.predictor(self.img1_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

                if self.visualize:
                    self.img1 = cv2.circle(self.img1, (x, y), 3, (0, 0, 255), -1)

            self.landmarks_points.append(landmarks_points)

    def second_face_landmarks(self):
        faces = self.detector(self.img2_gray)
        for face in faces:
            landmarks = self.predictor(self.img2_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

                if self.visualize:
                    self.img2 = cv2.circle(self.img2, (x, y), 3, (0, 0, 255), -1)

            self.landmarks_points2.append(landmarks_points)

    def first_convex_hull(self):

        for points_set  in self.landmarks_points:
            self.convex_hull_1 = cv2.convexHull(np.array(points_set, np.int32))

            if self.visualize:
                cv2.polylines(self.img1, [self.convex_hull_1], True, (0, 255, 0))
                cv2.fillConvexPoly(self.mask1, self.convex_hull_1, 255)
                self.mask1 = cv2.bitwise_and(self.img1, self.img1, mask=self.mask1)

    def second_convex_hull(self):

        for points_set in self.landmarks_points2:
            self.convex_hull_2 = cv2.convexHull(np.array(points_set, np.int32))

            if self.visualize:
                cv2.polylines(self.img2, [self.convex_hull_2], True, (0, 255, 0))
                cv2.fillConvexPoly(self.mask2, self.convex_hull_2, 255)
                # self.mask2 = cv2.bitwise_and(self.img2, self.img2, mask=self.mask2)


    def delaunay_triangle_1(self):
        rect = cv2.boundingRect(self.convex_hull_1)
        subdiv = cv2.Subdiv2D(rect)

        subdiv.insert(self.landmarks_points[0])
        triangles = np.array(subdiv.getTriangleList(), np.int32)

        for triangle in triangles:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            self.__extarct_index_points([pt1, pt2, pt3])

            if self.visualize:
                cv2.line(self.img1, pt1, pt2, (0, 0, 255), 2)
                cv2.line(self.img1, pt2, pt3, (0, 0, 255), 2)
                cv2.line(self.img1, pt3, pt1, (0, 0, 255), 2)

    def delaunay_triangle_2(self):

        for index_triangle in self.index_triangles:

            pt1 = self.landmarks_points2[0][index_triangle[0]]
            pt2 = self.landmarks_points2[0][index_triangle[1]]
            pt3 = self.landmarks_points2[0][index_triangle[2]]

            if self.visualize:
                cv2.line(self.img2, pt1, pt2, (0, 0, 255), 2)
                cv2.line(self.img2, pt2, pt3, (0, 0, 255), 2)
                cv2.line(self.img2, pt3, pt1, (0, 0, 255), 2)

    def combine_triangle(self):

        for index_triangle in self.index_triangles:

            tr1_pt1 = self.landmarks_points[0][index_triangle[0]]
            tr1_pt2 = self.landmarks_points[0][index_triangle[1]]
            tr1_pt3 = self.landmarks_points[0][index_triangle[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = self.img1[y: y+h, x:x+w]
            cropped_tr1_mask = np.zeros((h,w), np.int8)

            points = np.array([[tr1_pt1[0] -x, tr1_pt1[1] - y],
                              [tr1_pt2[0] -x, tr1_pt2[1] - y],
                              [tr1_pt3[0] -x, tr1_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask , points, 255)
            cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)


            tr2_pt1 = self.landmarks_points2[0][index_triangle[0]]
            tr2_pt2 = self.landmarks_points2[0][index_triangle[1]]
            tr2_pt3 = self.landmarks_points2[0][index_triangle[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
            cropped_triangle2 = self.img2[y: y + h, x:x + w]
            cropped_tr2_mask = np.zeros((h,w), np.int8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                               [tr2_pt2[0] - x, tr2_pt2[1] - y],
                               [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)


            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

            # # Reconstructing destination face
            img2_new_face_rect_area = self.destination_img[y: y + h, x: x + w]
            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            self.destination_img[y: y + h, x: x + w] = img2_new_face_rect_area

            if self.visualize:

                cv2.line(self.img1, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
                cv2.line(self.img1, tr1_pt2, tr1_pt3, (0, 0, 255), 2)
                cv2.line(self.img1, tr1_pt3, tr1_pt1, (0, 0, 255), 2)

                cv2.line(self.img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
                cv2.line(self.img2, tr2_pt2, tr2_pt3, (0, 0, 255), 2)
                cv2.line(self.img2, tr2_pt3, tr2_pt1, (0, 0, 255), 2)

    def after_math(self):
        # Face swapped (putting 1st face into 2nd face)
        # img2_face_mask = np.zeros_like(self.img2_gray)
        # img2_head_mask = cv2.fillConvexPoly(img2_face_mask, self.convex_hull_2, 255)
        img2_face_mask = cv2.bitwise_not(self.mask2)
        # img1_face_mask = cv2.bitwise_not(self.mask1)

        # img2_head_noface = np.zeros_like(self.img1)
        img2_head_noface = cv2.bitwise_and(self.img2, self.img2, mask=img2_face_mask)
        # img2_head_noface = cv2.bitwise_and(img2_head_noface, img2_head_noface, mask=img2_face_mask)

        self.result = cv2.add(img2_head_noface, self.destination_img)

        (x, y, w, h) = cv2.boundingRect(self.convex_hull_2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        # seamlessclone = cv2.seamlessClone(self.result, self.img2, self.mask2, center_face2, cv2.MIXED_CLONE)
        # cv2.imshow('fff', seamlessclone)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def __extarct_index_points(self, pts):
        triangle = []

        for i in range(len(pts)):
            index_pt = np.where((np.array(self.landmarks_points[0], np.int32) == pts[i]).all(axis=1))
            for num in index_pt[0]:
                triangle.append(num)

        self.index_triangles.append(triangle)
