import cv2
import numpy as np
import dlib
from utils.image_utilities import ImageUtilities
from collections import namedtuple
import os


class SWAP:

    visualize = False
    img_path = '/home/viraj-uk/HUSTLE/FACE_SWAPPING/images'

    def __init__(self, img1, img2):
        self.img1, self.img2 = self.init_images(img1, img2)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
        self.imgStack = ImageUtilities()
        self.landmarks_points = ()
        self.mask1 = np.zeros_like(self.img1_gray)
        self.mask2 = np.zeros_like(self.img2_gray)
        self.convex_hull = ()
        self.index_triangles = []
        self.ordered_triangles = {}
        # self.destination_img = np.zeros(self.img2.shape, np.uint8)
        # self.result = None

    @classmethod
    def enable_visualization(cls, show=True):
        if show:
            cls.visualize = True
            return
        cls.visualize = False

    @staticmethod
    def init_images(img1, img2):

        if img1 is None or img2 is None:
            raise FileNotFoundError

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        return img1, img2

    def show_images(self):

        # row_1 = [self.img1, self.img2, self.result]
        row_1 = [self.img1, self.img2]
        # row_2 = [self.mask1, self.mask2]

        stacked_image = self.imgStack.stack_images(0.8, (row_1))
        cv2.imshow('Stacked Image', stacked_image)
        cv2.imwrite(os.path.join(self.img_path, 'result.jpg'), stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __face_landmarks(self, gray):
        faces = self.detector(gray)
        landmarks_points = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

        return landmarks_points

    # def new_face_landmarks(self, gray):
    #     faces = self.detector(gray)
    #     for face in faces:
    #         landmarks = self.predictor(gray, face)
    #         my_list = [n for n in range(0, 68)]
    #         print(my_list)
    #         FaceLandmarkPoints = namedtuple('FaceLandmarkPoints', [1, 2])
    #         # for n in range(0, 68):
    #         #     pass
    #
    #     return landmarks

    # def delaunay_test(self):
    #     rect = cv2.boundingRect(self.convex_hull['first'])
    #
    #     # draw bounding rectangle around convex hull
    #     # x, y, w, h = rect
    #     # cv2.rectangle(self.img1, (x, y), (x+w, y+h), (0, 0, 255))
    #
    #     #
    #     subdiv = cv2.Subdiv2D()
    #     subdiv.initDelaunay(rect)
    #
    #     subdiv.insert(self.landmarks_points['first'])
    #     triangles = np.array(subdiv.getTriangleList(), np.int32)
    #
    #     triangle = triangles[0]
    #
    #     pt1 = (triangle[0], triangle[1])
    #     pt2 = (triangle[2], triangle[3])
    #     pt3 = (triangle[4], triangle[5])
    #
    #     cv2.line(self.img1, pt1, pt2, (0, 255, 0), 1)
    #     cv2.line(self.img1, pt2, pt3, (0, 255, 0), 1)
    #     cv2.line(self.img1, pt3, pt1, (0, 255, 0), 1)
    #     ############################################
    #
    #     rect2 = cv2.boundingRect(self.convex_hull['second'])
    #
    #     # draw bounding rectangle around convex hull
    #     # x, y, w, h = rect2
    #     # cv2.rectangle(self.img2, (x, y), (x + w, y + h), (0, 0, 255))
    #
    #     subdiv2 = cv2.Subdiv2D()
    #     subdiv2.initDelaunay(rect2)
    #
    #     subdiv2.insert(self.landmarks_points['second'])
    #     triangles2 = np.array(subdiv2.getTriangleList(), np.int32)
    #
    #     triangle2 = triangles2[0]
    #
    #     pt1 = (triangle2[0], triangle2[1])
    #     pt2 = (triangle2[2], triangle2[3])
    #     pt3 = (triangle2[4], triangle2[5])
    #
    #     cv2.line(self.img2, pt1, pt2, (0, 255, 0), 1)
    #     cv2.line(self.img2, pt2, pt3, (0, 255, 0), 1)
    #     cv2.line(self.img2, pt3, pt1, (0, 255, 0), 1)
    #
    #     self.show_images_test()

    # def delaunay_triangle_1(self):
    #     rect = cv2.boundingRect(self.convex_hull['first'])
    #     subdiv = cv2.Subdiv2D(rect)
    #
    #     subdiv.insert(self.landmarks_points['first'])
    #     triangles = np.array(subdiv.getTriangleList(), np.int32)
    #
    #     for triangle in triangles:
    #         pt1 = (triangle[0], triangle[1])
    #         pt2 = (triangle[2], triangle[3])
    #         pt3 = (triangle[4], triangle[5])
    #
    #         self.__extract_index_points([pt1, pt2, pt3], self.landmarks_points['first'])
    #
    #         if self.visualize:
    #             cv2.line(self.img1, pt1, pt2, (0, 255, 0), 1)
    #             cv2.line(self.img1, pt2, pt3, (0, 255, 0), 1)
    #             cv2.line(self.img1, pt3, pt1, (0, 255, 0), 1)

    # def delaunay_triangle_2(self):
    #     rect = cv2.boundingRect(self.convex_hull['second'])
    #     subdiv = cv2.Subdiv2D(rect)
    #
    #     subdiv.insert(self.landmarks_points['second'])
    #     triangles = np.array(subdiv.getTriangleList(), np.int32)
    #
    #     for triangle in triangles:
    #         pt1 = (triangle[0], triangle[1])
    #         pt2 = (triangle[2], triangle[3])
    #         pt3 = (triangle[4], triangle[5])
    #
    #         self.__extract_index_points([pt1, pt2, pt3], self.landmarks_points['second'])
    #
    #         if self.visualize:
    #             cv2.line(self.img2, pt1, pt2, (0, 0, 255), 1)
    #             cv2.line(self.img2, pt2, pt3, (0, 0, 255), 1)
    #             cv2.line(self.img2, pt3, pt1, (0, 0, 255), 1)

    def combine_triangle(self):

        for index_triangle in self.index_triangles:

            tr1_pt1 = self.landmarks_points['first'][index_triangle[0]]
            tr1_pt2 = self.landmarks_points['first'][index_triangle[1]]
            tr1_pt3 = self.landmarks_points['first'][index_triangle[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = self.img1[y: y+h, x:x+w]
            cropped_tr1_mask = np.zeros((h, w), np.int8)

            points = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                              [tr1_pt2[0]-x, tr1_pt2[1]-y],
                              [tr1_pt3[0]-x, tr1_pt3[1]-y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask , points, 255)
            cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

            tr2_pt1 = self.landmarks_points['second'][index_triangle[0]]
            tr2_pt2 = self.landmarks_points['second'][index_triangle[1]]
            tr2_pt3 = self.landmarks_points['second'][index_triangle[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.int8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                               [tr2_pt2[0] - x, tr2_pt2[1] - y],
                               [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)



            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # # Reconstructing destination face
            img2_new_face_rect_area = self.destination_img[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            self.destination_img[y: y + h, x: x + w] = img2_new_face_rect_area

    def final_show(self):
        img2_face_mask = np.zeros_like(self.img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, self.convex_hull['second'], 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(self.img2, self.img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, self.destination_img)

        (x, y, w, h) = cv2.boundingRect(self.convex_hull['second'])
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, self.img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        scale_percent = 40  # percent of original size
        width = int(seamlessclone.shape[1] * scale_percent / 100)
        height = int(seamlessclone.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.result = cv2.resize(seamlessclone, dim, interpolation=cv2.INTER_AREA)

    # def __extract_index_points(self, pts, landmarks_points):
    #     triangle = []
    #
    #     for i in range(len(pts)):
    #         index_pt = np.where((np.array(landmarks_points, np.int32) == pts[i]).all(axis=1))
    #         for num in index_pt[0]:
    #             triangle.append(num)
    #
    #     self.index_triangles.append(triangle)

    def get_triangles(self, which):
        rect = cv2.boundingRect(np.array(getattr(self.landmarks_points, which), np.int32))
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(getattr(self.landmarks_points, which))

        triangles = np.array(subdiv.getTriangleList(), np.int32)
        return triangles

    def first_order_triangles(self):

        # get delaunay triangles for face landmarks
        first_triangles = self.get_triangles('first')

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
        self.landmarks_points = LandmarkPoints(self.__face_landmarks(self.img1_gray), self.__face_landmarks(self.img2_gray))

        self.first_order_triangles()

        color = (0, 255, 0)
        pt1, pt2, pt3 = self.ordered_triangles[0]

        cv2.line(self.img1, self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], color, 1)
        cv2.line(self.img1, self.landmarks_points.first[pt2], self.landmarks_points.first[pt3], color, 1)
        cv2.line(self.img1, self.landmarks_points.first[pt3], self.landmarks_points.first[pt1], color, 1)

        cv2.line(self.img2, self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], color, 1)
        cv2.line(self.img2, self.landmarks_points.second[pt2], self.landmarks_points.second[pt3], color, 1)
        cv2.line(self.img2, self.landmarks_points.second[pt3], self.landmarks_points.second[pt1], color, 1)
