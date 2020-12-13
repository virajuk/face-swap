import cv2
import numpy as np
import dlib
from utils.image_utilities import ImageUtilities
from collections import namedtuple
import os
import time


class SWAP:

    visualize = False
    img_path = '/home/viraj-uk/HUSTLE/FACE_SWAPPING/images/'

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
        self.result = self.img2.copy()
        self.intermediate = self.img1.copy()
        self.img2_copy = self.img2.copy()

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

        # row_1 = [self.img1, self.img2, self.mask1, self.intermediate]
        row_1 = [self.img1, self.img2, self.show1, self.show2]
        # row_2 = [self.mask1, self.mask2]

        stacked_image = self.imgStack.stack_images(0.5, (row_1))
        cv2.imshow('Stacked Image', stacked_image)
        cv2.imwrite(os.path.join(self.img_path, 'result.jpg'), stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def show_animation(self, iter):
    #
    #     row_1 = [self.img1, self.img2]
    #     # row_1 = [self.img1, self.img2]
    #     # row_2 = [self.mask1, self.mask2]
    #
    #     stacked_image = self.imgStack.stack_images(0.5, (row_1))
    #     # cv2.imshow('Stacked Image', stacked_image)
    #     cv2.imwrite(os.path.join(self.img_path, str(iter) + ".jpg"), stacked_image)

        # cv2.waitKey(0)

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

    # def draw_face_landmarks(self, gray):
    #     faces = self.detector(gray)
    #     landmarks_points = []
    #     for face in faces:
    #         landmarks = self.predictor(gray, face)
    #         for n in range(0, 68):
    #             x = landmarks.part(n).x
    #             y = landmarks.part(n).y
    #             landmarks_points.append((x, y))
    #
    #
    #             self.img1 = cv2.circle(self.img1, (x, y), 5, (0, 0, 255), -1)
    #             row_1 = [self.img1]
    #             stacked_image = self.imgStack.stack_images(0.5, (row_1))
    #             cv2.imwrite(os.path.join(self.img_path, str(n) + ".jpg"), stacked_image)


        # return landmarks_points

    # def test(self):
    #
    #     self.draw_face_landmarks(self.img1_gray)

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

    # def combine_triangle(self):
    #
    #     for index_triangle in self.index_triangles:
    #
    #         tr1_pt1 = self.landmarks_points['first'][index_triangle[0]]
    #         tr1_pt2 = self.landmarks_points['first'][index_triangle[1]]
    #         tr1_pt3 = self.landmarks_points['first'][index_triangle[2]]
    #         triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    #
    #         rect1 = cv2.boundingRect(triangle1)
    #         (x, y, w, h) = rect1
    #         cropped_triangle = self.img1[y: y+h, x:x+w]
    #         cropped_tr1_mask = np.zeros((h, w), np.int8)
    #
    #         points = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
    #                           [tr1_pt2[0]-x, tr1_pt2[1]-y],
    #                           [tr1_pt3[0]-x, tr1_pt3[1]-y]], np.int32)
    #         cv2.fillConvexPoly(cropped_tr1_mask , points, 255)
    #         cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
    #
    #         tr2_pt1 = self.landmarks_points['second'][index_triangle[0]]
    #         tr2_pt2 = self.landmarks_points['second'][index_triangle[1]]
    #         tr2_pt3 = self.landmarks_points['second'][index_triangle[2]]
    #         triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    #
    #         rect2 = cv2.boundingRect(triangle2)
    #         (x, y, w, h) = rect2
    #
    #         cropped_tr2_mask = np.zeros((h, w), np.int8)
    #
    #         points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
    #                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
    #                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    #
    #         cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    #
    #
    #
    #         # Warp triangles
    #         points = np.float32(points)
    #         points2 = np.float32(points2)
    #         M = cv2.getAffineTransform(points, points2)
    #         warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    #         warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    #
    #         # # Reconstructing destination face
    #         img2_new_face_rect_area = self.destination_img[y: y + h, x: x + w]
    #         img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    #         _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    #         warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    #
    #         img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    #         self.destination_img[y: y + h, x: x + w] = img2_new_face_rect_area

    # def final_show(self):
    #     img2_face_mask = np.zeros_like(self.img2_gray)
    #     img2_head_mask = cv2.fillConvexPoly(img2_face_mask, self.convex_hull['second'], 255)
    #     img2_face_mask = cv2.bitwise_not(img2_head_mask)
    #
    #     img2_head_noface = cv2.bitwise_and(self.img2, self.img2, mask=img2_face_mask)
    #     result = cv2.add(img2_head_noface, self.destination_img)
    #
    #     (x, y, w, h) = cv2.boundingRect(self.convex_hull['second'])
    #     center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    #
    #     seamlessclone = cv2.seamlessClone(result, self.img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    #
    #     scale_percent = 40  # percent of original size
    #     width = int(seamlessclone.shape[1] * scale_percent / 100)
    #     height = int(seamlessclone.shape[0] * scale_percent / 100)
    #     dim = (width, height)
    #     self.result = cv2.resize(seamlessclone, dim, interpolation=cv2.INTER_AREA)

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

    def my(self):
        # landmark points for faces
        LandmarkPoints = namedtuple('LandmarkPoints', ['first', 'second'])
        self.landmarks_points = LandmarkPoints(self.__face_landmarks(self.img1_gray), self.__face_landmarks(self.img2_gray))

        mask1, mask2 = None, None

        # accumulate_warped = np.zeros_like((self.img1.shape[0], self.img1.shape[1]), np.uint8)
        accumulate_warped = np.zeros_like(self.img2)
        accumulate_mask = np.zeros((self.img2.shape[0], self.img2.shape[1]), np.uint8)

        self.first_order_triangles()
        color = (0, 255, 0)
        for key in self.ordered_triangles:

            pt1, pt2, pt3 = self.ordered_triangles[key]
            points = np.array([[self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], self.landmarks_points.first[pt3]]], np.int32)

            # cv2.line(self.img1, self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], color, 1)
            # cv2.line(self.img1, self.landmarks_points.first[pt2], self.landmarks_points.first[pt3], color, 1)
            # cv2.line(self.img1, self.landmarks_points.first[pt3], self.landmarks_points.first[pt1], color, 1)

            points2 = np.array([[self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], self.landmarks_points.second[pt3]]], np.int32)

            # cv2.line(self.result, self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], color, 1)
            # cv2.line(self.result, self.landmarks_points.second[pt2], self.landmarks_points.second[pt3], color, 1)
            # cv2.line(self.result, self.landmarks_points.second[pt3], self.landmarks_points.second[pt1], color, 1)

            M = cv2.getAffineTransform(np.float32(points), np.float32(points2))

            # mask1 = np.zeros((self.img1.shape[0], self.img1.shape[1]), np.uint8)
            # mask1 = cv2.fillPoly(mask1, points, (255))

            # mask2 = cv2.warpAffine(mask1, M, (self.img1.shape[1], self.img1.shape[0]))
            warped = cv2.warpAffine(self.img1, M, (self.img1.shape[1], self.img1.shape[0]))

            mask2 = np.zeros((warped.shape[0], warped.shape[1]), np.uint8)
            mask2 = cv2.fillPoly(mask2, points2, (255))

            # print(accumulate_mask.shape)

            # accumulate_mask = cv2.bitwise_or(accumulate_mask, accumulate_mask, mask=mask2)

            # self.img1 = cv2.bitwise_and(self.img1, self.img1, mask=mask1)
            warped_mask = cv2.bitwise_and(warped, warped, mask=mask2)

            three_ch_mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

            # self.img2[mask2] = (255)

            self.img2 = cv2.subtract(self.img2, three_ch_mask2)
            self.img2 = cv2.add(self.img2, warped_mask)
            # self.img2 = cv2.mean(self.img2, mask=three_ch_mask2)

            # self.img2 = cv2.GaussianBlur(self.img2, (3, 3), 1)

            # print(self.img2.shape)
            # break

            # accumulate_warped = cv2.add(accumulate_warped, warped_mask)
            accumulate_mask = cv2.add(accumulate_mask, mask2)



            # print(self.img2[900, 200])
            # self.img2 = cv2.circle(self.img2,(600, 200),50, (255), 4)
            # self.img2 = cv2.circle(self.img2, (600, 200), 5, (0, 0, 255), -1)

            # img_yuv = cv2.cvtColor(self.img2, cv2.COLOR_BGR2HSV)
            # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            # img_yuv[:, :, 1] = cv2.equalizeHist(img_yuv[:, :, 1])
            # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            #
            # self.img2 = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

            # self.img2[:, :, 0] = cv2.equalizeHist(self.img2[:, :, 0])
            # self.img2[:, :, 1] = cv2.equalizeHist(self.img2[:, :, 1])
            # self.img2[:, :, 2] = cv2.equalizeHist(self.img2[:, :, 2])
            # self.img2[:, :, 3] = cv2.equalizeHist(self.img2[:, :, 3])

            # self.img2 = np.stack((self.img2, warped_mask), axis=0)
            # self.img2 = cv2.addWeighted(warped_mask, 0.9999, self.img2, 0.99, 0)
            # self.img2 = cv2.bitwise_or(self.img2, warped_mask)

            # (x, y, w, h) = cv2.boundingRect(accumulate_mask)
            # # print(x, y, w, h)
            # center = (int(x + w / 2), int(y + h / 2))
            #
            # mixed_clone = cv2.seamlessClone(self.img2, self.img2_copy, accumulate_mask, center, cv2.NORMAL_CLONE)
            # row_1 = [self.img1, self.img2, mixed_clone]


            # cv2.imshow('Stacked Image', stacked_image)

            if 6 == key:
                pass
                # break

            # stacked_image = self.imgStack.stack_images(0.5, (row_1))
            # cv2.imwrite(os.path.join(self.img_path, str(key)+'.jpg'), stacked_image)



        # mask = 255 * np.ones(self.img2.shape, self.img2.dtype)

        # width, height, channels = self.img2.shape
        # center = (int(height / 2), int(width / 2))
        # center = (394, 640)

        (x, y, w, h) = cv2.boundingRect(accumulate_mask)
        # print(x, y, w, h)
        center = (int(x + w/2), int(y+h/2))

        mixed_clone = cv2.seamlessClone(self.img2, self.img2_copy, accumulate_mask, center, cv2.NORMAL_CLONE)

        # self.img2 = cv2.addWeighted(accumulate_warped, 0.9999, self.img2, 0.0001, 0)
        # print(warped_mask.shape)
        # warped_mask_on_dst_image = cv2.bitwise_and(self.img2, self.img2, mask=warped_mask)

        # _, mask_triangles_designed = cv2.threshold(self.img2, 1, 255, cv2.THRESH_BINARY_INV)
        # self.img2 = cv2.cvtColor(mask_triangles_designed, cv2.COLOR_GRAY2BGR)

        row_1 = [self.img1, self.img2_copy, mixed_clone]

        stacked_image = self.imgStack.stack_images(0.5, (row_1))
        cv2.imwrite(os.path.join(self.img_path, 'result.jpg'), stacked_image)
        cv2.imshow('Stacked Image', stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def swap(self):

        # landmark points for faces
        LandmarkPoints = namedtuple('LandmarkPoints', ['first', 'second'])
        self.landmarks_points = LandmarkPoints(self.__face_landmarks(self.img1_gray), self.__face_landmarks(self.img2_gray))

        self.first_order_triangles()

        color = (0, 255, 0)
        for key in self.ordered_triangles:

            pt1, pt2, pt3 = self.ordered_triangles[key]

            # triangle1 = np.array([self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], self.landmarks_points.first[pt3]], np.int32)
            # rect1 = cv2.boundingRect(triangle1)
            # (x, y, w, h) = rect1

            # cropped_triangle = self.img1[y: y + h, x:x + w]
            points = np.array([[self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], self.landmarks_points.first[pt3]]], np.int32)

            cv2.line(self.img1, self.landmarks_points.first[pt1], self.landmarks_points.first[pt2], color, 1)
            cv2.line(self.img1, self.landmarks_points.first[pt2], self.landmarks_points.first[pt3], color, 1)
            cv2.line(self.img1, self.landmarks_points.first[pt3], self.landmarks_points.first[pt1], color, 1)

            # cv2.rectangle(self.img1, (x,y), (x+w, y+h), (255, 255, 0), 2)

            #################################################################
            triangle2 = np.array([self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], self.landmarks_points.second[pt3]], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            points2 = np.array([[self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], self.landmarks_points.second[pt3]]], np.int32)

            cv2.line(self.img2, self.landmarks_points.second[pt1], self.landmarks_points.second[pt2], color, 1)
            cv2.line(self.img2, self.landmarks_points.second[pt2], self.landmarks_points.second[pt3], color, 1)
            cv2.line(self.img2, self.landmarks_points.second[pt3], self.landmarks_points.second[pt1], color, 1)

            self.mask1 = cv2.fillPoly(self.mask1, points, (255))

            M = cv2.getAffineTransform(np.float32(points), np.float32(points2))
            # print(M)

            # self.mask1 = cv2.cvtColor(self.mask1, cv2.COLOR_BGR2GRAY)
            # print(self.mask1.shape)

            self.show1 = cv2.bitwise_and(self.img1, self.img1, mask=self.mask1)
            self.show2 = cv2.warpAffine(self.mask1, M, (self.mask2.shape[1], self.mask2.shape[0]))


            print(self.img2.shape)
            print(self.show2.shape)
            print(self.mask2.shape)

            # cv2.imshow('temp', self.show2)
            # self.show2 = cv2.add(self.img2, self.show2)
            # self.show2 = cv2.bitwise_and(self.img2, self.img2, mask=self.mask2)

            print(self.show2.shape)

            # print(w, h)
            # print(warped_triangle.shape)
            # print(points2)

            # cropped_triangle = self.img2[y - 1: y - 1 + h, x - 1:x - 1 + w]
            # print(warped_triangle.shape)
            # print(type(warped_triangle))
            # cv2.imshow("Color Image", color_image)

            # print(cropped_triangle.shape)

            # print(points)

            # arr = [[[252, 508], [224, 446], [281, 494]]]
            # arr = [[[252, 508], [224, 446], [281, 494], [252, 508], [189, 484]]]
            # points = np.array(arr, np.int32)
            # points = [[[252, 508], [224, 446], [281, 494], [224, 446], [252, 508], [189, 484]]]

            # print(points)
            # cv2.fillPoly(self.mask1, points,(255))

            # print(self.mask1.shape)

            # print(mask.shape)
            # cv2.bitwise_and(self.img1, self.img1, mask=self.mask1)
            # self.intermediate = warped_triangle

            # cv2.imwrite(os.path.join(self.img_path, str(key)+'result.jpg'), warped_triangle)
            # cv2.imwrite(os.path.join(self.img_path, str(key)+'cropped.jpg'), cropped_triangle)

            # mask = cropped_triangle
            # add_to_mask = cropped_triangle.copy()

            # self.intermediate = warped_triangle

            # self.mask1 = mask + cv2.cvtColor(add_to_mask, cv2.COLOR_BGR2GRAY)
            # add_to_mask =
            # self.mask1 = cv2.bitwise_or(self.mask1, cv2.cvtColor(add_to_mask, cv2.COLOR_BGR2GRAY))

            # cv2.imshow('add_to_mask', self.mask1)
            # cv2.waitKey(0)

            # self.intermediate = cropped_triangle
            # self.result = warped_triangle

            # print(self.mask1.shape)

            # cv2.rectangle(self.img2, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # row_1 = [self.img1]
            # row_1 = [self.img1, self.img2]
            # row_2 = [self.mask1, self.mask2]

            # stacked_image = self.imgStack.stack_images(0.5, (row_1))
            # cv2.imshow('Stacked Image', self.img1)
            # cv2.imwrite(os.path.join(self.img_path, 'result.jpg'), stacked_image)

            # time.sleep(0.5)

            # self.show_animation(key)

            if 0 == key:
                # cv2.destroyAllWindows()
                break

