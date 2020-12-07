import cv2
import numpy as np
import dlib
from utils.image_utilities import ImageUtilities


class SWAP_T:

    visualize = False

    def __init__(self, img1, img2):
        self.img1, self.img2 = self.init_images(img1, img2)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
        self.imgStack = ImageUtilities()
        self.landmarks_points = {}
        self.mask1 = np.zeros_like(self.img1_gray)
        self.mask2 = np.zeros_like(self.img2_gray)
        self.convex_hull = {}
        self.index_triangles = []
        self.destination_img = np.zeros(self.img2.shape, np.uint8)
        self.result = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

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

        row_1 = [self.img1, self.img2, self.result]
        # row_2 = [self.mask1, self.mask2]

        stacked_image = self.imgStack.stack_images(0.75, (row_1))
        cv2.imshow('Stacked Image', stacked_image)
        cv2.imwrite('images/result.jpg', stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __face_landmarks(self, gray):
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

        return landmarks_points

    def first_face_landmarks(self):

        landmarks_points = self.__face_landmarks(self.img1_gray)
        self.landmarks_points['first'] = landmarks_points

        if self.visualize:

            for i, landmarks_point in enumerate(landmarks_points):
                self.img1 = cv2.circle(self.img1, (landmarks_point[0], landmarks_point[1]), 3, (0, 255, 0), -1)
                self.img1 = cv2.putText(self.img1, str(i), (landmarks_point[0], landmarks_point[1]), self.font, 1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                if 5 == i:
                    break

    def second_face_landmarks(self):

        landmarks_points = self.__face_landmarks(self.img2_gray)
        self.landmarks_points['second'] = landmarks_points

        if self.visualize:
            for i, landmarks_point in enumerate(landmarks_points):
                self.img2 = cv2.circle(self.img2, (landmarks_point[0], landmarks_point[1]), 3, (0, 0, 255), -1)
                self.img2 = cv2.putText(self.img2, str(i), (landmarks_point[0], landmarks_point[1]), self.font, 1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                if 5 == i:
                    break

    def first_convex_hull(self):

        self.convex_hull['first'] = cv2.convexHull(np.array(self.landmarks_points['first'], np.int32))
        cv2.fillConvexPoly(self.mask1, self.convex_hull['first'], 255)
        self.mask1 = cv2.bitwise_and(self.img1, self.img1, mask=self.mask1)

        if self.visualize:
            cv2.polylines(self.img1, [self.convex_hull['first']], True, (0, 255, 0))

    def second_convex_hull(self):

        self.convex_hull['second'] = cv2.convexHull(np.array(self.landmarks_points['second'], np.int32))
        cv2.fillConvexPoly(self.mask2, self.convex_hull['second'], 255)
        self.mask2 = cv2.bitwise_and(self.img2, self.img2, mask=self.mask2)

        if self.visualize:
            cv2.polylines(self.img2, [self.convex_hull['second']], True, (0, 0, 255))

    def delaunay_triangle_1(self):
        rect = cv2.boundingRect(self.convex_hull['first'])
        subdiv = cv2.Subdiv2D(rect)

        subdiv.insert(self.landmarks_points['first'])
        triangles = np.array(subdiv.getTriangleList(), np.int32)

        print(self.landmarks_points['first'][0])

        for triangle in triangles:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            self.__extract_index_points([pt1, pt2, pt3], self.landmarks_points['first'])

            if self.visualize:
                cv2.line(self.img1, pt1, pt2, (0, 255, 0), 1)
                cv2.line(self.img1, pt2, pt3, (0, 255, 0), 1)
                cv2.line(self.img1, pt3, pt1, (0, 255, 0), 1)

    def delaunay_triangle_2(self):

        rect = cv2.boundingRect(self.convex_hull['second'])
        subdiv = cv2.Subdiv2D(rect)

        subdiv.insert(self.landmarks_points['second'])
        triangles = np.array(subdiv.getTriangleList(), np.int32)

        for triangle in triangles:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            self.__extract_index_points([pt1, pt2, pt3], self.landmarks_points['second'])

            if self.visualize:
                cv2.line(self.img2, pt1, pt2, (0, 0, 255), 1)
                cv2.line(self.img2, pt2, pt3, (0, 0, 255), 1)
                cv2.line(self.img2, pt3, pt1, (0, 0, 255), 1)

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

    def __extract_index_points(self, pts, landmarks_points):
        triangle = []

        for i in range(len(pts)):
            index_pt = np.where((np.array(landmarks_points, np.int32) == pts[i]).all(axis=1))
            for num in index_pt[0]:
                triangle.append(num)

        self.index_triangles.append(triangle)