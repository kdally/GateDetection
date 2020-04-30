import time

from Networks import Model
from Evaluation import get_IoU
from DataProperties import DataProperties
import os
import numpy as np
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt
import csv


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_tuple(self):
        return self.x, self.y

    def as_list(self):
        return [self.x, self.y]

    def get_distance(self, other_point):
        return np.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)


class Quadrilateral:
    def __init__(self, list_pt):

        center_list = np.sum(list_pt, 0)[0] / 4
        self.center = Point(center_list[0], center_list[1])

        self.bot_left = Point(0, 0)
        self.top_left = Point(0, 0)
        self.bot_right = Point(0, 0)
        self.top_right = Point(0, 0)
        self.max_dist_to_c = 0.0

        for k in range(4):
            if list_pt[k, 0, 0] - self.center.x < 0:
                if list_pt[k, 0, 1] - self.center.y < 0:
                    self.bot_left = Point(list_pt[k, 0, 0], list_pt[k, 0, 1])
                    self.max_dist_to_c = max(self.bot_left.get_distance(self.center), self.max_dist_to_c)
                else:
                    self.top_left = Point(list_pt[k, 0, 0], list_pt[k, 0, 1])
                    self.max_dist_to_c = max(self.top_left.get_distance(self.center), self.max_dist_to_c)
            else:
                if list_pt[k, 0, 1] - self.center.y < 0:
                    self.bot_right = Point(list_pt[k, 0, 0], list_pt[k, 0, 1])
                    self.max_dist_to_c = max(self.bot_right.get_distance(self.center), self.max_dist_to_c)
                else:
                    self.top_right = Point(list_pt[k, 0, 0], list_pt[k, 0, 1])
                    self.max_dist_to_c = max(self.top_right.get_distance(self.center), self.max_dist_to_c)

    def contains_pt(self, point):

        return point.get_distance(self.center) < self.max_dist_to_c

    def contains_quad(self, other_quad):

        return (self.contains_pt(other_quad.bot_left) and self.contains_pt(other_quad.top_left) and
                self.contains_pt(other_quad.bot_right) and self.contains_pt(other_quad.top_right))

    def as_list(self):
        return np.array([self.top_left.as_list(), self.top_right.as_list(), self.bot_right.as_list(),
                self.bot_left.as_list()])


def is_square(pt_list):
    if len(pt_list) == 4 and cv2.contourArea(pt_list) > 300.:
        return True
    return False


def detect():
    image_list = os.listdir(f'outputs/predicted')
    image_list.sort()
    idx = range(1, len(image_list))
    file = open(f'outputs/preTrained.csv', 'w')

    for i in idx:

        st_img = time.time()
        original_mask = cv2.imread(f'outputs/predicted/{image_list[i]}')
        gray = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
        output = np.zeros((DataProperties.height, DataProperties.width, 1), np.uint8)

        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        _, binary = cv2.threshold(gray, 0.25 * 255, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        quad_list = []

        for j in range(len(contours)):
            epsilon = 0.1 * cv2.arcLength(contours[j], True)
            guess = cv2.approxPolyDP(contours[j], epsilon, True)
            if is_square(guess):
                cv2.polylines(original_mask, [guess], True, (0, 255, 255), 2)
                quad = Quadrilateral(guess)
                quad_list.append(quad)
                areas.append(cv2.contourArea(guess))

        # quad_list = [quad_list for _, quad_list in sorted(zip(areas, quad_list), reverse=True)]

        # writer = csv.writer(file)
        # for m in frames:
        #     writer.writerow([image_list[i],
        #                      quad.top_left.x,
        #                      quad.top_left.y,
        #                      quad.top_right.x,
        #                      quad.top_right.y,
        #                      quad.bot_right.x,
        #                      quad.bot_right.y,
        #                      quad.bot_left.x,
        #                      quad.bot_left.y])

        # if len(quad_list) > 1:
        #     if quad_list[0].contains_quad(quad_list[1]):
        #         cv2.fillConvexPoly(output, quad_list[0].as_list(), (255, 255, 255))
        #         cv2.fillConvexPoly (output, quad_list[1].as_list(), (0, 0, 0))
        #
        # if len(quad_list) > 3:
        #     if quad_list[2].contains_quad(quad_list[3]):
        #         cv2.fillConvexPoly(output, quad_list[2].as_list(), (255, 255, 255))
        #         cv2.fillConvexPoly (output, quad_list[3].as_list(), (0, 0, 0))

        cv2.imwrite(f'outputs/detected/{image_list[i]}', original_mask)
        print(time.time() - st_img)
    return


detect()
