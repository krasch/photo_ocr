"""
Methods to postprocess CRAFT model outputs.
The code corresponds to code from the original repository (./craft_utils.py),
but has been extensively refactored to improve clarity.
This refactoring is ongoing. todo
"""

import math

import numpy as np
import cv2


""" auxilary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


# todo refactor or remove
def _find_top_and_bottom_cordinates(word_label):
    for i in range(word_label.shape[1]):

        region = np.where(word_label[:, i] != 0)[0]

        if len(region) < 2:
            continue

        yield i, region[0], region[-1]

#def _break_up_into_segments(word_pixels, num_segments, segment_width):
#
#    total_width = word_pixels.shape[1]
#    for x in range(pixel_width):


# todo directly work on pixel level instead of top_bottoms
# todo explain wtf is going on
def _break_up_into_segments(top_bottoms, num_total_segments, segment_width):
    segment_number = 0
    segment = []

    for x, top, bottom in top_bottoms:
        if (segment_number + 1) * segment_width <= x and segment_number <= num_total_segments:
            yield np.array(segment)

            # reset the segment
            segment_number += 1
            segment = []

        segment.append(x)

    # handle final segment
    if len(segment) > 0:
        yield np.array(segment)


def _crazyness(top_bottoms, num_center_points, box_width):

    num_total_segments = num_center_points * 2 + 1
    segment_width = box_width / num_total_segments

    segments = _break_up_into_segments(top_bottoms, num_total_segments, segment_width)

    control_points = []
    segment_heights = []
    pivot_points = []

    for segment_number, xs in enumerate(segments):
        tops = np.array([top for x, top, bottom in top_bottoms if x in xs])
        bottoms = np.array([bottom for x, top, bottom in top_bottoms if x in xs])

        center_ys = (tops + bottoms) * 0.5
        heights = bottoms - tops + 1

        control_points.append([xs.mean(), center_ys.mean()])

        if segment_number % 2 != 0:
            max_height_index = np.argmax(heights)

            segment_heights.append(heights[max_height_index])
            pivot_points.append((xs[max_height_index], center_ys[max_height_index]))

    return control_points, pivot_points, segment_heights, segment_width


# calc gradiant and apply to make horizontal pivots
def _calculate_new_pivot_points(pivot_points, control_points, half_median_character_height):

    for i, (pivot_x, pivot_y) in enumerate(pivot_points):
        # print(i * 2 + 2)

        # calculating the gradient between two center points (i*2 + 2) and (i*2), where i is the index of a pivot point
        # ... why?
        # haha we are never going to use most of these control points!
        center_diff_x = control_points[i * 2 + 2][0] - control_points[i * 2][0]
        center_diff_y = control_points[i * 2 + 2][1] - control_points[i * 2][1]

        # both center points are at the same location, we can't get a gradient
        # but why would this happen?
        if center_diff_x == 0:
            # anyway, in this case we can't calculate an angle and just todo
            yield [pivot_x, pivot_y - half_median_character_height, pivot_x, pivot_y + half_median_character_height]

        else:
            # todo
            angle = -math.atan2(center_diff_y, center_diff_x)
            c, s = half_median_character_height * math.cos(angle), half_median_character_height * math.sin(angle)
            yield [pivot_x - s, pivot_y - c, pivot_x + s, pivot_y + c]


def adjust_edge_pivot_point(edge_pivot_point, edge_gradient, half_median_character_height, word_pixels, direction):
    """
    Move edge pivot point so that it is right at the edge of the word
    I have no idea what is going on here...
    :param edge_pivot_point:
    :param edge_gradient:
    :param half_median_character_height:
    :param word_pixels:
    :param direction:
    :return:
    """
    max_r = 2.0
    step_r = 0.2

    for r in np.arange(0.5, max_r, step_r):
        dx = 2 * half_median_character_height * r
        line_img = np.zeros(word_pixels.shape, dtype=np.uint8)
        dy = edge_gradient * dx
        p = np.array(edge_pivot_point) + direction * np.array([dx, dy, dx, dy])
        cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
        if np.sum(np.logical_and(word_pixels, line_img)) == 0 or r + 2 * step_r >= max_r:
            return p

    return None


def _unwarp_coordinates(poly, transformation_matrix):
    try:
        inverse_transformation_matrix = np.linalg.inv(transformation_matrix)  # would this actually happen?
    except:
        return None  # sic!


def calculate_polygon(word_bounding_box, word_segmentation):
    # a bunch of parameters whose meaning I only found out by reading the original code
    # why the parameters are set to exactly those values? ¯\_(ツ)_/¯
    num_center_points = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    min_bounding_box_size = 10

    # the bounding box can be angled - calculate its width and height
    box_width = int(np.linalg.norm(word_bounding_box[0] - word_bounding_box[1]) + 1)
    box_height = int(np.linalg.norm(word_bounding_box[1] - word_bounding_box[2]) + 1)

    # don't calculate polygon for small bounding boxes (why? no explanation in original code)
    if box_width < min_bounding_box_size or box_height < min_bounding_box_size:
        return None

    # calculate a transformation matrix that can be used to extract only the word we are interested in right now
    destination_coordinates = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    transformation_matrix = cv2.getPerspectiveTransform(word_bounding_box, destination_coordinates)

    # apply the transformation matrix -> now have a segmentation map that only contains the word + its background
    word_segmentation = cv2.warpPerspective(word_segmentation.astype(int), transformation_matrix,
                                            (box_width, box_height), flags=cv2.INTER_NEAREST)

    # todo move downstairs?
    try:
        Minv = np.linalg.inv(transformation_matrix)
    except:
        return None

    # for each "column" in the word segmentation map, find the top and bottom y value of the word
    top_bottoms = list(_find_top_and_bottom_cordinates(word_segmentation))

    # pass if max_len is similar to h, todo comment why???
    max_len = max([bottom - top + 1 for col, top, bottom in top_bottoms])
    if box_height * max_len_ratio < max_len:
        return None

    # get pivot points with fixed length
    control_points, pivot_points, segment_heights, segment_width = _crazyness(top_bottoms, num_center_points, box_width)

    # pass if num of pivots is not sufficient or segment width is smaller than character height; again: why?
    if None in pivot_points or segment_width < np.max(segment_heights) * 0.25:
        return None

    # calc median maximum of pivot points
    half_median_character_height = np.median(segment_heights) * expand_ratio / 2

    # calc gradiant and apply to make horizontal pivots
    new_pp = list(_calculate_new_pivot_points(pivot_points, control_points, half_median_character_height))

    # get edge points to cover character heatmaps
    grad_s = (pivot_points[1][1] - pivot_points[0][1]) / (pivot_points[1][0] - pivot_points[0][0]) + (pivot_points[2][1] - pivot_points[1][1]) / (pivot_points[2][0] - pivot_points[1][0])
    grad_e = (pivot_points[-2][1] - pivot_points[-1][1]) / (pivot_points[-2][0] - pivot_points[-1][0]) + (pivot_points[-3][1] - pivot_points[-2][1]) / (pivot_points[-3][0] - pivot_points[-2][0])
    spp = adjust_edge_pivot_point(new_pp[0], grad_s, half_median_character_height, word_segmentation, direction=-1)
    epp = adjust_edge_pivot_point(new_pp[-1], grad_e, half_median_character_height, word_segmentation, direction=+1)

    # pass if boundary of polygon is not found (why would that happen??)
    if spp is None or epp is None:
        return None

    #poly = [(spp[0], spp[1])] + list(new_pp) + epp + reversed(list(new_pp)) + [spp[2], spp[3]]
    #poly = [warpCoord(Minv, p) for p in poly]

    # make final polygon
    poly = []
    poly.append(warpCoord(Minv, (spp[0], spp[1])))
    for p in new_pp:
        poly.append(warpCoord(Minv, (p[0], p[1])))
    poly.append(warpCoord(Minv, (epp[0], epp[1])))
    poly.append(warpCoord(Minv, (epp[2], epp[3])))
    for p in reversed(new_pp):
        poly.append(warpCoord(Minv, (p[2], p[3])))
    poly.append(warpCoord(Minv, (spp[2], spp[3])))

    # print(poly)

    # add to final result
    return np.array(poly)
