"""
Methods for calculating a tight polygon around the characters of the detected words.
In the paper, this step is described as follows:

"Additionally, we can generate a polygon around the entire character region to deal with curved texts effectively.
 The procedure of polygon generation is illustrated in Fig. 7.The first step is to find the local maxima line of
 character regions along the scanning direction, as shown in the figure with arrows in blue. The lengths of the local
 maxima lines are equally set as the maximum length among them to prevent the final polygon result from becoming uneven.
 The line connecting all the center points of the local maxima is called the center line, shown in yellow. Then, the
 local maxima lines are rotated to be perpendicular to the center line to reflect the tilt angle of characters, as
 expressed by the red arrows. The endpoints of the local maxima lines are the candidates for the control points of the
 text polygon. To fully cover the text region, we move the two outer-most tilted lo-cal  maxima  lines  outward  along
 the  local  maxima  center line, making the final control points (green dots)."
 (from https://arxiv.org/pdf/1904.01941.pdf, Section 3.3)
"""

import math
from typing import List, Tuple
from collections import namedtuple

import numpy as np
import cv2

MIN_BOUNDING_BOX_SIZE = 10
NUM_PIVOT_POINTS = 5
EXPAND_RATIO = 1.45


class PolygonCalculationError(Exception):
    pass


Point = namedtuple("Point", ["x", "y"])
BinStatistics = namedtuple("BinStatistics", ["center_point", "pivot_point", "height"])


def _calculate_box_dimensions(box: np.array):
    box_width = int(np.linalg.norm(box[0] - box[1]) + 1)
    box_height = int(np.linalg.norm(box[1] - box[2]) + 1)

    return box_width, box_height


def _cutout_and_warp(segmentation: np.array, box: np.array):
    """
    Given the segmentation map over the full input image, keep only keep only those parts covered by the bounding box
    (i.e. that represent some word) and warp this segmentation map cutout such that it is horizontally aligned.
    :param segmentation:
    :param box:
    :return:
    """
    box_width, box_height = _calculate_box_dimensions(box)

    # calculate a transformation matrix that can be used to extract only the word we are interested in right now
    destination_coordinates = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    transformation_matrix = cv2.getPerspectiveTransform(box, destination_coordinates)

    # perform the transformation
    # result is a segmentation map containing only the word selected by the bounding box, horizontally aligned
    warped = cv2.warpPerspective(segmentation, transformation_matrix, (box_width, box_height), flags=cv2.INTER_NEAREST)

    # will need to map coordinates from warped segmentation map back to the original one
    # already prepare this here
    inverted_transformation_matrix = np.linalg.inv(transformation_matrix)

    def unwarp(point):
        x, y = point
        out = np.matmul(inverted_transformation_matrix, (x, y, 1))
        return np.array([out[0] / out[2], out[1] / out[2]])

    return warped, unwarp


def _calculate_bin_statistics(segmentation_map: np.array, num_bins: int) -> List[BinStatistics]:
    """
    Divide the segmentation map into a series of horizontal segments (from now on called bins to avoid confusion).
    All bins have the same width and cover the full height of the segmentation map.

    In the following, column always means the vertical line of pixels at some given x. A word pixel is a pixel
    that has a 1 in the segmentation map. A background pixel is a pixel that has a 0 in the segmentation map. Column
    height refers to the number of positive pixels in the column.

    For each bin calculate the
    - height as the maximum column height of the bin
    - center_point as the center of the positive pixels
    - pivot_point with x=column with maximum height and y=center of that column

    :param segmentation_map:
    :param num_bins:
    :return:
    """
    map_height, map_width = segmentation_map.shape

    # xs = x-index (= column index) for each column
    # tops = y-index of the top-most white pixel for each column
    # bottoms = y-index of the bottom-most white pixel for each location
    xs = np.array(range(map_width))
    tops = segmentation_map.argmax(axis=0)
    bottoms = map_height - segmentation_map[::-1].argmax(axis=0) - 1

    # we only consider columns that have at least two character pixels
    invalid = segmentation_map.sum(axis=0) < 2

    # should be self-explanatory...
    heights = bottoms - tops + 1
    centers = (tops + bottoms) * 0.5

    # get the boundaries of the bins
    # first bin should include the left boundary, whereas all following bins should not
    # -> setting left-most boundary to -1 (bit of a hack, but works well)
    boundaries = np.linspace(0, map_width, num_bins + 1)
    boundaries[0] = -1

    for left_border, right_border in zip(boundaries, boundaries[1:]):
        in_bin = (left_border < xs) & (xs <= right_border)
        in_bin = in_bin & ~invalid

        # there is no column in this bin with at least two character pixels
        if len(xs[in_bin]) == 0:
            raise PolygonCalculationError()

        # this is the x at which the tallest column of the bin is located
        # the argmax just gives us an x relative to the bin, therefore need to add starting point of the bin
        max_height_x = xs[in_bin].min() + heights[in_bin].argmax()

        yield BinStatistics(center_point=Point(x=xs[in_bin].mean(), y=centers[in_bin].mean()),
                            # pivot point is the center of tallest column of the bin
                            pivot_point=Point(x=max_height_x, y=centers[max_height_x]),
                            height=heights[max_height_x])


def _calculate_inner_vertices(bins: List[BinStatistics]) -> List[Tuple[Point, Point]]:
    """
    Calculate pairs of upper/lower polygon vertices. Returns a list of these pairs, with length=NUM_PIVOT_POINTS

    Input are the bins as produced by _calculate_bin_statistics. In total there are (2 * NUM_PIVOT_POINTS + 1).
    For each of the pivot points (located at position 1, 3, 5, etc) grab the bin left to the pivot point and right
    to the pivot point and rotate the pivot bin according to the gradient between the center points of the left
    and right bin. Calculate for each pivot bin the upper and lower y-value at the pivot location. These form the pair
    of upper/lower vertices for this bin.

    This function matches this part of the original description: "Then, the local maxima lines are rotated to be
    perpendicular to the center line to reflect the tilt angle of characters, as expressed by the red arrows."

    :param bins:
    :return:
    """

    # use the same height for each bin (rather than the actual bin height),
    # original paper: "to prevent the final polygon result from becoming uneven"
    pivot_bin_heights = [bin_.height for i, bin_ in enumerate(bins) if i % 2 != 0]
    reference_height = np.median(pivot_bin_heights) * EXPAND_RATIO
    reference_height_halved = reference_height / 2.0

    def perform_pivot(left, pivot, right):
        diff_x = right.x - left.x
        diff_y = right.y - left.y

        if diff_x == 0:  # not really sure when this situation can arise, keeping it since was part of the original code
            delta_x = 0
            delta_y = reference_height_halved

        else:
            rad = - math.atan2(diff_y, diff_x)
            delta_x = reference_height_halved * math.sin(rad)
            delta_y = reference_height_halved * math.cos(rad)

        upper = Point(pivot.x - delta_x, pivot.y - delta_y)
        lower = Point(pivot.x + delta_x, pivot.y + delta_y)

        return upper, lower

    for i in range(1, len(bins), 2):
        yield perform_pivot(left=bins[i-1].center_point,
                            pivot=bins[i].pivot_point,
                            right=bins[i+1].center_point)


def _calculate_corners(segmentation_map: np.array,
                       bins: List[BinStatistics],
                       inner_vertices: List[Tuple[Point, Point]]) -> Tuple[Tuple[Point, Point], Tuple[Point, Point]]:
    """
    From paper: "To fully cover the text region, we move the two outer-most tilted lo-cal  maxima  lines  outward  along
                 the local maxima  center line, making the final control points (green dots)."

    Those green dots are the corner points that we are calculating here,
    returned as ((left_upper, left_lower), (right_upper, right_lower))
    :param segmentation_map:
    :param bins:
    :param inner_vertices:
    :return:
    """

    # all pairs of vertices have this distance
    reference_height = np.linalg.norm(np.array(inner_vertices[0][0]) - np.array(inner_vertices[0][1]))

    def calculate_gradient(left, right):
        left_x, left_y = left
        right_x, right_y = right

        return (right_y - left_y) / (right_x - left_x)

    def push_to_edge(current_corner, grad, direction):
        max_r = 2.0
        step_r = 0.2

        for r in np.arange(0.5, max_r, step_r):
            dx = direction * reference_height * r
            dy = grad * dx

            upper = current_corner[0].x + dx, current_corner[0].y + dy
            lower = current_corner[1].x + dx, current_corner[1].y + dy
            if r + 2 * step_r >= max_r:
                return upper, lower

            line_img = np.zeros(segmentation_map.shape, dtype=np.uint8)
            cv2.line(line_img, (int(upper[0]), int(upper[1])), (int(lower[0]), int(lower[1])), 1, thickness=1)
            if np.sum(np.logical_and(segmentation_map, line_img)) == 0:
                return upper, lower

        raise PolygonCalculationError()

    # pick up the current corners
    current_left = inner_vertices[0]
    current_right = inner_vertices[-1]

    # the original pivot points on which we pivoted to arrive at the inner vertices
    pivots = [bin_.pivot_point for i, bin_ in enumerate(bins) if i % 2 != 0]

    # the gradient of the line between the first three / last three pivots
    gradient_left = calculate_gradient(pivots[0], pivots[1]) + calculate_gradient(pivots[1], pivots[2])
    gradient_right = calculate_gradient(pivots[-2], pivots[-1]) + calculate_gradient(pivots[-3], pivots[-2])

    # calculate the new corners by pushing the existing corners in the direction of the gradient
    new_left = push_to_edge(current_left, gradient_left, direction=-1)     # -1 = left
    new_right = push_to_edge(current_right, gradient_right, direction=+1)  # +1 = right

    return new_left, new_right


def calculate_polygon(segmentation: np.array, bounding_box: np.array) -> np.array:
    """
    Calculate a tight polygon around the word as represented by the segmentation map.
    :param segmentation:
    :param bounding_box:
    :return:
    """
    segmentation = segmentation.astype(int)

    box_width, box_height = _calculate_box_dimensions(bounding_box)

    # don't calculate polygon for small bounding boxes (why? no explanation in original code)
    if box_width < MIN_BOUNDING_BOX_SIZE or box_height < MIN_BOUNDING_BOX_SIZE:
        return None

    # keep only the part of the segmentation map covered by the bounding box and align horizontally
    # unwarp is a function we'll use later to adjust coordinates according to the original segmentation map
    segmentation, unwarp = _cutout_and_warp(segmentation, bounding_box)

    # divide the segmentation map into equally sized bins. for each of those bins, calculate a center point and a
    # pivot point (see description in _calculate_bin_statistics). Only every second bin is an actual pivot bin,
    # but we will need the other bins to calculate in which direction to pivot.
    num_bins = NUM_PIVOT_POINTS * 2 + 1
    bins = list(_calculate_bin_statistics(segmentation, num_bins))

    # now follow two checks for bin heights that contain a bunch of magic numbers
    # it is unclear why these checks are done, why both checks are needed and why the magic numbers were set this way
    # first check uses all the bins
    bin_heights = [bin_.height for bin_ in bins]
    if box_height * 0.7 < np.max(bin_heights):
        raise PolygonCalculationError()

    # second check uses only the pivot bins (every second bin)
    pivot_bin_heights = [bin_.height for i, bin_ in enumerate(bins) if i % 2 != 0]
    bin_width = box_width / num_bins
    if bin_width < np.max(pivot_bin_heights) * 0.25:
        raise PolygonCalculationError()

    # for each pivot bin, calculate a pair of (upper, lower) polygon vertices
    vertices = list(_calculate_inner_vertices(bins))

    # the first/last vertices might be somewhat far from the edges -> calculate special corner points
    left, right = _calculate_corners(segmentation, bins, vertices)
    vertices = [left] + vertices + [right]

    # put the polygon together, starting from left-upper vertice and going clockwise
    upper_vertices = [upper for upper, lower in vertices]
    lower_vertices = [lower for upper, lower in vertices]
    poly = upper_vertices + list(reversed(lower_vertices))

    # all our coordinates are currently relative to the cutout_and_warped segmentation map we calculated at
    # the beginning of this function. need to unwarp to map them back to the original segmentation map.
    poly = [unwarp(vertice) for vertice in poly]

    return np.array(poly)
