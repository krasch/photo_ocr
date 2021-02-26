from collections import namedtuple
import cv2

Component = namedtuple("Component", ["centroid_x", "centroid_y",
                                     "left", "top", "width", "height",
                                     "size", "segmentation"])


def parse_connected_components(components):
    """
    Makes output from cv2.connectedComponentsWithStats more accessible
    :param components: output from cv2.connectedComponentsWithStats (tuple of num_components, labels, stats, centroids)
    :return: iterator over parsed components
    """
    num_components, labels, stats, centroids = components

    # 0 = background -> ignore, start loop at i=1
    for i in range(1, num_components):
        component = Component(centroid_x=centroids[i][0],
                              centroid_y=centroids[i][1],
                              left=stats[i, cv2.CC_STAT_LEFT],
                              top=stats[i, cv2.CC_STAT_TOP],
                              width=stats[i, cv2.CC_STAT_WIDTH],
                              height=stats[i, cv2.CC_STAT_HEIGHT],
                              size=stats[i, cv2.CC_STAT_AREA],
                              segmentation=(labels == i))
        yield component
