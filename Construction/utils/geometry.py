"""From https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/?ref=lbp
TODO: refactor
"""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def get_segments(cell):
    row, col = cell
    row = float(row)
    col = float(col)
    return [
        [(row, col), (row, col + 1)],
        [(row, col), (row + 1, col)],
        [(row + 1, col + 1), (row, col + 1)],
        [(row + 1, col + 1), (row + 1, col)],
    ]


def is_visible(cell_A, cell_B, wall):
    """
    Is cell B visible from cell A through the wall

    Args
        cell_A, cell_B, wall: tuple of ints row, col
    """
    p1 = Point(cell_A[0] + 0.5, cell_A[1] + 0.5)
    q1 = Point(cell_B[0] + 0.5, cell_B[1] + 0.5)
    for segment in get_segments(wall):
        p2, q2 = Point(*segment[0]), Point(*segment[1])
        if doIntersect(p1, q1, p2, q2):
            return False
    return True
