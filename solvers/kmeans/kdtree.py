from operator import itemgetter


class KdTree(object):
    def __init__(self, points):
        self._root = build_kdtree(points)

    def search_kdtree(self, point, valid_dims, dims_bool):
        # type: (KdTree, np.array, np.array, np.array) -> int
        stack = list(self.root)

        cur_best = None
        ## DFS
        while len(stack > 0):
            node = stack[-1]  # peek
            dim = node.depth % point.size

            if ~dims_bool[dim]:
                cur_best_left = node.left.find_leaf(point, valid_dims, dims_bool)
                cur_best_right = node.right.find_leaf(point, valid_dims, dims_bool)

                dist_left = 0
                dist_right = 0
                for i in valid_dims:
                    dist_left += abs(point[i] - cur_best_left[dim])
                    dist_right += abs(point[i] - cur_best_right[dim])
                dist_left /= valid_dims.size
                dist_right /= valid_dims.size

                if dist_left < dist_right:
                    cur_best = cur_best_left
                else:
                    cur_best = cur_best_right

            else:
                if point[dim] < node._loc[dim]:
                    cur_best = node._left.find_leaf(point, valid_dims, dims_bool)
                else:
                    cur_best = node._right.find_leaf(point, valid_dims, dims_bool)


class Node(object):
    def __init__(self, left, right, loc, depth):
        self.left = left
        self.right = right
        self.loc = loc
        self.depth = depth


def build_kdtree(self, points, depth=0):
    # type: (list[np.ndarray], int) -> Node
    try:
        k = points[0].size
    except IndexError:
        return None

    dim = depth % k

    # Sort point list and choose median as pivot element
    points.sort(key=itemgetter(dim))
    median = len(points) // 2  # choose median

    return Node(
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1),
        loc=points[median],
        depth=depth)


