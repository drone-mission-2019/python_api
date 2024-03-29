from rtree import index


class Tree(object):
    def __init__(self, X):
        """
        Tree representation
        :param X: Search Space
        """
        p = index.Property()
        p.dimension = X.dimensions
        self.V = index.Index(interleaved=True, properties=p)  # vertices in an rtree
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent
        self.V_obs = index.Index(interleaved=True, properties=p)  # vertices of obstacles in rtree
