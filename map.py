import matplotlib.pyplot as plt
import numpy as np
import operator

class Node:
    """
    Node in the self-organizing map.
    Vector with equal number of dimensions as input data.
    Pos of [x, y] on the map.
    """
    def __init__(self, vector, pos):
        self.vector = vector
        self.pos = pos


class SOM:
    """
    Self-organizing map.
    n_dim dimensionality of input data.
    map_dim [height, width] of map
    """
    def __init__(self, n_dim, map_dim, n_nodes=125):
        self.n_dim = n_dim
        self.num_nodes = n_nodes
        self.map_dim = map_dim
        self.map = []

    def build_map(self):
        """
        Initializies map of map_dim size with nodes.
        """

        for x in range(self.map_dim[0]):
            for y in range(self.map_dim[1]):
                self.map.append(Node([np.random.uniform(0, 1) for x in range(self.n_dim)],
                                    [np.random.randint(0, self.map_dim[0]),
                                    np.random.randint(0, self.map_dim[1])]))

    def visualize_map(self):
        """
        Displays scatterplot of self organizing map nodes.
        """

        x = []
        y = []

        for node in self.map:
            x.append(node.pos[0])
            y.append(node.pos[1])

        plt.scatter(x, y, label='Self Organizing Map', color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Self Organizing Map')
        plt.show()

    def manhatten_distance(self, vector_x, vector_y):
        """
        Returns manhatten distance for vector_x and vector_y
        """
        return sum(abs(vector_x - vector_y))

    def fit(self, input_vector):
        """
        Returns closest node vector weight to input_vector.
        """
        distances = {}
        for node in self.map:
            #print(input_vector, node.vector)
            node_dist = self.manhatten_distance(input_vector, node.vector)
            distances[node] = node_dist
        closest = max(distances.items(), key=operator.itemgetter(1))[0]
        return closest

    def get_node_vectors(self):
        [print('Node #: {0} {1}'.format(i, node.vector)) for i, node in enumerate(self.map)]


def main():
    seed = 3
    np.random.seed(seed)

    X = [[1, 2, 3],
        [2, 4, 6],
        [1, 2, 3]]

    X = np.array(X)

    som = SOM(X[0].shape[0], [2, 2])
    som.build_map()
    
    for i in X:
        print((som.fit(i)).vector)

    som.get_node_vectors()

    #som.visualize_map()

if __name__ == '__main__':
    main()

