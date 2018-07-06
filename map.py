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
        vector = np.array(vector)
        self.vector = vector
        self.pos = pos


class SOM:
    """
    Self-organizing map.
    n_dim      = dimensionality of input data.
    map_dim    = [height, width] of map
    learn_rate = learning rate for nodes
    map        = array of array of nodes (not a passed param)
    n_vis      = number of figures for visualization (not a passed param)
    """
    def __init__(self, n_dim, map_dim, learn_rate=0.05):
        self.n_dim = n_dim
        self.map_dim = map_dim
        self.learn_rate = learn_rate
        self.map = []
        self.plots = []

    def build_map(self):
        """
        Initializies map of map_dim size with nodes.
        """

        for x in range(self.map_dim[0]):
            for y in range(self.map_dim[1]):
                self.map.append(Node([np.random.uniform(0, 1) for x in range(self.n_dim)],
                                    [np.random.randint(0, self.map_dim[0]),
                                    np.random.randint(0, self.map_dim[1])]))

    def visualize_map(self, title=None, show=False):
        """
        Displays scatterplot of self organizing map nodes.
        Title is a string for the title of the scatterplot.
        Show is bool that determines when plt.show() will be called.
        """

        x = []
        y = []

        for node in self.map:
            x.append(node.pos[0])
            y.append(node.pos[1])

        global plt

        plt.scatter(x, y, label='Self Organizing Map', color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)

        if show:
            for i, plot in enumerate(self.plots):
                plot.figure(i)
            plt.show()


    def manhatten_distance(self, vector_x, vector_y):
        """
        Returns manhatten distance for vector_x and vector_y
        """
        return sum(abs(vector_x - vector_y))

    def fit(self, input_vector):
        """
        Returns best matching unit (BMU) - closest node vector weight to input_vector.
        """
        distances = {}
        for node in self.map:
            #print(input_vector, node.vector)
            node_dist = self.manhatten_distance(input_vector, node.vector)
            distances[node] = node_dist
        bmu = max(distances.items(), key=operator.itemgetter(1))[0]

        self.update_node(bmu)

        return bmu

    def show_node_vectors(self):
        [print('Node: {0} {1}'.format(i, node.vector)) for i, node in enumerate(self.map)]
        return

    def update_node(self, bmu):
        for node in self.map:
            if node == bmu:
                print(bmu.vector, node.vector)
                node.vector += self.learn_rate * (bmu.vector - node.vector)


def main():
    seed = 2
    np.random.seed(seed)

    X = [[1, 2, 3],
        [2, 4, 6],
        [1, 2, 3]]

    X = np.array(X)

    som = SOM(X[0].shape[0], [4, 4])

    som.build_map()

    som.visualize_map(title='Initialization')

    for i in X:
        print((som.fit(i)).vector)

    som.show_node_vectors()

    som.visualize_map(title='Post Training', show=True)


if __name__ == '__main__':
    main()

