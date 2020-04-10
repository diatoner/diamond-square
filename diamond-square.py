import argparse
from enum import IntEnum
import random
from typing import Callable, Dict, Tuple


class SquareNodeGraph:
    def __init__(self, size: int):

        if size < 3:
            raise ValueError('Must be of size 3 or more!')

        if size % 2 == 0:
            raise ValueError('Must have odd-numbered size!')

        # Initialise some data structures
        self.size = size
        self.nodes: Dict[str, float]  = {}  # Coordinates are screen-space. Top left is (0,0)

        # Populate node graph with zeros
        for x in range(self.size):
            for y in range(self.size):
                self.set_node(x, y, 0.0)
    
    def get_node(self, x: int, y: int) -> float:
        coordinate_hash = SquareNodeGraph.get_coordinate_hash(x, y)
        return self.nodes[coordinate_hash]
    
    def set_node(self, x: int, y: int, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError('Node value must be within 0 and 1, inclusive!')

        coordinate_hash = SquareNodeGraph.get_coordinate_hash(x, y)
        self.nodes[coordinate_hash] = value

    @staticmethod
    def get_coordinate_hash(x: int, y: int) -> str:
        return f'{x},{y}'

    @staticmethod
    def get_coordinate_pair(coordinate_hash: str) -> [int, int]:
        return coordinate_hash.split(',')

    def __str__(self):
        total_output = ''

        for y in range(self.size):

            output_line = ''
            for x in range(self.size):
                output_line += str(self.get_node(x, y))[:3]
                output_line += '--' if x < self.size-1 else ''
            total_output += output_line

            if y < self.size-1:
                vertical_joins = ' '
                vertical_joins += '|    ' * self.size
                total_output += ('\n' + vertical_joins) * 2 + '\n'

        return total_output

    def as_nparray(self):
        arr = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                value = self.get_node(x, y)
                arr[y, x] = int(255 * value)
        return arr


class Edge(IntEnum):
    Top = 1
    Left = 2
    Bottom = 4
    Right = 8


def clamp(a: float, n: float, b: float) -> float:
    return max(a, min(b, n))


def get_midpoint_from_tuples(first: [int, int], second: [int, int]) -> [int, int]:
    return [int(0.5 * (first[0] + second[0])), int(0.5 * (first[1] + second[1]))]


def get_edge_coordinates_from_corners(corner_coordinates: Dict[Edge, Tuple[int, int]]) -> Dict[Edge, Tuple[int, int]]:
    return {
        Edge.Top: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Left],
            corner_coordinates[Edge.Top | Edge.Right],
        ),
        Edge.Left: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Left],
            corner_coordinates[Edge.Bottom | Edge.Left],
        ),
        Edge.Bottom: get_midpoint_from_tuples(
            corner_coordinates[Edge.Bottom | Edge.Left],
            corner_coordinates[Edge.Bottom | Edge.Right],
        ),
        Edge.Right: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Right],
            corner_coordinates[Edge.Bottom | Edge.Right],
        ),
    }


def diamond_square(
    graph: SquareNodeGraph,
    corner_coordinates: Dict[Edge, Tuple[int, int]],
    noise_scaling_factor: float = 1,
    noise_scaling_function: Callable = lambda f: f * 0.5
):

    # Find edge midpoints
    edges = get_edge_coordinates_from_corners(corner_coordinates)

    # Set edge values, storing them for later (when the midpoint needs to be set)
    edge_values: List[float] = []
    for edge, edge_coords in edges.items():

        # Gather relevant corner values
        relevant_corner_values: List[float] = []
        for corner, corner_coords in corner_coordinates.items():

            # Recursion tracking: if the edge _is_ a corner, exit the function
            # because clearly we've gone too deep
            is_edge_corner_collision: bool = edge_coords == corner_coords
            if is_edge_corner_collision:
                return

            is_relevant_corner: bool = (edge & corner) != 0
            if is_relevant_corner:
                corner_value = graph.get_node(corner_coords[0], corner_coords[1])
                relevant_corner_values.append(corner_value)

        # Calculate the average corner value
        mean_corner_value = sum(relevant_corner_values) / len(relevant_corner_values) 

        # Add some noise, scaled arbitrarily
        edge_value = mean_corner_value + random.uniform(-1, 1) * noise_scaling_factor
        edge_value = float(clamp(0, edge_value, 1))

        if graph.get_node(edge_coords[0], edge_coords[1]) == 0:
            edge_values.append(edge_value)
        else:
            edge_values.append(graph.get_node(edge_coords[0], edge_coords[1]))

        # Set the edge value on the graph
        graph.set_node(edge_coords[0], edge_coords[1], edge_value)

    # Find midpoint coordinates
    midpoint_coords = get_midpoint_from_tuples(
        corner_coordinates[Edge.Top | Edge.Left],
        corner_coordinates[Edge.Bottom | Edge.Right],
    )

    # Set midpoint value to the mean of its edges, plus some noise
    mean_edge_value = sum(edge_values) / len(edge_values)
    midpoint_value = mean_edge_value + random.uniform(-1, 1) * noise_scaling_factor
    midpoint_value = float(clamp(0, midpoint_value, 1))
    graph.set_node(midpoint_coords[0], midpoint_coords[1], midpoint_value)

    # Define sub-quads for recursion in terms of their corner vertices
    corner_subsets = [
        {
            Edge.Top | Edge.Left: corner_coordinates[Edge.Top | Edge.Left],
            Edge.Top | Edge.Right: edges[Edge.Top],
            Edge.Bottom | Edge.Left: edges[Edge.Left],
            Edge.Bottom | Edge.Right: midpoint_coords,
        },
        {
            Edge.Top | Edge.Left: edges[Edge.Top],
            Edge.Top | Edge.Right: corner_coordinates[Edge.Top | Edge.Right],
            Edge.Bottom | Edge.Left: midpoint_coords,
            Edge.Bottom | Edge.Right: edges[Edge.Right],
        },
        {
            Edge.Top | Edge.Left: edges[Edge.Left],
            Edge.Top | Edge.Right: midpoint_coords,
            Edge.Bottom | Edge.Left: corner_coordinates[Edge.Bottom | Edge.Left],
            Edge.Bottom | Edge.Right: edges[Edge.Bottom],
        },
        {
            Edge.Top | Edge.Left: midpoint_coords,
            Edge.Top | Edge.Right: edges[Edge.Right],
            Edge.Bottom | Edge.Left: edges[Edge.Bottom],
            Edge.Bottom | Edge.Right: corner_coordinates[Edge.Bottom | Edge.Right],
        },
    ]

    # Apply the diamond-square algorithm recursively to those sub-quads
    for subset in corner_subsets:
        diamond_square(
            graph,
            subset,
            noise_scaling_function(noise_scaling_factor),
            noise_scaling_function
        )


if __name__ == '__main__':

    # Usage: python diamond-square.py [size]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'size',
        type=int,
        help='The side length of the square node graph to generate',
        default=3,
    )

    args = parser.parse_args()
    size = int(args.size)

    # Set up the graph we'll be working on
    graph = SquareNodeGraph(size)

    # Initialise the corners
    graph.set_node(0, 0, random.uniform(0, 1))
    graph.set_node(0, size-1, random.uniform(0, 1))
    graph.set_node(size-1, 0, random.uniform(0, 1))
    graph.set_node(size-1, size-1, random.uniform(0, 1))

    # Conduct the actual algorithm
    diamond_square(
        graph=graph,
        corner_coordinates={
            Edge.Top | Edge.Left: [0, 0],
            Edge.Top | Edge.Right: [size-1, 0],
            Edge.Bottom | Edge.Left: [0, size-1],
            Edge.Bottom | Edge.Right: [size-1, size-1],
        },
        noise_scaling_factor=1,
        noise_scaling_function=lambda f: f / 3
    )

    # Output as NumPy array
    # array = grid.as_nparray()
    # ... do some plotting

    print(graph)
