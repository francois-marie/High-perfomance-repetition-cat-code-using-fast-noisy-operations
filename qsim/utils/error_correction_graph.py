from abc import ABC
from math import inf

import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph, draw, draw_networkx_edge_labels, get_edge_attributes

from qsim.utils.utils import combine


class ErrorCorrectionGraph(ABC):
    def __init__(
        self,
        num_rows: int,
        num_columns: int,
        graph_weights=None,
        verbose: bool = False,
    ):
        self.num_rows = (
            num_rows  # number of rounds of parity measurements + perfect
        )
        self.num_columns = num_columns  # number of ancillas
        self.g = Graph()
        self.edge_labels = {}
        self.create_nodes()
        if graph_weights is None:
            graph_weights = {
                'pvin': 0,
                'pvout': 0,
                'pdiag': 0,
                'phor': 0,
                'pbord': 0,
                'p_prepw': 0,
            }
        self.graph_weights = graph_weights
        self.verbose = verbose

    def create_nodes(self):
        # Create nodes
        for i in range(self.num_rows * self.num_columns):
            self.g.add_node(
                i, pos=(i // self.num_columns, i % self.num_columns)
            )
        # physical boundary node
        self.physical_boundary = self.num_rows * self.num_columns
        self.g.add_node(self.physical_boundary, pos=(0, -1))
        self.g.nodes[self.physical_boundary]['is_boundary'] = True

    def add_weight(
        self,
    ):
        self.add_vertical_weights()
        self.add_horizontal_weights()
        self.add_diagonal_weights()
        self.convert_probabilities_to_weights()

    def add_vertical_weights(self):
        # Vertical edges
        # input errors
        for i in range(self.num_rows - 1):
            # prepw and Z1Z2 inside the graph
            for j in range(self.num_columns - 1):
                node_id = i * self.num_columns + j
                self.add_edge(
                    node_id,
                    node_id + 1,
                    'pvin',
                    qubit_id=j + 1,
                    color='black',
                )
            # bottom qubit: prepw & Z1Z2 & CXw
            self.add_edge(
                i * self.num_columns + self.num_columns - 1,
                self.physical_boundary,
                'pvin',
                qubit_id=self.num_columns,
                color='b',
            )
            self.add_edge(
                i * self.num_columns + self.num_columns - 1,
                self.physical_boundary,
                'pbord',
                qubit_id=self.num_columns,
                color='g',
            )
            # top qubit: prepw & Z1Z2
            self.add_edge(
                i * self.num_columns,
                self.physical_boundary,
                'pvin',
                qubit_id=0,
                color='y',
            )
        # output errors
        for i in range(1, self.num_rows):
            # output errors inside the graph: Z2 & measw
            for j in range(self.num_columns - 1):
                node_id = i * self.num_columns + j
                self.add_edge(
                    node_id,
                    node_id + 1,
                    'pvout',
                    qubit_id=j + 1,
                    color='orange',
                )
            # top qubit: CXw & Z2 & measw
            self.add_edge(
                i * self.num_columns,
                self.physical_boundary,
                'pbord',
                qubit_id=0,
                color='grey',
            )
            self.add_edge(
                i * self.num_columns,
                self.physical_boundary,
                'pvout',
                qubit_id=0,
                color='cyan',
            )
            # bottom qubit: Z2 & measw
            self.add_edge(
                i * self.num_columns + self.num_columns - 1,
                self.physical_boundary,
                'pvout',
                qubit_id=self.num_columns,
            )

        # # input error of the last & perfect parity measurement
        # i = self.num_rows - 1
        # for j in range(self.num_columns - 1):
        #     # prepw inside the graph
        #     node_id = i * self.num_columns + j
        #     self.add_edge(
        #         node_id,
        #         node_id + 1,
        #         'p_prepw',
        #         qubit_id=j + 1,
        #         color='orange',
        #     )
        # # bottom qubit: prepw & CXw
        # self.add_edge(
        #     i * self.num_columns + self.num_columns - 1,
        #     self.physical_boundary,
        #     'p_prepw',
        #     qubit_id=self.num_columns,
        #     color='b',
        # )
        # self.add_edge(
        #     i * self.num_columns + self.num_columns - 1,
        #     self.physical_boundary,
        #     'pbord',
        #     qubit_id=self.num_columns,
        #     color='g',
        # )
        # # top qubit: prepw
        # self.add_edge(
        #     i * self.num_columns,
        #     self.physical_boundary,
        #     'p_prepw',
        #     qubit_id=0,
        #     color='y',
        # )

    def add_horizontal_weights(self):
        # Horizontal edges
        for i in range(self.num_rows - 1):
            for j in range(self.num_columns):
                node_id = i * self.num_columns + j
                self.add_edge(node_id, node_id + self.num_columns, 'phor')

    def add_diagonal_weights(self):
        # Diagonal edges
        for i in range(self.num_rows - 1):
            for j in range(self.num_columns - 1):
                node_id = i * self.num_columns + j
                self.add_edge(
                    node_id,
                    node_id + self.num_columns + 1,
                    'pdiag',
                    qubit_id=j + 1,
                )

    def add_weight_data_reconvergence(self, p: float, k2a: int):
        physical_boundary = self.num_rows * self.num_columns
        # output errors
        for i in range(1, self.num_rows):
            if i % k2a == 0:
                for j in range(self.num_columns - 1):
                    node_id = i * self.num_columns + j
                    self.add_edge(
                        node_id,
                        node_id + 1,
                        'p_tr',
                        qubit_id=j + 1,
                        color='red',
                        p=p,
                    )
                self.add_edge(
                    i * self.num_columns,
                    physical_boundary,
                    'p_tr',
                    qubit_id=0,
                    color='blue',
                    p=p,
                )
                self.add_edge(
                    i * self.num_columns + self.num_columns - 1,
                    physical_boundary,
                    'p_tr',
                    qubit_id=self.num_columns,
                    p=p,
                )
        self.convert_probabilities_to_weights()

    def convert_probabilities_to_weights(self):
        # Converting lists of errors probabilities into weights
        for edge in self.g.edges():
            p = combine(self.g.edges[edge]['error_probability'])
            self.g.edges[edge]['error_probability'] = p
            if p == 0:
                self.g.edges[edge]['weight'] = inf
            else:
                self.g.edges[edge]['weight'] = np.log((1 - p) / p)

    def plot_graph(self, labels_type=None, colors: bool = True, legend=True):
        print('>>> plotting graph')
        pos = {}
        for node in self.g.nodes():
            pos[node] = self.g.nodes[node]['pos']
        options = {
            'node_size': 5,
            'width': 1,
        }
        colors = (
            [self.g[u][v]['color'] for u, v in self.g.edges()]
            if colors
            else ['black' for _ in self.g.edges()]
        )
        figsize = (2 * self.num_rows, 2 * self.num_columns)
        if labels_type is None:
            edge_labels = None
        elif labels_type == 'error_probability':
            edge_labels = get_edge_attributes(self.g, 'error_probability')
            for edge in edge_labels.keys():
                weight = edge_labels[edge]
                edge_labels[edge] = round(weight, 3)
        elif labels_type == 'probability_names':
            edge_labels = self.edge_labels
        plt.figure(figsize=figsize)
        draw(
            self.g,
            pos=pos,
            with_labels=(edge_labels is not None),
            edge_color=colors,
            **options,
        )
        if edge_labels is not None:
            draw_networkx_edge_labels(self.g, pos, edge_labels=edge_labels)
        if legend:
            for edge_label, value in self.graph_weights.items():
                plt.plot([], [], label=f'{edge_label} = {value:.3f}')
        plt.legend(bbox_to_anchor=(1.05, 1))

    def __str__(self):
        print(f'Num columns: {self.num_columns}')
        print(f'Num rows: {self.num_rows}')
        for u, v, attr in self.g.edges(data=True):
            u, v = int(u), int(v)
            print(f'Node: ({u}, {v})')
            print(f'\t{attr.get("error_probability", 0)}')
            print(f'\t{attr.get("weight", 1)}')

    def add_edge(
        self,
        node1: int,
        node2: int,
        p_label: str,
        qubit_id=-1,
        color=None,
        p=None,
    ):
        if p is None:
            try:
                p = self.graph_weights[p_label]
            except KeyError:
                p = 0
        add_edge(
            g=self.g,
            node1=node1,
            node2=node2,
            p=p,
            qubit_id=qubit_id,
            color=color,
        )
        self.update_edge_label(key=(node1, node2), value=p_label)

    def update_edge_label(self, key, value: str):
        if key not in self.edge_labels:
            self.edge_labels[key] = value
        else:
            self.edge_labels[key] += f' + {value}'


def add_edge(g, node1: int, node2: int, p: float, qubit_id=-1, color=None):
    """Adding an edge to a error correction graph

    Args:
        g ([type]): [description]
        node1 ([type]): [description]
        node2 ([type]): [description]
        p ([type]): [description]
        qubit_id (int, optional): [description]. Defaults to -1.
    """
    if color is None:
        color = 'black'
    if p == 0:
        return
    try:
        if isinstance(g.edges[node1, node2]['error_probability'], np.float64):
            old_p = g.edges[node1, node2]['error_probability']
            # g.edges[node1, node2]['error_probability'] = [
            #     g.edges[node1, node2]['error_probability']
            # ]
            g.edges[node1, node2]['error_probability'] = combine([p, old_p])
            w = np.log((1 - (combine([p, old_p]))) / (combine([p, old_p])))
            g.edges[node1, node2]['weight'] = w
        if color != 'black':
            g.edges[node1, node2]['color'] = color
    except KeyError:
        w = np.log((1 - p) / p)
        g.add_edge(
            node1,
            node2,
            qubit_id=qubit_id,
            error_probability=p,
            color=color,
            weight=w,
        )
