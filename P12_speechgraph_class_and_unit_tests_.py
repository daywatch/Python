from dataclasses import dataclass
import networkx as nx
from typing import List
import re
import numpy as np
from collections import Counter
import pytest

@dataclass
class SpeechGraphBundle:
    Clean_word_list: List
    Graph: nx.classes.multidigraph.MultiDiGraph
    Number_of_nodes: int
    Number_of_edges: int
    Number_of_parallel_edges: int
    Number_of_repeated_edges: int
    Number_of_nodes_in_largest_strongly_connected_component: int
    Diameter: int
    Average_clustering_coefficient: float
    Average_shortest_path: float
    Density: float
    Average_total_degree: float
    L1: float
    L2: float
    L3: float

    @classmethod
    def calc(cls, raw_transcript):
        clean_word_list = cls.clean_transcript(raw_transcript)
        graph = cls.create_graph(clean_word_list)
        number_of_nodes = cls.get_node_number(graph)
        number_of_edges = cls.get_edge_number(graph)
        number_of_parallel_edges = cls.get_parallel_edges(graph)
        number_of_repeated_edges = cls.get_repeated_edges(graph, number_of_parallel_edges)
        number_of_nodes_in_largest_strongly_connected_component = cls.get_largest_strongly_connected_component(graph)
        diameter = cls.get_diameter(graph)
        average_clustering_coefficient = cls.get_average_clustering_coefficient(cls.convert_multidigraph_to_weighted_graph(graph))
        average_shortest_path = cls.get_average_shortest_path(graph)
        density = cls.get_density(graph)
        average_total_degree = cls.get_average_total_degree(graph)
        l1_info = cls.get_L1(graph)
        l1 = l1_info[1]
        l2_info = cls.get_L2(l1_info[0])
        l2 = l2_info[1]
        l3 = cls.get_L3(l1_info[0], l2_info[0])

        return cls(

            Clean_word_list=clean_word_list,
            Graph=graph,
            Number_of_nodes=number_of_nodes,
            Number_of_edges=number_of_edges,
            Number_of_parallel_edges=number_of_parallel_edges,
            Number_of_repeated_edges=number_of_repeated_edges,
            Number_of_nodes_in_largest_strongly_connected_component=number_of_nodes_in_largest_strongly_connected_component,
            Diameter=diameter,
            Average_clustering_coefficient=average_clustering_coefficient,
            Average_shortest_path=average_shortest_path,
            Density=density,
            Average_total_degree=average_total_degree,
            L1=l1,
            L2=l2,
            L3=l3
        )

    @staticmethod
    def clean_transcript(transcript):
        word_tokenizer = lambda x: x.split(" ")
        cleaned_text = re.sub("[^\w ]+", " ", transcript.lower().strip())
        words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]

        return words

    @staticmethod
    def create_graph(word_list):
        graph = nx.MultiDiGraph()
        graph.add_edges_from(list(zip(word_list[:-1], word_list[1:])))
        return graph

    @staticmethod
    def get_node_number(graph):
        return graph.number_of_nodes()

    @staticmethod
    def get_edge_number(graph):
        return graph.number_of_edges()

    @staticmethod
    def get_parallel_edges(graph):
        edge_count_dict = Counter(graph.edges())
        return (np.array(list(edge_count_dict.values())) > 1).sum()

    @staticmethod
    def get_repeated_edges(graph, parallel_edge_counts):
        edge_count_dict = Counter(graph.edges())  # can be its own class
        repeated_edge_list = list({*map(tuple, map(sorted, list(edge_count_dict.keys())))})
        reciprocal_counts = sum([( edge_count_dict[tuple(reversed(item))] * edge_count_dict[item]) for item in repeated_edge_list if len(set(item))>1 if tuple(reversed(item)) in edge_count_dict] )

        return reciprocal_counts + parallel_edge_counts


    @staticmethod
    def get_largest_strongly_connected_component(graph):
        strongly_connected_list = nx.strongly_connected_components(graph)
        if len(list(strongly_connected_list)) > 2:
            return len(max(nx.strongly_connected_components(graph), key=len))
        else:
            return 0

    @staticmethod
    def get_diameter(graph):
        shortest_path_list = [j.values() for (i,j) in nx.shortest_path_length(graph)]
        if len(shortest_path_list) > 1:
            return max([max(j.values()) for (i, j) in nx.shortest_path_length(graph)])
        else:
            return 0

    @staticmethod
    def get_average_clustering_coefficient(graph):
        if graph.number_of_nodes() > 0:
            return nx.average_clustering(graph,weight='weight')
        else:
            return 0

    @staticmethod
    def get_average_shortest_path(graph):
        if graph.number_of_nodes() > 0:
            return nx.average_shortest_path_length(graph)
        else:
            return 0

    @staticmethod
    def get_density(graph):
        number_of_nodes = graph.number_of_nodes()
        if number_of_nodes == 0:
            return 0
        else:
            return graph.number_of_edges() / (number_of_nodes ** 2)

    @staticmethod
    def get_average_total_degree(graph):
        number_of_nodes = graph.number_of_nodes()
        degrees = list(dict(graph.degree()).values())
        if number_of_nodes == 0:
            return 0
        else:
            return sum(degrees) / graph.number_of_nodes()

    @staticmethod
    def get_L1(graph):
        if graph.number_of_nodes() > 0:
            adj_matrix = nx.linalg.adjacency_matrix(graph).toarray()
            return adj_matrix, np.trace(adj_matrix)
        else:
            return None, 0

    @staticmethod
    def get_L2(adjacent_matrix):
        if isinstance(adjacent_matrix,np.ndarray):
            adj_matrix2 = np.dot(adjacent_matrix, adjacent_matrix)
            return adj_matrix2, np.trace(adj_matrix2) / 2
        else:
            return None, 0

    @staticmethod
    def get_L3(adjacent_matrix1, adjacent_matrix2):
        if isinstance(adjacent_matrix1,np.ndarray):
            adj_matrix3 = np.dot(adjacent_matrix2, adjacent_matrix1)
            return np.trace(adj_matrix3) / 3
        else:
            return 0

    def convert_multidigraph_to_weighted_graph(graph):
        weighted_graph = nx.DiGraph()
        for u, v in graph.edges():
            if weighted_graph.has_edge(u, v):
                weighted_graph[u][v]['weight'] += 1
            else:
                weighted_graph.add_edge(u, v, weight=1)
        return weighted_graph

if __name__ == '__main__':
    graph_bundle = SpeechGraphBundle.calc("apple orange banana orange")
    print(f"word list: {graph_bundle.Clean_word_list}")
    print(f"#nodes: {graph_bundle.Number_of_nodes}")
    print(f"#edges: {graph_bundle.Number_of_edges}")
    print(f"#PE: {graph_bundle.Number_of_parallel_edges}")
    print(f"#RE: {graph_bundle.Number_of_repeated_edges}")
    print(f"LSC: {graph_bundle.Number_of_nodes_in_largest_strongly_connected_component}")
    print(f"diameter: {graph_bundle.Diameter}")
    print(f"clustering: {graph_bundle.Average_clustering_coefficient}")
    print(f"average shortest path: {graph_bundle.Average_shortest_path}")
    print(f"density: {graph_bundle.Density}")
    print(f"average total degree: {graph_bundle.Average_total_degree}")
    print(f"l1: {graph_bundle.L1}")
    print(f"l2: {graph_bundle.L2}")
    print(f"l3 {graph_bundle.L3}")

    # nx.draw_networkx(graph_bundle.Graph, node_size=800, font_size=12, font_color='b')


@pytest.mark.parametrize("transcript, expected_bundle",
                         [
                             ("", (0,0)),
                             ("apple", (0,0)),
                             ("apple apple apple apple", (1,1)),
                             ("apple pear apple pear apple", (2,6)),
                             ("apple pear banana vegetables tree", (0,0)),
                             ("1 2 3 2 1 3 2", (1,4)),
                             ("Se debe donar mucho más dinero para que este departamento tenga éxito.",(0,0))
                         ])
def test_PE_and_RE(transcript, expected_bundle):
    calculation = SpeechGraphBundle.calc(transcript)
    calculated_bundle = (calculation.Number_of_parallel_edges,calculation.Number_of_repeated_edges)
    assert calculated_bundle == expected_bundle


@pytest.mark.parametrize("transcript, expected_bundle",
                         [
                             ("", (0,0,0,0,0,0)),
                             (" ",(0,0,0,0,0,0)),
                             ("apple", (0,0,0,0,0,0)),
                             ("apple pear", (2,1,0.25,1,0.5,1)),
                             ("apple apple", (1,1,1,0,0,2)),
                             ("apple lemon peach strawberry",(4,3,0.1875,3,5/6,1.5)),
                             ("I don't remember what I said", (6,6,1/6,5,65/30,2)),
                             ("Michelle Galler was on her way to work. When she observed a car colliding with a barrier and the engine uh, bursting into flames. She called-- Michelle called 911 uh, and brought an-- Uh a fire truck. And the police showed up to take care of the accident and still she managed to arrive at work on time.",(41,57,57/(41*41),14, 5.788414634146341,2.7804878048780486)),
                             ("Se debe donar mucho más dinero para que este departamento tenga éxito.",(12, 11, 11/(12*12), 11, 286/132,1.8333333333333333))
                         ])
def test_number_of_nodes_edges_density_diameter_averageshortestpath_averagetotaldegree(transcript, expected_bundle):
    calculation = SpeechGraphBundle.calc(transcript)
    calculated_bundle = (calculation.Number_of_nodes,calculation.Number_of_edges,calculation.Density,calculation.Diameter,calculation.Average_shortest_path,calculation.Average_total_degree)
    assert calculated_bundle == expected_bundle

@pytest.mark.parametrize("transcript, expected_bundle",
                         [
                             ("", (0,0)),
                             ("apple", (0,0)),
                             ("apple apple apple apple", (1,1)),
                             ("apple pear apple pear apple", (2,6)),
                             ("apple pear banana vegetables tree", (0,0)),
                             ("1 2 3 2 1 3 2", (1,4)),
                             ("Se debe donar mucho más dinero para que este departamento tenga éxito.",(0,0))
                         ])
def test_PE_and_RE(transcript, expected_bundle):
    calculation = SpeechGraphBundle.calc(transcript)
    calculated_bundle = (calculation.Number_of_parallel_edges,calculation.Number_of_repeated_edges)
    assert calculated_bundle == expected_bundle


@pytest.mark.parametrize("transcript, expected_bundle",
                         [
                             ("", 0),
                             ("apple apple apple", 0),
                             ("apple bee peach apple peach bee apple chocolate banana", 3),
                             ("apple pear banana vegetables tree", 1),
                             ("1 2 3 2 1 3 2 3 4 4 5 6 7 9 7 6 5 9", 4),
                             ("1 0 2 1 0 3 4", 3),
                             ("Se debe donar mucho más dinero para que este departamento tenga éxito.",1)
                         ])
def test_LSC(transcript, expected_bundle):
    calculation = SpeechGraphBundle.calc(transcript)
    calculated_bundle = (calculation.Number_of_nodes_in_largest_strongly_connected_component)
    assert calculated_bundle == expected_bundle


#???
@pytest.mark.parametrize("transcript, expected_bundle",
                         [
                             ("", (0,0,0,0)),
                             ("apple apple apple", (2,2,8/3,0)),
                             ("apple bee peach apple peach bee apple chocolate banana", (0,3,2,1)),
                             ("apple pear banana vegetables tree", (0,0,0,1)),
                             ("apple pear apple pear",(0,2,0,1)),
                             ("apple pear orange apple pear orange",(0,0,4,1)),
                             ("1 2 3 2 1 3 2 3 4 4 5 6 7 9 7 6 5 9", (1,8.5, 2.3333333333333335,1)),
                             ("1 0 2 1 0 3 4 4 3", (1, 1.5, 3.3333333333333335,1)),
                             ("Se debe donar mucho más dinero para que este departamento tenga éxito.",(0,0,0,1))
                         ])
def test_L123_Clustering(transcript, expected_bundle):
    calculation = SpeechGraphBundle.calc(transcript)
    calculated_bundle = (calculation.L1,calculation.L2,calculation.L3,calculation.Average_clustering_coefficient)
    assert calculated_bundle == expected_bundle

#("hello world", SpeechGraphBundle(1, 0, 0, 0, 0, 0)),
#("you might go", SpeechGraphBundle(0, 1, 0, 0, 0, 1))
# SpeechGraphBundle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

# rest: clustering, l1, l2, l3