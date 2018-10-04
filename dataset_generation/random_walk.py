"""
This random walk code is adapted from the code published by the authors of deepwalk
https://github.com/phanein/deepwalk/blob/master/deepwalk/graph.py

Our goal is to find 30,000 paths of variable length (3, 5, 7) entities and relations to comprise as our dataset.

In this dataset, we need it to be comparable with PTransE so we need to ensure that the paths that they train on
(output of the PCRA algorithm) are contained as a subset of our paths.

This means that relations and entities that appear in the PTransE paper must appear in ours.
However, as we are using FB15k with around 15000 entities and 1300 relations generating 90,000 paths of varying lengths
should capture this.
"""

# Should I remove self loops?

import random

from data_structures.knowledge_graph import KnowledgeGraph


def random_walk(graph: KnowledgeGraph, max_entities_on_path, start_node):
    """ Does a random walk through the graph from a given start node.
    Note: the graph is of data_structures.knowledge_graph.KnowledgeGraph which is an adjacency list.
     It is constructed using entity and relation ids so the notion of a node is simply an index into the graph.

    :param graph: the knowledge graph.
    :param max_entities_on_path: length to truncate the search.
    :param start_node: node to start from.
    :return: a path of length max_len.
    """
    path = [start_node]
    max_len = max_entities_on_path * 2 - 1
    while True:
        current_node = path[-1]

        outgoing_edges = graph[current_node].get_outgoing_edges_as_list()

        if not outgoing_edges:  # Nothing to traverse down this path.
            break

        chosen_edge = random.choice(outgoing_edges)  # random.choice() takes a sequence that needs to support indexing.
        path.append(chosen_edge.relation)
        path.append(chosen_edge.tail)

        if len(path) >= max_len:  # Note: deepwalk does the same as this in ln: 138 graph.py
            break
    # TODO:
    # TODO: It seems that gensim's TaggedDocument accepts strings and integers, based on the result of this issue
    # https://github.com/RaRe-Technologies/gensim/issues/2171 either return a list of ints or list of strings here.

    # Will return something like ['9800', '844', '7374', '65', '7777'] so we can just eval() it when reading from disk
    # this is better than writing it as strings because gensim's TaggedDocument requires a list of strings.
    # [str(s) for s in path]
    return path


def random_walker_generator(graph: KnowledgeGraph, max_entities_on_path, num_paths_to_produce, force_path_len):
    """ Generates num_paths_to_produce of max length (max_entities_on_path * 2 - 1) from a given knowledge graph.

    :param graph: object of type data_structures.knowledge_graph.KnowledgeGraph.
    :param max_entities_on_path: max number of entities on the path, the total path length will be
        max_entities_on_path * 2 - 1 as each two entities are connected by a relation.
    :param num_paths_to_produce: the number of paths to produce.
    :param force_path_len: if True, will for all paths to be of length max_entities_on_path.
    """
    count = 0
    while count < num_paths_to_produce:
        # Start at a random node.
        start_node = random.choice(graph.get_nodes())
        start_node_entity_id = start_node.entity
        path = random_walk(graph, max_entities_on_path, start_node_entity_id)
        if force_path_len:
            if not len(path) == max_entities_on_path * 2 - 1:
                continue
        yield path
        count += 1


if __name__ == "__main__":
    import pickle

    KG_DUMP_PATH = "../../honours-data-round-2/FB15K_DUMP/FB15K_PTransE_dump.pkl"

    with open(KG_DUMP_PATH, "rb") as fh:
       kg = pickle.load(fh)

    kg.remove_self_loops(verbose=False)

    walk = random_walk(kg, 3, 0)
    print(walk)

    for path in random_walker_generator(kg, 3, 2):
        print(path)

