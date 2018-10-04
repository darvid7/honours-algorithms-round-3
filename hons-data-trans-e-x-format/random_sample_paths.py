import pickle
import os

from typing import List

from dataset_generation.random_walk import random_walker_generator
from data_structures.knowledge_graph import KnowledgeGraph


def dump_paths(path_dump_dir: str, knowledge_graph: KnowledgeGraph,
               paths_per_entity_count: int, entity_counts: List[int]):
    if not os.path.exists(path_dump_dir):
        os.makedirs(path_dump_dir)

    for entity_count in entity_counts:
        dump_file = os.path.join(path_dump_dir, "max_entities_on_path_%s_paths.txt" % entity_count)
        with open(dump_file, "w") as fh:
            for path in random_walker_generator(graph=knowledge_graph, max_entities_on_path=entity_count,
                                                num_paths_to_produce=paths_per_entity_count, force_path_len=True):
                fh.write(str(path) + "\n")
            print("Dumped %s paths of entity length %s to %s" % (paths_per_entity_count, entity_count, dump_file))


if __name__ == "__main__":
    KG_DUMP_PATH = "../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"
    NUM_PATHS_PER_ENTITY_COUNT = 50000
    ENTITY_COUNTS = [4, 5, 6]

    with open(KG_DUMP_PATH, "rb") as fh:
        kg = pickle.load(fh)
    kg.remove_self_loops(verbose=False)

    dump_paths(path_dump_dir="./created_paths", knowledge_graph=kg,
               paths_per_entity_count=NUM_PATHS_PER_ENTITY_COUNT, entity_counts=ENTITY_COUNTS)
