import os
import pickle

from data_structures.knowledge_graph import KnowledgeGraph


def dump_knowledge_graph(PTransE_data_files, dump_path, knowledge_graph_num_entities):

    entity_2_id_file = os.path.join(PTransE_data_files, "entity2id.txt")
    relation_2_id_file = os.path.join(PTransE_data_files, "relation2id.txt")

    # TODO(dlei): check if I should be using the train, test and validation datasets to construct the graph.
    train_triples_file = os.path.join(PTransE_data_files, "train.txt")
    test_triples_file = os.path.join(PTransE_data_files, "test.txt")
    validation_triples_file = os.path.join(PTransE_data_files, "valid.txt")

    knowledge_graph = KnowledgeGraph(num_entities=knowledge_graph_num_entities,
                                     entity_2_id_file=entity_2_id_file,
                                     relation_2_id_file=relation_2_id_file)

    knowledge_graph.parse_triples_from_file(train_triples_file, head_index=0, relation_index=2, tail_index=1)
    knowledge_graph.parse_triples_from_file(test_triples_file, head_index=0, relation_index=2, tail_index=1)
    knowledge_graph.parse_triples_from_file(validation_triples_file, head_index=0, relation_index=2, tail_index=1)

    print("Writing FB15K dumps to disk")

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    with open(os.path.join(dump_path, "FB15K_PTransE_dump.pkl"), "wb") as fh:
        pickle.dump(knowledge_graph, fh, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(dump_path, "FB15K_PTransE_dump.txt"), "w") as fh:
        for node in knowledge_graph.get_nodes():
            fh.write(str(node) + "\n")


if __name__ == "__main__":
    FB15K_NUM_ENTITIES = 14951
    PTransE_DATA_FILES = "../../KB2E-david-local/PTransE/data"
    DUMP_PATH = "../../honours-data-round-2/FB15K/dumps"

    dump_knowledge_graph(
        PTransE_data_files=PTransE_DATA_FILES,
        dump_path=DUMP_PATH,
        knowledge_graph_num_entities=FB15K_NUM_ENTITIES
    )