import os
import argparse
import pickle
from data_structures.knowledge_graph import *
# parser = argparse.ArgumentParser()
#
# parser.add_argument("-p", "--path")
# parser = parser.parse_args()

pcra_output_paths_files = ["paths_from_pcra_train.log_no_rev_relations.txt", "paths_from_pcra_test.log_no_rev_relations.txt"]

DIR_I_CARE_ABOUT = "../../KB2E-david-local/PTransE/doc2vec_transformation"
KNOWLEDGE_GRAPH_DUMP_PATH = "../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"


# Things I need to do the missing entity lookup.
with open(KNOWLEDGE_GRAPH_DUMP_PATH, "rb") as fh:
    kg = pickle.load(fh)
    entity_mid_2_id = kg.entity_to_id
    entity_int_id_2_mid = {v: k for k, v in entity_mid_2_id.items()}


def make_path_from_missing_entity(head_entity_mid: str, relation_1_id: str, relation_2_id: str, tail_entity_mid: str):
    head_entity_int_id = entity_mid_2_id[head_entity_mid]
    tail_entity_int_id = entity_mid_2_id[tail_entity_mid]
    relation_1_int_id = int(relation_1_id)
    relation_2_int_id = int(relation_2_id)

    candidate_connecting_entities = []
    constructed_paths = []

    outgoing_edges = kg[head_entity_int_id].outgoing_edges
    for adj_list_edge in outgoing_edges:
        adj_relation = adj_list_edge.relation
        adj_entity = adj_list_edge.tail
        if relation_1_int_id == adj_relation:
            candidate_connecting_entities.append((adj_entity))

    # Now we have all entities starting from the head and connected by relation 1.

    for candidate_connecting_entity_int_id in candidate_connecting_entities:
        outgoing_edges = kg[candidate_connecting_entity_int_id].outgoing_edges
        for adj_list_edge in outgoing_edges:
            adj_relation = adj_list_edge.relation
            adj_entity = adj_list_edge.tail

            if adj_relation == relation_2_int_id and adj_entity == tail_entity_int_id:
                missing_entity_mid = entity_int_id_2_mid[candidate_connecting_entity_int_id]
                path = [head_entity_mid, relation_1_int_id, missing_entity_mid, relation_2_int_id, tail_entity_mid]
                if path == ['/m/02k6hp', 76, '/m/0dh1n_', 1144, '/m/02k6hp']:
                    print("hi")
                constructed_paths.append(path)
    return constructed_paths


for pcra_output_file in pcra_output_paths_files:
    relevant_file = os.path.join(DIR_I_CARE_ABOUT, pcra_output_file)
    """
    Format
    /m/01qscs 2546 missing_intermediate_entity 495 /m/02x8n1n
    /m/01qscs 2198 /m/02x8n1n
    """
    all_paths = []
    couldnt_find_path_count = 0
    with open(relevant_file, "r") as fh:
        for line in fh:
            # This is a path!
            line = eval(line)
            if len(line) > 3:
                # Has a missing entity in index 2.
                head_entity_mid, relation_1_id, _, relation_2_id, tail_entity_mid = line
                constructed_paths = make_path_from_missing_entity(head_entity_mid, relation_1_id, relation_2_id, tail_entity_mid)
                if not constructed_paths:
                    couldnt_find_path_count += 1
                all_paths.extend(constructed_paths)
            elif len(line) == 3:
                all_paths.append(line)
            else:
                print("WTF " + str(line))

    out_path = pcra_output_file + "_constructed_paths.txt"
    # with open(out_path, "w") as fh:
    #     for line in all_paths:
    #         fh.write(str(line) + "\n")

    print("Finished processing %s" % pcra_output_file)
    print("Couldn't find %s paths" % couldnt_find_path_count)
    print("Found %s many paths" % len(all_paths))
    words = 14951 + 1345
    unique_embds = words + len(all_paths)
    emb_size = 1000
    bytes = emb_size * unique_embds
    gb = bytes / 1024 / 1024 / 1024
    print("Estimated model size %s GB" % gb)

