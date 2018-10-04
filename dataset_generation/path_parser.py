import pickle
import os

path_file = "paths_from_pcra_train.log_no_rev_relations.txt_constructed_paths.txt"
train_pte_raw_triples = "../../KB2E-david-local/PTransE/data/train.txt"
out_file = "constructed_paths_and_og_train_triples_as_ints.txt"

KNOWLEDGE_GRAPH_DUMP_PATH = "../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"


# Things I need to do the missing entity lookup.
with open(KNOWLEDGE_GRAPH_DUMP_PATH, "rb") as fh:
    kg = pickle.load(fh)

unique_path_set = set()


with open(path_file, "r") as fh:  # Contains processed e -r-> e -r-> e paths.
    for line in fh:
        line = eval(line)
        for i in range(len(line)):
            if i % 2 == 0:  # is entity.
                line[i] = kg.entity_to_id[line[i]]
            else:
                # Some relation ids are strings, just make sure they are ints to be consistent.
                line[i] = int(line[i])
        if str(line) in unique_path_set:
            print("Path already in set: %s" % line)
        else:
            unique_path_set.add(str(line))
print(len(unique_path_set))
# with open(train_pte_raw_triples, "r") as pte_train_triples_fh:
#     for line in pte_train_triples_fh:
#         line = line.strip().split()
#         e1, e2, r = line
#         e1_id = kg.entity_to_id[e1]
#         e2_id = kg.entity_to_id[e2]
#         r_id = kg.relation_to_id[r]
#         path = str([e1_id, r_id, e2_id])
#         if path not in unique_path_set:
#             unique_path_set.add(path)
#
# with open(out_file, "w") as out_fh:
#     for path in unique_path_set:  # Are strings.
#         out_fh.write(path + "\n")