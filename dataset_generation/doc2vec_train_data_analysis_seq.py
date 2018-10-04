import time

start = time.time()

seq = "190, 11231"


train_data_file = "constructed_paths_and_og_train_triples_as_ints.txt"

analysis_dict = {} # maps to count, word rep

count = 0
with open(train_data_file, "r") as fh:
    for line in fh:
        count += 1
        line = eval(line)

        sl = str(line)
        if seq in sl:

            for i in range(len(line)):
                d_id = line[i]
                if i % 2 != 0:  # relation
                    d_id = d_id + 14951  # Add entity offset.
                if d_id not in analysis_dict:
                    analysis_dict[d_id] = [0, "idk"]
                analysis_dict[d_id][0] += 1

counted_trained_items = list(analysis_dict.items())
counted_trained_items.sort(key=lambda t:t[1][0], reverse=True) # sort by count.

import pickle
import os
SUFFIX_DATA_DIR = "../../honours-data-round-2/FB15K/"

KG_DUMP_PATH = "dumps/FB15K_PTransE_dump.pkl"
with open(os.path.join(SUFFIX_DATA_DIR, KG_DUMP_PATH), "rb") as fh:
    kg = pickle.load(fh)

relation_id_to_word = {v:k for k, v in kg.relation_to_id.items()}
entity_id_to_mid = {v:k for k, v in kg.entity_to_id.items()}


MAPPED_ENTITIES_PATH = "entity_lookup/FB15K_PTransE_mapped_entities.pkl"
MAPPED_MISSING_MIDS_ENTITIES_PATH = "entity_lookup/PTransE_mapped_missing_mids.pkl"

with open(os.path.join(SUFFIX_DATA_DIR, MAPPED_ENTITIES_PATH), "rb") as fh:
    mapped_mid_entities = pickle.load(fh)


with open(os.path.join(SUFFIX_DATA_DIR, MAPPED_MISSING_MIDS_ENTITIES_PATH), "rb") as fh:
    mapped_missing_mid_entities = pickle.load(fh)

def lookup_entity_mid(entity_mid):
    try:
        if entity_mid in mapped_mid_entities:
            return mapped_mid_entities[entity_mid]['itemListElement'][0]['result']['name']
        elif entity_mid in mapped_missing_mid_entities:
            return mapped_missing_mid_entities[entity_mid]['word_representation']
        return False
    except Exception:
        return False


print("total train paths: %s" % count)

for i in range(len(counted_trained_items)):
    int_id, tup = counted_trained_items[i]
    if int_id > 14951:
        # Is relation.
        int_id = int_id - 14951
        word_rep = relation_id_to_word[int_id]
    else:
        mid_rep = entity_id_to_mid[int_id]
        word_rep = lookup_entity_mid(mid_rep)

    counted_trained_items[i][1][1] = word_rep

with open("seq_out_taraget_%s.txt" % seq, "w") as fh:
    for i in range(len(counted_trained_items)):
        _, tup = counted_trained_items[i]
        count, word_rep = tup
        # write out count, word rep
        fh.write("%s %s\n" % (count, word_rep))

print("Finished, took %s mins" % ((time.time() - start)/60))
