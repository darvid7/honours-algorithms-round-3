import time
import os
from data_structures.knowledge_graph import KnowledgeGraphTriple
import pickle

OUT_DIR = ""

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


def process_file(path, filename):
    with open(os.path.join(path, filename), 'r') as fh:
        lines = [l.strip() for l in fh.readlines()]
    head, relation, tail = eval(lines[0].split(': ')[1])

    test_triple = KnowledgeGraphTriple(head=head, relation=relation, tail=tail)
    relation_type = int(lines[1][15])  # parses out the 3.

    start_hits_offset = 4
    num_failed_hits = 15

    head_failed_hits = [eval(t) for t in lines[start_hits_offset: num_failed_hits]]

    tail_failed_hits_start = start_hits_offset + 1 + num_failed_hits
    tail_failed_hits = [eval(t) for t in lines[tail_failed_hits_start: tail_failed_hits_start + 15]]
    head_id = test_triple.head
    tail_id = test_triple.tail
    relation_id = test_triple.relation
    if not os.path.exists(os.path.join(path, "word_rep")):
        os.makedirs(os.path.join(path, "word_rep"))
    with open(os.path.join(path, "word_rep", filename), 'w', encoding='utf-8') as fh:
        fh.write("%s\n" % test_triple)

        relation_word = relation_id_to_word[relation_id]
        head_mid = entity_id_to_mid[head_id]
        head_entity = lookup_entity_mid(head_mid)
        head_entity = head_entity if head_entity else head_mid
        tail_mid = entity_id_to_mid[tail_id]
        tail_entity = lookup_entity_mid(tail_mid)
        tail_entity = tail_entity if tail_entity else tail_mid

        fh.write("[Triple word rep]\n%s %s %s\n" % (head_entity, relation_word, tail_entity))



        fh.write("relation type: %s\n" % relation_type)
        fh.write("HEAD HITS\n")
        for head_failed_hit in head_failed_hits:
            head_id = int(head_failed_hit[1])
            relation_word = relation_id_to_word[relation_id]
            head_mid = entity_id_to_mid[head_id]
            head_entity = lookup_entity_mid(head_mid)
            head_entity = head_entity if head_entity else head_mid
            tail_mid = entity_id_to_mid[tail_id]
            tail_entity = lookup_entity_mid(tail_mid)
            tail_entity = tail_entity if tail_entity else tail_mid
            fh.write("%s %s %s\n" % (head_entity, relation_word, tail_entity))
        fh.write("TAIL HITS\n")
        head_id = test_triple.head
        for tail_failed_hit in tail_failed_hits:
            tail_id = int(tail_failed_hit[1])
            relation_word = relation_id_to_word[relation_id]
            head_mid = entity_id_to_mid[head_id]
            head_entity = lookup_entity_mid(head_mid)
            head_entity = head_entity if head_entity else head_mid
            tail_mid = entity_id_to_mid[tail_id]
            tail_entity = lookup_entity_mid(tail_mid)
            tail_entity = tail_entity if tail_entity else tail_mid
            fh.write("%s %s %s\n" % (head_entity, relation_word, tail_entity))






def process(path_to_dir):
    failed_hits_file_paths = os.listdir(path_to_dir)

    for failed_file in failed_hits_file_paths:
        if not failed_file.endswith('.txt'):
            continue
        process_file(path_to_dir, failed_file)


if __name__ == "__main__":
    data_dir = "../../honours-data-round-2/FB15K/model_out/evaluation/"

    relevant_files = [
        #"d2v_suggested_pv_dbow",
        #"d2v_suggested_pv_dm",
        "pte_pv_dbow",
        "pte_pv_dm"
    ]

    for f in relevant_files:
        print("Processing folder: %s" % f)
        full_path = os.path.join(data_dir, f)
        process(full_path)


