import time
import os
from data_structures.knowledge_graph import KnowledgeGraphTriple
import pickle
"""
Test triple: (10005, 875, 727) # (h: 935, r: 655, t: 8353)
Relation type: 3, many-to-many
Hit at 10 - head: False, tail: False
HEAD HITS
(-29.38029938354157, '727')
(-30.65288613177836, '10217')
(-30.69479403924197, '1257')
(-30.747158598154783, '477')
(-30.84070898592472, '2308')
(-30.905750952661037, '1844')
(-30.935720498673618, '14761')
(-31.15135241858661, '13046')
(-31.227668860927224, '14704')
(-31.409513311460614, '1649')
(-31.41770484112203, '10457')
(-31.422240833751857, '12919')
(-31.44576427526772, '14825')
(-31.448252241127193, '13086')
(-31.452052319422364, '6248')
TAIL HITS
(-14.161205190233886, '875')
(-16.48345481418073, '8020')
(-16.77982402034104, '11640')
(-18.47306928411126, '7584')
(-18.89617403782904, '1600')
(-19.111332394182682, '4360')
(-19.645306289196014, '6115')
(-19.937525417655706, '13878')
(-20.1098378226161, '13886')
(-20.57825714442879, '12135')
(-21.098912866786122, '6888')
(-21.099131640046835, '4834')
(-21.387501686811447, '14620')
(-21.38976562023163, '2752')
(-21.410351011902094, '10404')
"""

RELATION_TYPES = {'one-to-one': 0, 'many-to-one': 1, 'one-to-many': 2, 'many-to-many': 3}
RELATION_TYPE_ID_TO_TYPE = {v: k for k, v in RELATION_TYPES.items()}


files = []

def open_failed_hit_at_10_file(file_path, filtered, train_triples):
    """

    :return: the test triple, the relation type, hits at 10 for replacing the head entity, hits at 10 for replacing the tail entitiy.
    """

    with open(file_path, 'r') as fh:
        lines = [l.strip() for l in fh.readlines()]
    head, relation, tail = eval(lines[0].split(': ')[1])
    # int ids.

    test_triple = KnowledgeGraphTriple(head=head, relation=relation, tail=tail)
    relation_type = int(lines[1][15]) # parses out the 3.

    start_hits_offset = 4
    num_failed_hits = 15

    head_failed_hits = [eval(t) for t in lines[start_hits_offset: num_failed_hits]]

    tail_failed_hits_start = start_hits_offset + 1 + num_failed_hits
    tail_failed_hits = [eval(t) for t in lines[tail_failed_hits_start: tail_failed_hits_start + 15]]

    head_hit = None
    tail_hit = None

    # Some weird stuff going on where after replacing the head the tail often pops up as a hit and vice versa.
    head_replaced_with_tail_hit = None
    tail_replaced_with_head_hit = None

    filtered_sub_head = 0
    filtered_sub_tail = 0

    # There is a bug here somewhere.
    for index, hit_tuple in enumerate(head_failed_hits):  # 15.

        candidate_score, entity_id = hit_tuple
        # I was comparing str == int before..oops.
        if int(entity_id) == int(test_triple.head):
            head_hit = index  # This is our hit.

        if filtered:
            cur = "%s %s %s" % (int(entity_id), tail, relation)
            if cur in train_triples:
                filtered_sub_head += 1

        # Ignore this one for now.
        if int(entity_id) == int(test_triple.tail):
            head_replaced_with_tail_hit = index

    for index, hit_tuple in enumerate(tail_failed_hits):  # 15.

        candidate_score, entity_id = hit_tuple
        # I was comparing str == int before..oops.
        if int(entity_id) == int(test_triple.tail):
            tail_hit = index  # this is our hit.

        # Ignore this one for now.
        if int(entity_id) == int(test_triple.head):
            tail_replaced_with_head_hit = index

        if filtered:
            cur = "%s %s %s" % (head, int(entity_id), relation)
            if cur in train_triples:
                filtered_sub_tail += 1
    if filtered:
        head_hit -= filtered_sub_head
        tail_hit -= filtered_sub_tail


    return test_triple, relation_type, head_hit, tail_hit, head_replaced_with_tail_hit, tail_replaced_with_head_hit

def print_evaluation_result(hit_at_10_dict):
    relation_types = ['one-to-one', 'many-to-one', 'one-to-many', 'many-to-many']
    for relation_type in relation_types:
        relation_type_key = RELATION_TYPES[relation_type]
        print("\t%s: %s/%s = %s\n%s" % (
            relation_type,
            hit_at_10_dict[relation_type_key][0],
            hit_at_10_dict[relation_type_key][1],
            hit_at_10_dict[relation_type_key][0] / hit_at_10_dict[relation_type_key][1] if
                hit_at_10_dict[relation_type_key][1] != 0 else "n/a",
            hit_at_10_dict[relation_type_key][0] / hit_at_10_dict[relation_type_key][1] * 100 if
            hit_at_10_dict[relation_type_key][1] != 0 else "n/a"
        ))

def calculate_fail_hits_at_10(path_to_dir, filtered=False):
    start = time.time()
    # dict maps relation types to a list representing [num_hits, total_num]

    # hit@10 for replacing the head entity.
    predicting_head_hit_at_10 = {
        RELATION_TYPES['one-to-one']: [0, 0],
        RELATION_TYPES['many-to-one']: [0, 0],
        RELATION_TYPES['one-to-many']: [0, 0],
        RELATION_TYPES['many-to-many']: [0, 0]
    }

    # When we replace the head, if it the tail is a hit at 10 store it here.
    rev_weird_predicting_head_replaced_with_tail_hit_at_10 = {
        RELATION_TYPES['one-to-one']: [0, 0],
        RELATION_TYPES['many-to-one']: [0, 0],
        RELATION_TYPES['one-to-many']: [0, 0],
        RELATION_TYPES['many-to-many']: [0, 0]
    }

    # hit@10 for replacing the tail entity.
    predicting_tail_hit_at_10 = {
        RELATION_TYPES['one-to-one']: [0, 0],
        RELATION_TYPES['many-to-one']: [0, 0],
        RELATION_TYPES['one-to-many']: [0, 0],
        RELATION_TYPES['many-to-many']: [0, 0]
    }

    # When we replace the tail, if the head is a hit at 10 store it here.
    rev_weird_predicting_tail_replaced_with_head_hit_at_10 = {
        RELATION_TYPES['one-to-one']: [0, 0],
        RELATION_TYPES['many-to-one']: [0, 0],
        RELATION_TYPES['one-to-many']: [0, 0],
        RELATION_TYPES['many-to-many']: [0, 0]
    }

    mean_head_hit_rank = 0
    mean_tail_hit_rank = 0
    head_hit_at_10_total = 0
    tail_hit_at_10_total = 0

    train_triples = set()

    if filtered:

        KG_DUMP_PATH = "../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"
        with open(KG_DUMP_PATH, "rb") as fh:
            kg = pickle.load(fh)

        train_data_path = "../../KB2E-david-local/PTransE/data/train.txt"
        with open(train_data_path, "r") as fh:
            for line in fh:
                h, t, r = line.split()
                h_id = kg.entity_to_id[h]
                t_id = kg.entity_to_id[t]
                r_id = kg.relation_to_id[r]
                train_triples.add("%s %s %s" % (h_id, t_id, r_id))


    failed_hits_file_paths = os.listdir(path_to_dir)

    for failed_file in failed_hits_file_paths:
        if not failed_file.endswith(").txt"):
            continue
        #  test_triple, relation_type, head_hit, tail_hit, head_replaced_with_tail_hit, tail_replaced_with_head_hit
        test_triple, relation_type_id, head_hit, tail_hit,  head_replaced_with_tail_hit, tail_replaced_with_head_hit = \
            open_failed_hit_at_10_file(os.path.join(path_to_dir, failed_file), filtered, train_triples)

        predicting_head_hit_at_10[relation_type_id][1] += 1
        if head_hit and head_hit < 10:
            predicting_head_hit_at_10[relation_type_id][0] += 1
            head_hit_at_10_total += 1
        # mean_head_hit_rank += head_hit

        predicting_tail_hit_at_10[relation_type_id][1] += 1
        if tail_hit and tail_hit < 10:
            predicting_tail_hit_at_10[relation_type_id][0] += 1
            tail_hit_at_10_total += 1
        # mean_tail_hit_rank += tail_hit

    if filtered:
        print("\t FILTERED~~")
    else:
        print("\t RAW~")
    print("[calculate_fail_hits_at_10 for %s] took %s minutes" % (path_to_dir, (time.time() - start) / 60))

   # print("AVG")
   #  print("Mean rank head: %s" % mean_head_hit_rank/head_hit_at_10_total)
   # print("Mean rank tail: %s" % mean_tail_hit_rank/tail_hit_at_10_total)

    print("Hits @ 10 predicting heads")
    print_evaluation_result(predicting_head_hit_at_10)
    # print("\nHits @ 10 rev_weird_predicting_head_replaced_with_tail_hit_at_10")
    # print_evaluation_result(rev_weird_predicting_head_replaced_with_tail_hit_at_10)

    print("\nHits @ 10 predicting tails")
    print_evaluation_result(predicting_tail_hit_at_10)
    # print("\nHits @ 10 rev_weird_predicting_tail_replaced_with_head_hit_at_10")
    # print_evaluation_result(rev_weird_predicting_tail_replaced_with_head_hit_at_10)


if __name__ == "__main__":
    # test_path = "../../honours-data-round-2/FB15K/model_out/evaluation/d2v_suggested_pv_dbow/hit_@_10_(0, 421, 13534).txt"
    #
    # test_triple, relation_type, head_hit, tail_hit = open_failed_hit_at_10_file(test_path)
    # print("\ttesting if this works >> ")
    # print("test triple: %s" % test_triple)
    # print("relation type: %s" % relation_type)
    # print("head hit: %s" % head_hit)
    # print("tail hit: %s" % tail_hit)

    data_dir = "../../honours-data-round-2/FB15K/model_out/evaluation/"

    relevant_files = [
        "d2v_suggested_pv_dbow",
        "d2v_suggested_pv_dm",
        "pte_pv_dbow",
        "pte_pv_dm"
    ]

    for f in relevant_files:
        print("Processing folder: %s" % f)
        full_path = os.path.join(data_dir, f)
        calculate_fail_hits_at_10(full_path)
        #calculate_fail_hits_at_10(full_path, filtered=True)




