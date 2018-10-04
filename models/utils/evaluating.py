from typing import List

from gensim.models.doc2vec import Doc2Vec
from data_structures.knowledge_graph import KnowledgeGraphTriple, KnowledgeGraph


def parse_test_dataset_word_representation(test_data_file_path, kg: KnowledgeGraph) -> List[KnowledgeGraphTriple]:
    test_triples = []
    with open(test_data_file_path, "r") as fh:
        # Line format: /m/01qscs	/m/02x8n1n	/award/award_nominee/award_nominations./award/award_nomination/award
        for line in fh:
            head, tail, relation = line.strip().split()
            head_id = kg.entity_to_id[head]
            tail_id = kg.entity_to_id[tail]
            relation_id = kg.relation_to_id[relation]
            kgt = KnowledgeGraphTriple(head=str(head_id), tail=str(tail_id), relation=str(relation_id))
            test_triples.append(kgt)
    return test_triples


# Used for hits @ 10.
def calculate_ranking_score(model: Doc2Vec, head_id: str, tail_id: str, relation_id: str):
    sum_score = 0
    head_vector = model.wv[head_id]
    tail_vector = model.wv[tail_id]
    relation_vector = model.wv[relation_id]
    # If triple holds, sum score should be low?
    for i in range(len(head_vector)):
        # TODO: If use L2 norm change eqn.
        # Follows impl outlined by the paper and
        # https://github.com/darvid7/KB2E/blob/589f87f1d33935c7dbeea73e1177d5d079d7e844/TransE/Test_TransE.cpp#L98
        # if the triple holds then h + r ~= t meaning | h + r - t | should be low or close to 0.
        # so if low = | h + r - t | then dissimilarity can be measured as h + r - t or
        # - h - r +  t which is what is done below.
        sum_score += -abs(tail_vector[i] - head_vector[i] - relation_vector[i])
    return sum_score


def hit_at_10(model: Doc2Vec, head_id: str, tail_id: str, relation_id: str, all_entities: List[str]):
    hits_heads = []  # [(candidate_score: float, entity_id: str), ...]
    hits_tails = []
    try:
        test_triple_score = calculate_ranking_score(model, head_id=head_id, tail_id=tail_id, relation_id=relation_id)
        # print("test triple score: " + str(test_triple_score))
        # print("Test hits at 10 test_triple_score: %s" % test_triple_score)
        # Implements the raw stat and not filtered (removing triples in test, training and valid datasets).
        for entity_id in all_entities:
            if entity_id not in model.wv.vocab:  # TODO: Remove this hack that is due to training dataset.
                continue
            # Replaces the head entity.
            candidate_score = calculate_ranking_score(model, head_id=entity_id, tail_id=tail_id, relation_id=relation_id)
            hits_heads.append((candidate_score, entity_id))
            # Replaces the tail entity.
            candidate_score = calculate_ranking_score(model, head_id=head_id, tail_id=entity_id, relation_id=relation_id)
            hits_tails.append((candidate_score, entity_id))

        # These scores are negative the highest disimilarity are larger negatives.
        # sorting normally in acesendingr order will put the larger negatives (more dissimilar) and are at the front of
        # the array.
        # sorting in reverse order will put the largest values which have lowest dissimilarity as they are small
        # negatives at the front of the array.
        # this is equivalent of sorting normally then looping backwards like done by the authors of PTRansE.
        hits_heads.sort(reverse=True)
        hits_tails.sort(reverse=True)

        rank_hit_head = None
        rank_hit_tail = None

        for index, hit_tuple in enumerate(hits_heads):
            candidate_score, entity_id = hit_tuple
            if entity_id == head_id:
                rank_hit_head = index # This is our hit.
                break

        for index, hit_tuple in enumerate(hits_tails):
            candidate_score, entity_id = hit_tuple
            if entity_id == tail_id:
                rank_hit_tail = index  # this is our hit.
                break
        return rank_hit_head, rank_hit_tail

    except KeyError as e:
        # print("[Error] %s" % e)
        return "Error", "Error"

# TODOL Hits at 10 filtered.
# TODO: Hits at 10 filtered, is this cheating if it is out of all valid outgoing edges from the head?


def query_closest_entity_by_vector(model: Doc2Vec,
                                   lookup_vector, top_n: int, total_num_entities: int, total_num_relations: int):
    max_to_consider = total_num_relations + top_n
    # TODO: .most_similar() should return results sorted by highest sim, double check this.
    # TODO: .most_similar() uses cosine similarity, might need to implement euclidean distance.
    nearest = model.wv.most_similar([lookup_vector], topn=max_to_consider)
    results = []
    for string_id, similarity in nearest:
        int_id = int(string_id)
        if int_id >= total_num_entities:
            # This is a relation, don't consider it.
            continue
        results.append((string_id, similarity))
        if len(results) == top_n:
            return results
    raise Exception("WTF how did it get here?")

