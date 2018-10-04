import os
import time
import logging
import pickle
import copy
import multiprocessing
import numpy as np
from typing import List, Iterable

from gensim.models.doc2vec import Doc2Vec

from data_structures.knowledge_graph import KnowledgeGraphTriple

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class Model:
    # Hack because I know all the possible entity ids. Num ids = max index - 1.
    ALL_ENTITIES = [str(i) for i in range(14951)]
    RELATION_TYPES = {'one-to-one': 0, 'many-to-one': 1, 'one-to-many': 2, 'many-to-many': 3}
    RELATION_TYPE_ID_TO_TYPE = {v: k for k, v in RELATION_TYPES.items()}

    # Only accepts hyperparamters of gensim.models.doc2vec that we care about.
    def __init__(self, tag, distributed_memory: bool, epochs: int, learning_rate: float, dimensions: int,
                 write_out_dir: str):
        """

        :param tag:
        :param distributed_memory:
        :param epochs:
        :param learning_rate:
        :param dimensions:
        """
        self.num_cores = multiprocessing.cpu_count()  # Use all CPUs, num threads/processes = num CPUs.
        self.num_epochs = epochs  # All calls to train will have the same epoch.
        self.tag = tag  # Name of the model, seperate words with '_'.

        self.write_out_dir = write_out_dir
        self.trained = False

        # Populated after training the model to save memory, can delete model after.
        self.word_vectors = None
        self.doc_vectors = None
        self.word_vector_lookup = None  # TODO: might remove.
        self.doc_vector_lookup = None  # TODO: might remove.
        self.word_2_index = {}
        self.deleted_model = False

        self.relation_id_2_type = None

        self.d2v_model = Doc2Vec(
            min_count=1,  # A single entity/relation may occur once, we still want to consider it.
            window=1,  # Considers
            workers=self.num_cores,  # Use all CPU cores.
            vector_size=dimensions,  # Dimension of embeddings.
            alpha=learning_rate,
            # Trains word vectors simultaneously with PV-DBOW (dm=1),
            # has no effect if dm=0 https://groups.google.com/forum/#!topic/gensim/vGrnyFwPqd8
            dbow_words=1,
            dm=1 if distributed_memory else 0
        )
        print("Initalized model object: %s" % self.tag)
        print("\tavaliable cores: %s" % self.num_cores)
        print("\tepochs: %s" % self.num_epochs)
        print("\ttype: %s" % "Paragraph-Vector " +
              "Distributed Memory" if distributed_memory else "Distributed Bag of Words")
        print("\tlearning rate: %s" % learning_rate)
        print("\tdimensions: %s" % dimensions)
        self.train_triple_set = set()

    def build_vocab(self, sentences_can_be_generator: Iterable):
        """ Builds vocab for model, call this before .train() """
        start = time.time()
        self.d2v_model.build_vocab(sentences_can_be_generator)
        print("[%s_build_vocab] took %s minutes" % (self.tag, (time.time() - start) / 60))

    def train(self, sentences_not_generator: List):
        """ Train the model for self.num_epochs using self.num_cores threads.

        :param sentences_not_generator: Must be an iterable that isn't one use.
        """
        start = time.time()

        did_train_sanity_check = self.d2v_model.wv.syn0[0].copy()
        self.d2v_model.train(
            documents=sentences_not_generator,
            epochs=self.num_epochs,
            total_examples=self.d2v_model.corpus_count
        )

        if np.all(did_train_sanity_check == self.d2v_model.wv.syn0[0]):
            print("[%s train] training word vectors might not have gone as well" % self.tag)
        else:
            print("[%s train] word vectors trained!" % self.tag)
        print("[%s train] took %s minutes" % (self.tag, (time.time() - start) / 60))

    def save(self):
        """ Save trained model to disk, might not use this because querying vectors directly is faster. """
        self.d2v_model.save(self.tag + ".trained_model")

    def save_vectors_in_memory_and_delete_model(self):
        """ Save vectors of the trained model that we care about (word & doc vectors) then delete the model to
        reduce RAM consumption. """
        # Keep only embeddings we care about.
        self.word_vectors = copy.deepcopy(self.d2v_model.wv.syn0)
        self.word_vector_lookup = copy.deepcopy(self.d2v_model.wv.index2entity)
        for i, word in enumerate(self.word_vector_lookup):
            self.word_2_index[word] = i

        self.doc_vectors = copy.deepcopy(self.d2v_model.docvecs.vectors_docs)
        self.doc_vector_lookup = copy.deepcopy(self.d2v_model.docvecs.index2entity)
        # Can delete the model to save space now.
        del self.d2v_model
        self.deleted_model = True
        print("[%s save_vectors_in_memory_and_delete_model] model deleted" % self.tag)

    def save_vectors_from_model_to_disk(self):
        """ Dumps trained word vectors & doc vectors to disk from the model."""
        if self.deleted_model:
            print("[%s Error] trying to save from a deleted model" % self.tag)
            return

        start = time.time()
        # Dump trained word vectors (entities/relations).
        with open('%s_word_vectors.pkl' % self.tag, 'wb') as fh:
            pickle.dump(self.d2v_model.wv.syn0, fh)

        # Index to word lookups.
        with open('%s_word_vectors_lookup.txt' % self.tag, "w") as fh:
            # index2entity is a list where the index i returns the entity tag of i, vectors[i] is the vector rep of entity_tag[i].
            for i, lookup in enumerate(self.d2v_model.wv.index2entity):
                fh.write("%s:%s\n" % (i, lookup))  # index:word

        # Dump trained doc vectors (path)
        with open('%s_doc_vectors.pkl' % self.tag, 'wb') as fh:
            pickle.dump(self.d2v_model.docvecs.vectors_docs, fh)

        with open('%s_doc_vectors_lookup.txt' % self.tag, "w") as fh:
            for i, lookup in enumerate(self.d2v_model.docvecs.index2entity):
                fh.write("%s:%s\n" % (i, lookup))  # index:doc_tag

        end = time.time()
        print("[%s_save_vectors] took %s minutes to dump word & doc vectors" % (self.tag, (end - start) / 60))

    def calculate_score(self, head_vector, tail_vector, relation_vector):
        """ Calculate the disimilarity score for triples as in thunlp implemtation of PTransE & TransE."""
        sum_score = 0
        for i in range(len(head_vector)):
            # TODO: If use L2 norm change eqn.
            # Follows impl outlined by the paper and
            # https://github.com/darvid7/KB2E/blob/589f87f1d33935c7dbeea73e1177d5d079d7e844/TransE/Test_TransE.cpp#L98
            # if the triple holds then h + r ~= t meaning | h + r - t | should be low or close to 0.
            # so if low = | h + r - t | then dissimilarity can be measured as h + r - t or
            # - h - r +  t which is what is done below.
            sum_score += -abs(tail_vector[i] - head_vector[i] - relation_vector[i])
        return sum_score

    def calculate_hit_at_10(self, kgt: KnowledgeGraphTriple, verbose=False):
        """ Calculating the hits at 10 for one single test triple by replacing both the head and tail with the other
        14591 entities respectively and ranking them by disimilarity score.

        This is parallelised, one process per triple.

        :param: verbose, if True will write out the top 10 entities that replaced the correct head and tail in the
            tested triple in a text file to disk.

        Precondition: vectors saved in memory.
        """
        verbose = False
        head_id_str = kgt.head
        tail_id_str = kgt.tail
        relation_id_str = kgt.relation

        replaced_heads = []
        replaced_tails = []

        head_vector = self.word_vectors[self.word_2_index[head_id_str]]
        tail_vector = self.word_vectors[self.word_2_index[tail_id_str]]
        relation_vector = self.word_vectors[self.word_2_index[relation_id_str]]

        # Calculate hit @ 10 for this triple.

        for entity_id in self.ALL_ENTITIES:
            entity_vector_id = self.word_2_index[str(entity_id)]
            entity_vector = self.word_vectors[entity_vector_id]

            # Replaces the head entity.
            candidate_score = self.calculate_score(
                head_vector=entity_vector, tail_vector=tail_vector, relation_vector=relation_vector)
            replaced_heads.append((candidate_score, entity_id))

            # Replaces the tail entity.
            candidate_score = self.calculate_score(
                head_vector=head_vector, tail_vector=entity_vector, relation_vector=relation_vector)
            replaced_tails.append((candidate_score, entity_id))

        # These scores are negative the highest disimilarity are larger negatives.
        # sorting normally in acesending order will put the larger negatives (more dissimilar) and
        # are at the front of the array.
        # sorting in reverse order will put the largest values which have lowest dissimilarity as they are small
        # negatives at the front of the array.
        # this is equivalent of sorting normally then looping backwards like done by the authors of PTRansE.
        replaced_heads.sort(reverse=True, key=lambda t: t[0])  # 15000.
        replaced_tails.sort(reverse=True, key=lambda t: t[0])  # 15000.

        # use a heap? for faster eval method.
        # pop 10 times.

        rank_hit_head = None
        rank_hit_tail = None

        head_filt_offset = 0
        tail_filt_offset = 0

        # TODO: can break early if we only care about if it got a hit or not and not the actual rank.

        # There is a bug here somewhere.
        for index, hit_tuple in enumerate(replaced_heads):  # 15000.
            candidate_score, entity_id = hit_tuple
            # I was comparing str == int before..oops.
            if int(entity_id) == int(head_id_str):
                rank_hit_head = index  # This is our hit.
                break
            if "%s %s %s" % (int(entity_id), int(tail_id_str), int(relation_id_str)) in self.train_triple_set:
                head_filt_offset += 1


        for index, hit_tuple in enumerate(replaced_tails):  # 15000.
            candidate_score, entity_id = hit_tuple
            # I was comparing str == int before..oops.
            if int(entity_id) == int(tail_id_str):
                rank_hit_tail = index  # this is our hit.
                break
            if "%s %s %s" % (int(head_id_str), int(entity_id), int(relation_id_str)) in self.train_triple_set:
                tail_filt_offset += 1

        relation_type = self.relation_id_2_type[relation_id_str]

        if verbose:
            if not os.path.exists(self.write_out_dir):
                os.makedirs(self.write_out_dir)
            # Will print out info for this triple.
            with open(os.path.join(self.write_out_dir, "hit_@_10_" + str(kgt) + ".txt"), 'w') as fh:

                fh.write("Test triple: " + str(kgt) + "\n")  # Print actual triple.
                fh.write("Relation type: %s, %s\n" % (relation_type, self.RELATION_TYPE_ID_TO_TYPE[relation_type]))
                fh.write("Hit at 10 - head: %s, tail: %s\n" % (rank_hit_head is not None, rank_hit_tail is not None))
                fh.write("HEAD HITS\n")
                for i in range(0, 15):
                    fh.write(str(replaced_heads[i]) + "\n")
                fh.write("TAIL HITS\n")
                for i in range(0, 15):
                    fh.write(str(replaced_tails[i]) + "\n")

        return rank_hit_head, rank_hit_tail, head_filt_offset, tail_filt_offset, relation_type

    def parse_relation_types(self, n2n_file_path):
        relation_id_2_type = {}
        with open(n2n_file_path, "r") as fh:
            for relation_id, line in enumerate(fh):
                # TODO: I don't know if this is correct if x is the first thing and y is the second.
                x, y = [float(x) for x in line.split()]
                if x < 1.5:
                    if y < 1.5:

                        relation_id_2_type[str(relation_id)] = self.RELATION_TYPES['one-to-one']
                    else:
                        relation_id_2_type[str(relation_id)] = self.RELATION_TYPES['many-to-one']
                else:
                    if y < 1.5:
                        relation_id_2_type[str(relation_id)] = self.RELATION_TYPES['one-to-many']
                    else:
                        relation_id_2_type[str(relation_id)] = self.RELATION_TYPES['many-to-many']
        print("[%s parse_relation_types] n2n file: %s" % (self.tag, n2n_file_path))
        self.relation_id_2_type = relation_id_2_type

    def print_evaluation_result(self, hit_at_10_dict):
        relation_types = ['one-to-one', 'many-to-one', 'one-to-many', 'many-to-many']
        for relation_type in relation_types:
            relation_type_key = self.RELATION_TYPES[relation_type]
            print("\t%s: %s/%s = %s" % (
                relation_type,
                hit_at_10_dict[relation_type_key][0],
                hit_at_10_dict[relation_type_key][1],
                hit_at_10_dict[relation_type_key][0] / hit_at_10_dict[relation_type_key][1] if
                    hit_at_10_dict[relation_type_key][1] != 0 else "n/a"
            ))

    def evaluate(self, test_triples,
                 kg_dump_path="../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl",
                 train_data_path="../../KB2E-david-local/PTransE/data/train.txt"):
                """
                Precondition: vectors saved in memory.
                """
                if not self.deleted_model:
                    print("Precondition: vectors saved in memory")
                    return
                if not self.relation_id_2_type:
                    print("Precondition: n2n parsed and stored in self.relation_id_2_type")
                    return
                # Populate train triple set.

                with open(kg_dump_path, "rb") as fh:
                    kg = pickle.load(fh)

                with open(train_data_path, "r") as fh:
                    for line in fh:
                        h, t, r = line.split()
                        h_id = kg.entity_to_id[h]
                        t_id = kg.entity_to_id[t]
                        r_id = kg.relation_to_id[r]
                        self.train_triple_set.add("%s %s %s" % (h_id, t_id, r_id))

                start = time.time()
                # TODO: open issue on KB2E about n2n.
                # index 0 is the hit, index 1 is the total number of things.
                predicting_head_hit_at_10_raw = {
                    self.RELATION_TYPES['one-to-one']: [0, 0],
                    self.RELATION_TYPES['many-to-one']: [0, 0],
                    self.RELATION_TYPES['one-to-many']: [0, 0],
                    self.RELATION_TYPES['many-to-many']: [0, 0]
                }
                predicting_head_hit_at_10_filtered= {
                    self.RELATION_TYPES['one-to-one']: [0, 0],
                    self.RELATION_TYPES['many-to-one']: [0, 0],
                    self.RELATION_TYPES['one-to-many']: [0, 0],
                    self.RELATION_TYPES['many-to-many']: [0, 0]
                }

                predicting_tail_hit_at_10_raw = {
                    self.RELATION_TYPES['one-to-one']: [0, 0],
                    self.RELATION_TYPES['many-to-one']: [0, 0],
                    self.RELATION_TYPES['one-to-many']: [0, 0],
                    self.RELATION_TYPES['many-to-many']: [0, 0]
                }
                predicting_tail_hit_at_10_filtered = {
                    self.RELATION_TYPES['one-to-one']: [0, 0],
                    self.RELATION_TYPES['many-to-one']: [0, 0],
                    self.RELATION_TYPES['one-to-many']: [0, 0],
                    self.RELATION_TYPES['many-to-many']: [0, 0]
                }

                with multiprocessing.Pool(self.num_cores) as p:
                    result = p.starmap(self.calculate_hit_at_10, test_triples)

                mean_hit_head_raw = 0
                mean_hit_head_filt = 0
                num_head_hits = 0

                mean_hit_tail_raw = 0
                mean_hit_tail_filt = 0
                num_tail_hits = 0

                for rank_hit_head, rank_hit_tail, head_filt_offset, tail_filt_offset, relation_type in result:

                    predicting_head_hit_at_10_raw[relation_type][1] += 1
                    predicting_head_hit_at_10_filtered[relation_type][1] += 1

                    # RAW.
                    if rank_hit_head and rank_hit_head < 10:
                        predicting_head_hit_at_10_raw[relation_type][0] += 1
                    # Filtered.
                    if rank_hit_head and (rank_hit_head - head_filt_offset) < 10:
                        predicting_head_hit_at_10_filtered[relation_type][0] += 1

                    mean_hit_head_raw += rank_hit_head
                    mean_hit_head_filt += (rank_hit_head - head_filt_offset)
                    num_head_hits += 1

                    predicting_tail_hit_at_10_raw[relation_type][1] += 1
                    predicting_tail_hit_at_10_filtered[relation_type][1] += 1

                    # RAW.
                    if rank_hit_tail and rank_hit_tail < 10:
                        predicting_tail_hit_at_10_raw[relation_type][0] += 1
                    # Filtered.
                    if rank_hit_tail and (rank_hit_tail - tail_filt_offset) < 10:
                        predicting_tail_hit_at_10_filtered[relation_type][0] += 1

                    mean_hit_tail_raw += rank_hit_tail
                    mean_hit_tail_filt += (rank_hit_tail - tail_filt_offset)
                    num_tail_hits += 1

                end = time.time()

                print("Eval time mins = %s" % ((end - start)/60))


                print("~~ RAW RESULTS ~~")
                print("Mean head rank\n%s" % (mean_hit_head_raw / num_head_hits))
                print("Mean tail rank\n%s" % (mean_hit_tail_raw / num_tail_hits))
                print("Head hits")
                self.print_evaluation_result(predicting_head_hit_at_10_raw)
                print("Tail hits")
                self.print_evaluation_result(predicting_tail_hit_at_10_raw)

                print("~~ FILTERED RESULTS ~~")
                print("Mean head rank\n%s" % (mean_hit_head_filt/ num_head_hits))
                print("Mean tail rank\n%s" % (mean_hit_tail_filt / num_tail_hits))
                print("Head hits")
                self.print_evaluation_result(predicting_head_hit_at_10_filtered)
                print("Tail hits")
                self.print_evaluation_result(predicting_tail_hit_at_10_filtered)

    def print_evaluation_result(self, hit_at_10_dict):
        relation_types = ['one-to-one', 'many-to-one', 'one-to-many', 'many-to-many']
        for relation_type in relation_types:
            relation_type_key = self.RELATION_TYPES[relation_type]
            print("\t%s: %s/%s = %s\n%s" % (
                relation_type,
                hit_at_10_dict[relation_type_key][0],
                hit_at_10_dict[relation_type_key][1],
                hit_at_10_dict[relation_type_key][0] / hit_at_10_dict[relation_type_key][1] if
                hit_at_10_dict[relation_type_key][1] != 0 else "n/a",
                hit_at_10_dict[relation_type_key][0] / hit_at_10_dict[relation_type_key][1] * 100 if
                hit_at_10_dict[relation_type_key][1] != 0 else "n/a"
            ))

if __name__ == "__main__":
    from models.utils.corpus_generator import build_tagged_document_corpus_gen
    from models.utils.evaluating import parse_test_dataset_word_representation

    start = time.time()

    small_example_model = Model(
        tag="example",
        distributed_memory=True,
        epochs=1,
        learning_rate=0.01,
        dimensions=5,
        write_out_dir="./test_out"
    )

    # Things to read training data.
    DATA_DIR = "../"
    PATH_DIR = "dataset_generation"
    PATH_FILE_SUFFIX = "as_ints.txt"  # Use a v small subset for testing if i did things right or not.
    path_files = os.path.join(DATA_DIR, PATH_DIR)
    sentences = list(build_tagged_document_corpus_gen(target_dir=path_files, path_file_suffix=PATH_FILE_SUFFIX))

    small_example_model.build_vocab(sentences)
    small_example_model.train(sentences)

    small_example_model.save_vectors_in_memory_and_delete_model()

    # Trains to read testing data.
    TEST_DATA_TRIPLES = "../../KB2E-david-local/PTransE/data/test.txt"
    KG_DUMP_PATH = "../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"
    with open(KG_DUMP_PATH, "rb") as fh:
        kg = pickle.load(fh)
    test_triples = parse_test_dataset_word_representation(TEST_DATA_TRIPLES, kg)
    test_triples = test_triples[0:10]
    N2N_FILE = "../../KB2E-david-local/PTransE/data/n2n.txt"

    small_example_model.parse_relation_types(n2n_file_path=N2N_FILE)

    triple_star_map = [[t, True] for t in test_triples]
    small_example_model.evaluate(triple_star_map)

    print("OVERALL TIME TAKEN %s mins" % ((time.time() - start) / 60))
