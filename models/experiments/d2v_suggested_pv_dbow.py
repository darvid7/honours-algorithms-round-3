import os
import time
import pickle

import models.base_model
from models.utils.corpus_generator import build_tagged_document_corpus_gen
from models.utils.evaluating import parse_test_dataset_word_representation

start = time.time()

d2v_dbow = models.base_model.Model(
    tag="Doc2Vec suggested hyperparams Paragraph Vector - Distributed Bag of Words",
    distributed_memory=False,
    epochs=20,
    learning_rate=0.01,
    dimensions=100,
    write_out_dir="../../../honours-data-round-2/FB15K/model_out/evaluation/d2v_suggested_pv_dbow"
)

# Things to read training data.
DATA_DIR = "../../"
PATH_DIR = "dataset_generation"
PATH_FILE_SUFFIX = "as_ints.txt"  # Use a v small subset for testing if i did things right or not.
path_files = os.path.join(DATA_DIR, PATH_DIR)
sentences = list(build_tagged_document_corpus_gen(target_dir=path_files, path_file_suffix=PATH_FILE_SUFFIX))

d2v_dbow.build_vocab(sentences)
d2v_dbow.train(sentences)

d2v_dbow.save_vectors_in_memory_and_delete_model()

# Trains to read testing data.
TEST_DATA_TRIPLES = "../../../KB2E-david-local/PTransE/data/test.txt"
KG_DUMP_PATH = "../../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"
with open(KG_DUMP_PATH, "rb") as fh:
    kg = pickle.load(fh)
test_triples = parse_test_dataset_word_representation(TEST_DATA_TRIPLES, kg)
N2N_FILE = "../../../KB2E-david-local/PTransE/data/n2n.txt"

d2v_dbow.parse_relation_types(n2n_file_path=N2N_FILE)

triple_star_map = [[t, True] for t in test_triples]
d2v_dbow.evaluate(triple_star_map,
                  kg_dump_path="../../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl",
                  train_data_path="../../../KB2E-david-local/PTransE/data/train.txt")

print("OVERALL TIME TAKEN %s mins" % ((time.time() - start) / 60))
