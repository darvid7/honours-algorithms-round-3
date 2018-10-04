import os
import time
import pickle

import models.base_model
from models.utils.corpus_generator import build_tagged_document_corpus_gen
from models.utils.evaluating import parse_test_dataset_word_representation

start = time.time()

pte_dm = models.base_model.Model(
    tag="PTransE hyperparams Paragraph Vector - Distributed Memory",
    distributed_memory=True,
    epochs=500,
    learning_rate=0.001,
    dimensions=100,
    write_out_dir="../../../honours-data-round-2/FB15K/model_out/evaluation/pte_pv_dm"
)

# Things to read training data.
DATA_DIR = "../../"
PATH_DIR = "dataset_generation"
PATH_FILE_SUFFIX = "as_ints.txt"  # Use a v small subset for testing if i did things right or not.
path_files = os.path.join(DATA_DIR, PATH_DIR)
sentences = list(build_tagged_document_corpus_gen(target_dir=path_files, path_file_suffix=PATH_FILE_SUFFIX))

pte_dm.build_vocab(sentences)
pte_dm.train(sentences)

pte_dm.save_vectors_in_memory_and_delete_model()

# Trains to read testing data.
TEST_DATA_TRIPLES = "../../../KB2E-david-local/PTransE/data/test.txt"
KG_DUMP_PATH = "../../../honours-data-round-2/FB15K/dumps/FB15K_PTransE_dump.pkl"
with open(KG_DUMP_PATH, "rb") as fh:
    kg = pickle.load(fh)
test_triples = parse_test_dataset_word_representation(TEST_DATA_TRIPLES, kg)
N2N_FILE = "../../../KB2E-david-local/PTransE/data/n2n.txt"

pte_dm.parse_relation_types(n2n_file_path=N2N_FILE)

triple_star_map = [[t, True] for t in test_triples]
pte_dm.evaluate(triple_star_map)

print("OVERALL TIME TAKEN %s mins" % ((time.time() - start) / 60))
