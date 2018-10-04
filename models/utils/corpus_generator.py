import os
import sys

from gensim.models.doc2vec import TaggedDocument

# print("~~ Generating data streaming from disk!! hopefully this stuff won't run out of memory ~~")
# print("PID %s" % os.getpid())
# sys.stdout.flush()

# Should not take upp all the ram.

TOTAL_NUM_ENTITIES = 14951


def get_files_gen(target_dir, path_file_suffix):
    dirs = os.listdir(target_dir)
    for file_name in dirs:
        if file_name.endswith(path_file_suffix):
            yield os.path.join(target_dir, file_name)


def build_tagged_document_corpus_gen(target_dir, path_file_suffix):
    # Tag: This has to be unique for each path. Guaranteed to be because path_processed_count will be unique and
    # will be used as the path id.
    # {origin_node/head_node}_{path_id}_{count_tail_node_id}

    tag_template = "%s_%s_%s"
    path_processed_count = 0
    start_path_num = 0
    for full_path in get_files_gen(target_dir, path_file_suffix):  # Out generator.
        with open(full_path, "r") as fh:
            for line in fh:  # Inner generator.
                line = eval(line)  # List of integers.
                for i in range(len(line)):
                    if not i % 2 == 0:  # not an entity.
                        # Need to offset the relation id so they are unique.
                        relation_id = line[i] + TOTAL_NUM_ENTITIES
                        line[i] = relation_id
                # THE TAG SHOULD DEFINITELY NOT BE INDEX FML.
                head_id = line[0]
                tail_id = line[-1]  # is a string.
                sentence = TaggedDocument([str(c) for c in line], [tag_template %
                                                                   (head_id, path_processed_count, tail_id)])
                path_processed_count += 1
                yield sentence
        print("Processed %s - %s paths in file %s" % (start_path_num, path_processed_count, full_path))
        start_path_num = path_processed_count


if __name__ == "__main__":

    DATA_DIR = "../../../honours-data-round-2/FB15K/"
    PATH_DIR = "paths_random_test"
    PATH_FILE_SUFFIX = "paths.txt"

    target = os.path.join(DATA_DIR, PATH_DIR)
    for path in build_tagged_document_corpus_gen(target_dir=target, path_file_suffix=PATH_FILE_SUFFIX):
        pass
