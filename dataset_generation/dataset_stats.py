import os
import sys

from collections import OrderedDict
from typing import List

from scipy import stats
import numpy as np


def parse(path_dump_file, num_entities, num_relations):
    # Ordered dict will maintain insert order.
    relations = OrderedDict({i: 0 for i in range(num_relations)})
    entities = OrderedDict({i: 0 for i in range(num_entities)})

    with open(path_dump_file, "r") as fh:
        for line in fh:
            line = eval(line)
            for index, int_id in enumerate(line):
                if index % 2 == 0:  # Is an entity.
                    entities[int_id] += 1
                else:  # Is a relation.
                    relations[int_id] += 1
    return entities, relations


def calculate_stats(dataset: List[int]):
    """ Returns some important stats we might care about our dataset.

    :param dataset: a List of integers representing frequencies.
    :return:
        mean: the average frequency
        median: the middle frequency
        mode: the most common frequency
        std_dev: a measure of dispersion (also sqrt(variance))
        variance: a measure of dispersion (also std^2).
    """
    mean = np.mean(dataset)
    median = np.median(dataset)
    mode = stats.mode(dataset)  # scipy.stats is weird, returns ModeResult(mode=array([6]), count=array([1361]))
    mode = mode.mode[0]
    std_dev = np.std(dataset)
    variance = np.var(dataset)
    return mean, median, mode, std_dev, variance


def pretty_summary_print(title, mean, median, mode, std_dev, variance, file):
    print("%s" % title, file=file)
    print("Mean: %s" % mean, file=file)
    print("Median: %s" % median, file=file)
    print("Mode: %s" % mode, file=file)
    print("Standard deviation: %s" % std_dev, file=file)
    print("Variance: %s" % variance, file=file)


def count_frequencies(frequencies):
    frequency_occurrences = OrderedDict()
    frequencies.sort()
    for f in frequencies:
        if f in frequency_occurrences:
            frequency_occurrences[f] += 1
        else:
            frequency_occurrences[f] = 1
    return frequency_occurrences


def add_to_accum(accum, other):
    for key in other.keys():
        if key not in accum:
            accum[key] = 0
        accum[key] += other[key]


def main():
    PATH_DIR = "../../honours-data-round-2/FB15K/paths_random_test"
    NUM_ENTITIES = 14951
    NUM_RELATIONS = 1345

    accum_entity_counts = {}
    accum_relation_counts = {}

    paths = os.listdir(PATH_DIR)
    for path_file in paths:
        if not path_file.endswith(".txt") or path_file.endswith("stats.txt") or path_file.endswith("freq.txt"):
            continue
        # Format: max_entities_on_path_4_paths.txt
        num_entities, _ = path_file.rsplit("_paths")
        _, num_entities = num_entities.rsplit("_", 1)
        # Maps either an entity or relation id to it's count (occurrences in a path file).
        entity_counts, relation_counts = parse(os.path.join(PATH_DIR, path_file),
                                               num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS)

        add_to_accum(accum=accum_entity_counts, other=entity_counts)
        add_to_accum(accum=accum_relation_counts, other=relation_counts)

        entity_dump_path = os.path.join(PATH_DIR, "max_entities_on_path_%s_entity_stats.txt" % num_entities)
        entity_freq_dump_path = os.path.join(PATH_DIR, "max_entities_on_path_%s_entity_stats_freq.txt" % num_entities)

        relation_dump_path = os.path.join(PATH_DIR, "max_entities_on_path_%s_relation_stats.txt" % num_entities)
        relation_freq_dump_path = os.path.join(PATH_DIR, "max_entities_on_path_%s_relation_stats_freq.txt" % num_entities)

        # Process entities.
        with open(entity_dump_path, "w") as fh:

            print("For paths of length %s (%s entities)" % (int(num_entities) * 2 - 1, num_entities), file=fh)
            e_mean, e_median, e_mode, e_std_dev, e_variance = calculate_stats(list(entity_counts.values()))
            pretty_summary_print("Entity stats", e_mean, e_median, e_mode, e_std_dev, e_variance, file=fh)

            entity_counts_sorted = list(entity_counts.items())
            entity_counts_sorted.sort(key=lambda t: t[1], reverse=True)  # Most frequent ids at the start.
            entity_frequency_occurrences = count_frequencies([t[1] for t in entity_counts_sorted])

            fh.write("Frequency of occurrences\n")
            for times in entity_frequency_occurrences:
                id_count = entity_frequency_occurrences[times]
                fh.write("%s entities occurred %s times\n" % (id_count, times))

        with open(entity_freq_dump_path, "w") as fh:
            fh.write("Sorted by highest frequency\n")
            fh.write("Entity id, count\n")
            for entity_id, occurrence in entity_counts_sorted:
                fh.write("%s, %s\n" % (entity_id, occurrence))

        # Process relations.
        with open(relation_dump_path, "w") as fh:

            print("For paths of length %s (%s entities)" % (int(num_entities) * 2 - 1, num_entities), file=fh)
            r_mean, r_median, r_mode, r_std_dev, r_variance = calculate_stats(list(relation_counts.values()))
            pretty_summary_print("Relation stats", r_mean, r_median, r_mode, r_std_dev, r_variance, file=fh)

            relation_counts_sorted = list(relation_counts.items())
            relation_counts_sorted.sort(key=lambda t: t[1], reverse=True)  # Most frequent ids at the start.
            relation_frequency_occurrences = count_frequencies([t[1] for t in relation_counts_sorted])

            fh.write("Frequency of occurrences\n")
            for times in relation_frequency_occurrences:
                id_count = relation_frequency_occurrences[times]
                fh.write("%s relations occurred %s times\n" % (id_count, times))

        with open(relation_freq_dump_path, "w")  as fh:
            fh.write("Sorted by highest frequency\n")
            fh.write("Relation id, count\n")
            for relation_id, occurrence in relation_counts_sorted:
                fh.write("%s, %s\n" % (relation_id, occurrence))
        print("Processed %s" % path_file)

    total_entity_freq_dump = os.path.join(PATH_DIR, "total_entities_counts_stats.txt")
    all_entity_counts = list(accum_entity_counts.items())
    all_entity_counts.sort(key=lambda t: t[1], reverse=True)  # Sort by highest freq.
    with open(total_entity_freq_dump, "w") as fh:
        fh.write("Entity id, count\n")
        for entity_id, count in all_entity_counts:
            fh.write("%s, %s\n" % (entity_id, count))
    print("Wrote out total entities %s" % total_entity_freq_dump)

    mean, median, mode, std_dev, variance = calculate_stats(list(accum_entity_counts.values()))
    pretty_summary_print("total entity stats", mean, median, mode, std_dev, variance, file=sys.stdout)

    total_relation_freq_dump = os.path.join(PATH_DIR, "total_relation_counts_stats.txt")
    all_relation_counts = list(accum_relation_counts.items())
    all_relation_counts.sort(key=lambda t: t[1], reverse=True)  # Sort by highest freq.
    with open(total_relation_freq_dump, "w") as fh:
        fh.write("Relation id, count\n")
        for relation_id, count in all_relation_counts:
            fh.write("%s, %s\n" % (relation_id, count))
    print("Wrote out total relations %s" % total_relation_freq_dump)

    mean, median, mode, std_dev, variance = calculate_stats(list(accum_relation_counts.values()))
    pretty_summary_print("total relation stats", mean, median, mode, std_dev, variance, file=sys.stdout)


if __name__ == "__main__":

    main()