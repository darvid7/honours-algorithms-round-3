# Converts the data from pTransE that we are using into the format that TransE expects.
import os
import pickle

from data_structures.knowledge_graph import KnowledgeGraph

OUT_DIR = ""

SUFFIX_DATA_DIR = "../../honours-data-round-2/FB15K/"

KG_DUMP_PATH = "dumps/FB15K_PTransE_dump.pkl"
with open(os.path.join(SUFFIX_DATA_DIR, KG_DUMP_PATH), "rb") as fh:
    knowledge_graph = pickle.load(fh)


def convert_relations(in_file, out_file):
    """
    TODO: does the '.' in the relation mean anything significant?
    pTransE relation2id.txt file expects:
        /people/appointed_role/appointment./people/appointment/appointed_by	0

    TransE relation2id.txt file expects:
        1345
        /location/country/form_of_government	0

    Difference:
        TransE datafile expects the count.
    """
    with open(in_file, 'r') as fh:
        lines = fh.readlines()
    with open(out_file, 'w') as fh:
        fh.write("%s\n" % len(lines))
        fh.writelines(lines)
    print("Finished %s" % out_file)


def convert_entities(in_file, out_file):
    """
     pTransE entity2id.txt file expects:
        /m/06rf7	0

    TransE entity2id.txt file expects:
        14951
        /m/027rn	0

    Difference:
        TransE datafile expects the count.
    """
    with open(in_file, 'r') as fh:
        lines = fh.readlines()
    with open(out_file, 'w') as fh:
        fh.write("%s\n" % len(lines))
        fh.writelines(lines)
    print("Finished %s" % out_file)


def convert_training_set(kg: KnowledgeGraph, in_file, out_file):
    """
     pTransE train.txt file expects:
        /m/027rn	/m/06cx9	/location/country/form_of_government

    TransE train2id.txt file expects:
        483142
        0 1 0
        2 3 1

    Difference:
        TransE datafile expects the count and represents tuples as ids.
        e1 e2 rel
    """
    with open(in_file, 'r') as fh:
        lines = fh.readlines()
    with open(out_file, 'w') as fh:
        fh.write("%s\n" % len(lines))
        for line in lines:
            line = line.split()
            e1_mid, e2_mid, relation = line
            e1_id = kg.entity_to_id[e1_mid]
            e2_id = kg.entity_to_id[e2_mid]
            rel_id = kg.relation_to_id[relation]
            fh.write("%s %s %s\n" % (e1_id, e2_id, rel_id))
    print("Finished %s" % out_file)

def convert_validation_set(kg: KnowledgeGraph, in_file, out_file):
    """
     pTransE valid.txt file expects:
        /m/027rn	/m/06cx9	/location/country/form_of_government

    TransE valid2id.txt file expects:
        50000
        5167 1427 52

    Difference:
        TransE datafile expects the count and represents tuples as ids.
        e1 e2 rel
    """
    with open(in_file, 'r') as fh:
        lines = fh.readlines()
    with open(out_file, 'w') as fh:
        fh.write("%s\n" % len(lines))
        for line in lines:
            line = line.split()
            e1_mid, e2_mid, relation = line
            e1_id = kg.entity_to_id[e1_mid]
            e2_id = kg.entity_to_id[e2_mid]
            rel_id = kg.relation_to_id[relation]
            fh.write("%s %s %s\n" % (e1_id, e2_id, rel_id))
    print("Finished %s" % out_file)


def convert_testing_set(kg: KnowledgeGraph, in_file, out_file):
    """
     pTransE test.txt file expects:
        /m/01qscs	/m/02x8n1n	/award/award_nominee/award_nominations./award/award_nomination/award

    TransE test2id.txt file expects:
        59071
        453 1347 37
    TransE test2id_all.txt file expects:
        59071
        3	453 1347 37

    Difference:
        TransE datafile expects the count and represents tuples as ids as well as the relation type.
        relation_type e2 e2 rel
    """
    with open(in_file, 'r') as fh:
        lines = fh.readlines()
    with open(out_file, 'w') as fh:
        fh.write("%s\n" % len(lines))
        for line in lines:
            line = line.split()
            e1_mid, e2_mid, relation = line
            e1_id = kg.entity_to_id[e1_mid]
            e2_id = kg.entity_to_id[e2_mid]
            rel_id = kg.relation_to_id[relation]
            fh.write("%s %s %s\n" % (e1_id, e2_id, rel_id))
    print("Finished %s" % out_file)


if __name__ == "__main__":
    p_trans_e_data_dir = "../../KB2E-david-local/PTransE/data"
    out_data_dir = "."
    convert_relations(
        in_file=os.path.join(p_trans_e_data_dir, "relation2id.txt"),
        out_file=os.path.join(out_data_dir, "relation2id.txt")
    )
    convert_entities(
        in_file=os.path.join(p_trans_e_data_dir, "entity2id.txt"),
        out_file=os.path.join(out_data_dir, "entity2id.txt")
    )

    convert_training_set(
        kg=knowledge_graph,
        in_file=os.path.join(p_trans_e_data_dir, "train.txt"),
        out_file=os.path.join(out_data_dir, "train2id.txt")
    )

    convert_validation_set(
        kg=knowledge_graph,
        in_file=os.path.join(p_trans_e_data_dir, "valid.txt"),
        out_file=os.path.join(out_data_dir, "valid2id.txt")
    )

    convert_testing_set(
        kg=knowledge_graph,
        in_file=os.path.join(p_trans_e_data_dir, "test.txt"),
        out_file=os.path.join(out_data_dir, "test2id.txt")
    )



