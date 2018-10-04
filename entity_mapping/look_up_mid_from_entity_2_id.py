import time

from entity_mapping.freebase_mid_lookups import FreebaseGoogleKnowledgeGraphEntityMapper


def look_up_freebase_mids_from_entity_2_id_file(entity_2_id_file,
                                                entity_mapper: FreebaseGoogleKnowledgeGraphEntityMapper):
    start = time.time()
    count = 0
    errors = 0
    mids_missing_from_api = []
    mapped_entities = {}
    with open(entity_2_id_file, "r") as fh:
        for line in fh:
            if not line.startswith("/m"):
                continue
            try:
                freebase_mid, _ = line.split()
                result = entity_mapper.look_up_gkg_with_freebase_mid(freebase_mid)
                if "itemListElement" not in result or not result["itemListElement"]:
                    mids_missing_from_api.append(freebase_mid)
                    continue
                mapped_entities[freebase_mid] = result
            except Exception as e:
                print("Some bad stuff happened but I dun wanna crash: %s" % e)
                errors += 1
            finally:
                count += 1
                if count % 50 == 0:
                    print("[freebase -> gkg: progress] %s done, %s error" % (count, errors))

    print("Time taken %s mins" % ((time.time() - start) / 60))
    print("Total number of entities queried %s" % count)
    print("Successfully mapped %s entities, %s entities missing" %
          (count - len(mids_missing_from_api), len(mids_missing_from_api)))
    return mapped_entities, mids_missing_from_api


if __name__ == "__main__":
    import json
    import pickle
    import os

    ENTITY_2_ID_FILE = "../../KB2E-david-local/PTransE/data/entity2id.txt"
    MAPPED_ENTITIES_DUMP_DIR = "../../honours-data-round-2/FB15K/dumps/"

    with open("../apikeys_dont_commit.json", "r") as fh:
        keys = json.loads(fh.read())
    entity_id_mapper = FreebaseGoogleKnowledgeGraphEntityMapper(keys["GOOGLE_KNOWLEDGE_GRAPH"])
    mapped_entities, missing_mids = look_up_freebase_mids_from_entity_2_id_file(ENTITY_2_ID_FILE, entity_id_mapper)

    with open(os.path.join(MAPPED_ENTITIES_DUMP_DIR, "FB15K_PTransE_mapped_entities.pkl"), "wb") as fh:
        pickle.dump(mapped_entities, fh, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(MAPPED_ENTITIES_DUMP_DIR, "FB15K_PTransE_mapped_entities.txt"), "w", encoding="ascii",
              errors="ignore") as fh:
        for key, value in mapped_entities.items():
            fh.write("%s: %s\n" % (key, value))

    with open(os.path.join(MAPPED_ENTITIES_DUMP_DIR, "FB15K_PTransE_missing_entities.txt"), "w") as fh:
        for missing_mid in missing_mids:
            fh.write("%s\n" % missing_mid)
