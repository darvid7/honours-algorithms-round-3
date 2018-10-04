import time
import random
import time
import sys
import traceback

from typing import Tuple, Dict, List

from entity_mapping.freebase_mid_lookups import FreebaseGoogleKnowledgeGraphEntityMapper
from data_structures.knowledge_graph import KnowledgeGraphEntityRepresentation

PAUSE_SECONDS = 60 * 10


def look_up_freebase_mid_missing_in_api_call(
        missing_mid_file,
        robot_response_dir,
        entity_mapper: FreebaseGoogleKnowledgeGraphEntityMapper,
        fast=False) -> Tuple[Dict[str, KnowledgeGraphEntityRepresentation], List[str]]:
    found_robot = False
    successful_fixes = {}
    robot_queries = []
    start_time = time.time()

    try:
        with open(missing_mid_file, "r") as missing_mid_fh:  # Don't write here.
            for line in missing_mid_fh:
                if found_robot:
                    # This process/ip has been detected to be a robot, this will expire shortly after requests stop.
                    # Will increase the wait time by a decent amount.
                    # 10 mins + rand % of 5 mins * count
                    wait_time = int(PAUSE_SECONDS + 5 * random.random() * len(robot_queries))
                    print("Was identified as a robot, waiting %s seconds or %s mins" % (wait_time, wait_time / 60))
                    time.sleep(wait_time)
                    # Assume it is not identified as a robot anymore and try again.
                    found_robot = False
                elif not fast:
                    # Wait between 10 to 45 seconds before making requests.
                    random_time_seconds_to_wait = random.randint(10, 45)
                    time.sleep(random_time_seconds_to_wait)  # Buffer requests to be slower.
                missing_mid = line.strip()

                # Try to fix this mid again.
                fix_attempt = entity_mapper.auxiliary_look_up_gkg_non_api_with_freebase_mid(
                    freebase_mid=missing_mid,
                    bad_response_html_dump_path=os.path.join(robot_response_dir,
                                                             "%s_bad_response.html" % missing_mid.replace('/', '-')))
                if fix_attempt:
                    if fix_attempt == "ROBOT_DETECTED":
                        found_robot = True
                        robot_queries.append(missing_mid)
                        print("%s attempt to fix query was identified as a robot" % line)
                        continue  # Do not alter the converted list.

                    entity_rep = KnowledgeGraphEntityRepresentation(fix_attempt["name"], source="aux")
                    entity_rep.set_meta_data(fix_attempt["meta"])
                    successful_fixes[missing_mid] = entity_rep

    except Exception as e:
        print("WTF BAD THING HAPPENED: %s" % e)
        traceback.print_exc()
    finally:
        print("Attempted to fix %s missing MIDs unmapped from the API query" %
              (len(successful_fixes) + len(robot_queries)))
        print("Number of missing unmapped MIDs from the API fixed: %s" % len(successful_fixes))
        print("Number of robot queries: %s" % len(robot_queries))
        print("Time taken %s mins" % ((time.time() - start_time) // 60))
        return successful_fixes, robot_queries


if __name__ == "__main__":
    import os
    import pickle
    from data_structures.knowledge_graph import KnowledgeGraphEntityRepresentation

    MISSING_MID_FILE_PATH = "../../honours-data-round-2/FB15K/entity_lookup/FB15K_PTransE_missing_entities.txt"
    ROBOT_DUMP_DIR = "../../honours-data-round-2/FB15K/entity_lookup/robot_dumps"
    OUTPUT_DUMP_DIR = "../../honours-data-round-2/FB15K/entity_lookup"

    dirs = [ROBOT_DUMP_DIR, OUTPUT_DUMP_DIR]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    entity_mapper = FreebaseGoogleKnowledgeGraphEntityMapper()

    successful_fixes_dict, robot_queries = look_up_freebase_mid_missing_in_api_call(
        missing_mid_file=MISSING_MID_FILE_PATH,
        robot_response_dir=ROBOT_DUMP_DIR,
        entity_mapper=entity_mapper, fast=True)

    print("Writing out successful fixes dict and robot queries")

    x = {k: v.as_dict() for k, v in successful_fixes_dict.items()}

    with open(os.path.join(OUTPUT_DUMP_DIR, "PTransE_mapped_missing_mids.txt"), "w") as fh:
        for key, value in successful_fixes_dict.items():
            fh.write("%s: %s\n" % (key, value))

    # TODO: rly hacky but I can't get x to pickle.
    with open(os.path.join(OUTPUT_DUMP_DIR, "PTransE_mapped_missing_mids.pkl"), "wb") as fh:
        pickle.dump(eval(str(x)), fh, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(OUTPUT_DUMP_DIR, "PTransE_robot_queries.txt"), "w") as fh:
        fh.writelines([str(s) + "\n" for s in robot_queries])
