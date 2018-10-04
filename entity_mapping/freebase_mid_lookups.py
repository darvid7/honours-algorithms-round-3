""" The purpose of this class is to map Freebase MIDs to Google Knowledge Graph Entities without needing to download
the 200gb data dumps.

According to various sources the Google Knowledge Graph (GKG) consumes parts of Freebase and has maintained the MIDs of
entities. (1) shows how you can query the Google Knowledge Graph for a Freebase MID.

Sources:
(1) https://plus.google.com/106943062990152739506/posts/SHniitXKodd
(2) https://searchengineland.com/laymans-visual-guide-googles-knowledge-graph-search-api-241935

So far I have found two ways to query the GKG.
1. via the GKG API
2. directly using (1)
Note, not all MIDs in FB15k are queryable via the API, some that are missing from the API can be mapped directly using
the url http://g.co/kg/m/**** (1).

Run time:
- errrrrrrr oops, this should take around 5-10 hours assuming 50 queries per minute for 15000 entities.
"""
import json
import requests

from typing import Dict, Union
from bs4 import BeautifulSoup


class FreebaseGoogleKnowledgeGraphEntityMapper:

    # Primarily use this.
    GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT = "https://kgsearch.googleapis.com/v1/entities:search?ids={0}&key={1}"

    # Only use if the API endpoint fails to return useful info, need to parse HTML get response.
    GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT_AUXILIARY = "http://g.co/kg%s"  # MID of form /m/...

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    # Methods using the API, I assumed that the API could map everything, boy was I wrong.
    def look_up_gkg_with_freebase_mid(self, freebase_mid: str) -> Dict:
        """ Looks up real world value of a freebase MID in GKG and saves that in self.mappings. """
        query = self.GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT.format(freebase_mid, self.api_key)
        response = requests.get(query)
        return json.loads(response.text)

    # This can violate robots.txt I think.
    # When calling add some randomness and wait so Google doesn't think I am a robot.
    def auxiliary_look_up_gkg_non_api_with_freebase_mid(self, freebase_mid: str,
                                                        bad_response_html_dump_path=None) -> Union[str, bool, Dict]:
        """ Looks GKG entity up using method (1). If a bad response is returned will dump the response to a file
        if bad_response_html_dump_path is provided. """
        AWKS_ROBOT_STOP_SPAMMING = "Our systems have detected unusual traffic from your computer network."
        response = requests.get(self.GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT_AUXILIARY % freebase_mid)
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        # findAll returns [<class 'bs4.element.Tag'>, ..], can get contents using .contents!
        name = soup.findAll("div", {"class": "FSP1Dd"})
        super_group_topic = soup.findAll("div", {"class": "F7uZG Rlw09"})
        description = soup.findAll("div", {"class": "mraOPb"})
        if len(name) > 1 or len(super_group_topic) > 1 or len(description) > 1:
            print(
                "Got multiple results for %s, taking index 0. "
                "Num results for name: %s, super_group_topic: %s, description: %s") % (
                len(name), len(super_group_topic), len(description))

        if not name:
            print("MID %s has no name result, must have been a bad search, url: %s" %
                  (freebase_mid, self.GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT_AUXILIARY % freebase_mid))
            if bad_response_html_dump_path:
                with open(bad_response_html_dump_path, "w") as html_dump:
                    html_dump.write(html)
                print("dumped bad mid %s html to %s" % (freebase_mid, bad_response_html_dump_path))
            if AWKS_ROBOT_STOP_SPAMMING in html:
                return "ROBOT_DETECTED"
            return False
        name = name[0].contents  # Weirdly returns it ina list.
        name = name[0]  # Get rid of the list.
        if super_group_topic:
            super_group_topic = super_group_topic[0].contents  # Weirdly returns it in a list.
            super_group_topic = super_group_topic[0]
        if description:
            description = description[0].contents  # Get rid of outer div.
            description = description[0].contents  # Get rid of span.
            description = description[0].strip()  # Get rid of wiki link that needs to be parsed, its now a string!

        print("mid %s aux result %s, url: %s" % (
            freebase_mid, name, self.GOOGLE_KNOWLEDGE_GRAPH_ENDPOINT_AUXILIARY % freebase_mid))

        gkg_entity_description = {
            "name": name,
            "source": "aux",
            "meta": {
                "description": description if description else None,
                "super_group_topic": super_group_topic if super_group_topic else None
            }
        }

        # print out the data structure so can eval() it incase the process dies it will still be in the log file.
        temp = {freebase_mid: gkg_entity_description}
        print(temp)

        return gkg_entity_description


if __name__ == "__main__":
    with open("../apikeys_dont_commit.json", "r") as fh:
        keys = json.loads(fh.read())
    entity_id_mapper = FreebaseGoogleKnowledgeGraphEntityMapper(keys["GOOGLE_KNOWLEDGE_GRAPH"])

    # API response, looks up USA.
    api_response = entity_id_mapper.look_up_gkg_with_freebase_mid("/m/09c7w0")
    print(api_response)

    # API response, missing data.
    missing_data = entity_id_mapper.look_up_gkg_with_freebase_mid("/m/02dsz1")
    print(missing_data)

    # Direct response, looks up lounge music.
    direct_response = entity_id_mapper.auxiliary_look_up_gkg_non_api_with_freebase_mid("/m/02dsz1")
    print(direct_response)