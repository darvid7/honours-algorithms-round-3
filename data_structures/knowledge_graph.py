from typing import Dict


class KnowledgeGraphTriple:
    def __init__(self, head, relation, tail):
        self.head = head
        self.relation = relation
        self.tail = tail

    def __str__(self):
        return "(h: %s, r: %s, t: %s)" % (self.head, self.relation, self.tail)

    def __eq__(self, other):  # Use so it can be compared to other elements in a set.
        return self.head == other.head and self.relation == other.relation and self.tail == other.tail

    def __hash__(self):  # So it can be placed into a set.
        return hash((self.head, self.relation, self.tail))


class AdjacencyListNode:
    def __init__(self, entity):
        self.entity = entity
        self.outgoing_edges = set()

    def __str__(self):
        return "%s : %s" % (self.entity, [str(e) for e in self.outgoing_edges])

    def get_outgoing_edges_as_list(self):
        return [s for s in self.outgoing_edges]


class AdjacencyListEdge:
    def __init__(self, relation, tail):
        self.relation = relation
        self.tail = tail

    def __str__(self):
        """ Returns this edge in the format (relation, tail). """
        return "(%s, %s)" % (self.relation, self.tail)

    def __eq__(self, other):  # Use so it can be compared to other elements in a set.
        return self.relation == other.relation and self.tail == other.tail

    def __hash__(self):  # So it can be placed into a set.
        return hash((self.relation, self.tail))

    def __lt__(self, other):
        if self.relation < other.relation:
            return True
        if other.relation < self.relation:
            return False
        # relations are equal
        if self.tail < other.tail:
            return True
        return False


class KnowledgeGraph:
    """ Represents the knowledge graph form a file of triples as an adjacency list.

    Nodes are represented as indexes in the adjacency list from [0, num_entities -1] inclusive.
    Edges can be accessed via self[node_index].outgoing_edges or self[node_index].get_outgoing_edges_as_list.
    Handles parsing from triple files where triples are represented as an id or as words.

    Args:
        self.adj_list: adjacency list of AdjacencyListNode
        self.entity_to_id: used to look up an entities integer id representation.
        self.relation_to_id: used to look up a relations integer id representation.
    """
    def __init__(self, num_entities, entity_2_id_file, relation_2_id_file):
        self.adj_list = [AdjacencyListNode(e) for e in range(num_entities)]
        # *_to_id map a word to an integer id.
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.parse_word_representation_to_id_map(entity_2_id_file, self.entity_to_id)
        self.parse_word_representation_to_id_map(relation_2_id_file, self.relation_to_id)

    def __getitem__(self, item):
        return self.adj_list[item]

    def _print_if_verbose(self, verbose, msg):
        if verbose:
            print(msg)

    def get_nodes(self):
        return self.adj_list

    def remove_self_loops(self, verbose=True):
        """ Removes a node looping back to itself through one of it's direct outgoing edges. """
        self_loops_removed = 0
        total_num_edges = 0
        for node in self.get_nodes():
            entity_id = node.entity
            self_loops = set()
            for outgoing_edge in node.outgoing_edges:
                total_num_edges += 1
                if outgoing_edge.tail == entity_id:  # Loops back to itself.
                    self_loops.add(outgoing_edge)
            # Remove self loops from outgoing edges.
            for looping_edge in self_loops:
                self._print_if_verbose(
                    verbose, "[removing self loops] %s loops back to itself with %s" % (entity_id, looping_edge))
                node.outgoing_edges.remove(looping_edge)
                self_loops_removed += 1
        print("Total Num Edges: %s" % total_num_edges)
        print("Removed %s self loops" % self_loops_removed)
        print("Non self looping edges: %s" % (total_num_edges - self_loops_removed))

    def parse_word_representation_to_id_map(self, word_rep_2_id_file, word_rep_to_id_dict):
        """ Uses entity_2_id/relation_2_id files to read in the mappings. """
        with open(word_rep_2_id_file, "r") as fh:
            for line in fh:
                word_rep, int_id_rep = line.strip().split()
                word_rep_to_id_dict[word_rep] = int(int_id_rep)

    def add_edge(self, head_entity, relation, tail_entity):
        """ Adds an edge handling if the head entity, relation or tail entity need to be converted
         into it's integer id or not. """
        # Handle digits (int ids) read in from file.
        if head_entity.isdigit():
            head_entity = int(head_entity)
        if relation.isdigit():
            relation = int(relation)
        if tail_entity.isdigit():
            tail_entity = int(tail_entity)
        # Handles word representation read in from file.
        if not isinstance(head_entity, int):
            head_entity = self.entity_to_id[head_entity]
        if not isinstance(relation, int):
            relation = self.relation_to_id[relation]
        if not isinstance(tail_entity, int):
            tail_entity = self.entity_to_id[tail_entity]
        # relation adn tail_entity are now integers.
        edge = AdjacencyListEdge(relation=relation, tail=tail_entity)
        if edge in self[head_entity].outgoing_edges:
            print("edge: %s already in node %s edge_set, %s" % (
                edge, head_entity, self[head_entity].outgoing_edges))
            return False
        else:
            self.adj_list[head_entity].outgoing_edges.add(edge)
            return True

    def parse_triples_from_file(self, file_path, head_index=0, relation_index=1, tail_index=2):
        """ Expects triples to be a series of integer ids, eg: 1231 98 19. """
        print("Parsing triples from %s" % file_path)
        triple_count = 0
        with open(file_path, "r") as fh:
            for line in fh:
                line = line.split()
                if len(line) != 3:
                    continue
                # Models a directed graph, a one way relationship from the head entity to the tail entity via the
                # relation r, h -r-> t.
                if self.add_edge(
                        head_entity=line[head_index],
                        relation=line[relation_index],
                        tail_entity=line[tail_index]):
                    triple_count += 1
                # TODO: Add undirected option to consider the reverse relation if needed.
        print("parsed %s triples from %s" % (triple_count, file_path))


class KnowledgeGraphEntityRepresentation:
    """ Class to represent an entity and it's associated meta data.

    I think it was necessary to put it in its own class because there needs to be a common interface between different
    knowledge graphs. Even within a single knowledge graph such as GKG there are two sources I am using to get entity
    data that return different structured information.

    Attributes:
        self.word_representation: set at initialization, compulsory.
        self.source: set at initialization, compulsory
        self._meta_data: set using methods.
    """

    def __init__(self, word_representation, source):
        self.word_representation = word_representation
        self.source = source
        self._meta_data = {}

    def set_meta_data(self, meta_data: Dict):
        meta_data_keys = ["score", "description", "detailed_description", "type_list", "gkg_id", "super_group_topic"]
        # TODO: Type list is very similar to super_group_topic I think, might map them together later.
        for key in meta_data_keys:
            if key in meta_data:
                self._meta_data[key] = meta_data[key]
            else:
                self._meta_data[key] = None

    def __str__(self):
        d = str({"word_representation": self.word_representation, "source": self.source, "meta_data": self._meta_data})
        return d

    def as_dict(self):
        return {"word_representation": self.word_representation, "source": self.source, "meta_data": self._meta_data}



