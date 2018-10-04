import numpy as np
import os

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size=1, method='skip_gram'):
    # We want a window of 1 such that it is entity-relation-entity as our sentence.
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((index - window_size), 0):(index + window_size + 1)] for index, word in
                            enumerate(rand_sentence)]

        # window_sequences not used in doc2vec.
        # window_sequences = []

        # for i in range(len(rand_sentence)):
        #     if i % 2 == 0:
        #         # is entity, skip.
        #         pass
        #     else:
        #         # is relation, append ['e', 'r', 'e']
        #         window_sequences.append(rand_sentence[i - 1: i + 1 + 1])
        #
        # print("window sequences")
        # print(window_sequences)

        # label_indices not used for doc2vec.
        # Denote which element of each window is the center word of interest
        label_indices = [index if index < window_size else window_size for index, word in enumerate(window_sequences)]
        # print(label_indices)

        # Pull out center word of interest for each window and create a tuple for each window
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec':
            # i think e1 + e1 should predict r. or e1 + r should predict e2.
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [
                (rand_sentence[i:i + window_size], rand_sentence[i + window_size]) for i in
                range(0, len(rand_sentence) - window_size)
            ]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]

        else:
            raise ValueError('Method {} not implmented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    # print("Batch data")
    # print(batch_data)
    # print("label data")
    # print(label_data)
    return (batch_data, label_data)

def load_fb15k_shared_model_data():
    """
    Needs to return a list of numbers that represent paths through the knowledge graph in the format
        entity -> relation -> entity with their relations offset.

    :return:
    """
    num_entities = 14951

    path_files = [
        "max_entities_on_path_4_paths.txt",
        "max_entities_on_path_5_paths.txt",
        "max_entities_on_path_6_paths.txt"
    ]
    paths = []
    path_files_dir = "created_paths"
    data_dir = "../../../hons-data-trans-e-x-format"
    triple_file = "train2id.txt"
    # with open(os.path.join(data_dir, triple_file), 'r') as fh:
    #     for line in fh:
    #         line = line.split()
    #         if len(line) <= 1:
    #             continue
    #         line = [int(x) for x in line]
    #         h, t, r = line
    #         r = r + num_entities
    #         paths.append([h, r, t])

    for path_file in path_files:
        with open(os.path.join(data_dir, path_files_dir, path_file), 'r') as fh:
            for line in fh:
                line = eval(line)
                for i in range(len(line)):
                    if i % 2 == 0:
                        # is entity.
                        pass
                    else:
                        # is relation.
                        line[i] = line[i] + num_entities
                paths.append(line)
    return paths

def load_fb15k_data(paths_text_file):
    # 4152216 paths from pTransE paths + transE triples.
    paths = []
    num_entities = 14951


    with open(paths_text_file, 'r') as fh:
        for line in fh:
            line = eval(line)
            for i in range(len(line)):
                if i % 2 == 0:
                    # is entity.
                    pass
                else:
                    # is relation.
                    line[i] = line[i] + num_entities

            # TODO: offset the relations.
            paths.append(line)
            if len(paths) >= 10000:
                return paths
    return paths

if __name__ == "__main__":
    x = load_fb15k_shared_model_data()
    print(x[0])
    print(x[-1])
    print(len(x))