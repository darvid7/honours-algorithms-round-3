import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import text_helpers
import subprocess

p = os.getcwd()

ll = ctypes.cdll.LoadLibrary
lib = ll(os.path.join(p, "./init.so"))
test_lib = ll(os.path.join(p, "./test.so"))

print("working dir: %s, process id: %s" % (p, os.getpid()))


class Config(object):

    def __init__(self):
        lib.setBernFlag(0)
        self.learning_rate = 0.001
        self.L1_flag = True
        self.hidden_size = 100
        self.nbatches = 100
        self.entity = 0
        self.relation = 0
#        self.trainTimes = 2
        self.margin = 1.0
        # Doc2vec config items.
        self.num_docs = 150000 # number of paths


class JointModel:

    def __init__(self, config):
       #  print("total epochs: %s" % config.trainTimes)
        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size
        margin = config.margin

        # doc2vec things.
        self.NUM_ENTITIES = 14951
        self.NUM_RELATIONS = 1345
        self.window_size = 2  # How many words to consider to the left.
        self.vocabulary_size = self.NUM_ENTITIES + self.NUM_RELATIONS
        self.word_embedding_size = size
        self.doc_embedding_size = 100
        self.doc2vec_epochs = 100
        self.d2v_embd_out_dir = "./shared_doc2vec_half"
        # TransE things.
        self.transe_embd_out_dir = "./shared_transE_half"
        self.config = config
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            # note: tf.get_variable() pre much the same as tf.Variable().
            print("in tf.name_scope('embedding'), entity total: %s, relation_total: %s" % (entity_total, relation_total))
            # Embedding scope for TransE.
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[entity_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[relation_total, size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            # Embedding scope for doc2vec.
            self.word_embeddings = tf.get_variable(
                name="word_embeddings",
                initializer=tf.concat([self.ent_embeddings, self.rel_embeddings], 0)
            )
                # tf.concat([self.ent_embeddings, self.rel_embeddings], 0)
            # Word embeddings are entity & relation embeddings.
            self.doc_embeddings = tf.get_variable(
                name="doc_embedding",
                #shape=[config.num_docs, self.doc_embedding_size],
                initializer=tf.random_uniform([config.num_docs, self.doc_embedding_size], -1.0, 1.0))

            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
        print("YAY IT INITALIZED!")

    def train_doc2vec(self, sess):
        # From ML cookbook.

        text_data = text_helpers.load_fb15k_shared_model_data()
        batch_size = 1000
        num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
        model_learning_rate = 0.001

        concat_word_doc_size = self.doc_embedding_size + self.word_embedding_size
        # Uses Noise Contrastive Estimation this instead of hierarchical softmax.
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, concat_word_doc_size], stddev=1.0 / np.sqrt(concat_word_doc_size))
        )
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Create data/target placeholders
        x_inputs = tf.placeholder(tf.int32, shape=[None, self.window_size + 1])  # plus 1 for doc index
        y_target = tf.placeholder(tf.int32, shape=[None, 1])

        # Lookup the word embedding
        # Add together element embeddings in window:
        embed = tf.zeros([batch_size, self.word_embedding_size])
        for element in range(self.window_size):
            embed += tf.nn.embedding_lookup(self.word_embeddings, x_inputs[:, element])

        doc_indices = tf.slice(x_inputs, [0, self.window_size], [batch_size, 1])
        doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, doc_indices)  # look up doc_embeddings via the doc indicies.

        # concatenate embeddings
        final_embed = tf.concat([embed, tf.squeeze(doc_embed)], 1)

        # Get loss from prediction
        loss = tf.reduce_mean(tf.nn.nce_loss(
            nce_weights, nce_biases, y_target, final_embed,
            num_sampled, self.vocabulary_size))

        # Create optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
        train_step = optimizer.minimize(loss)

        # Create model saving operation
        saver = tf.train.Saver({"embeddings": self.word_embeddings, "doc_embeddings": self.doc_embeddings})

        # Add variable initializer.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Run the skip gram model.
        print('Starting Training Skip Gram Doc2Vec Model')
        loss_vec = []
        loss_x_vec = []
        for i in range(self.doc2vec_epochs):
            batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                          self.window_size, method='doc2vec')
            feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

            # Run the train step
            sess.run(train_step, feed_dict=feed_dict)

            # Return the loss
            if (i + 1) % 50 == 0:
                loss_val = sess.run(loss, feed_dict=feed_dict)
                loss_vec.append(loss_val)
                loss_x_vec.append(i + 1)
                print('[doc2vec] Loss at step {} : {}'.format(i + 1, loss_val))


def numpy_list_of_lists_differ(l1, l2):
    """
    Returns True if l1 == l2. Both args are numpy lists of numpy lists of real numbers.
    Can't just use np.all(np_matrix1 == np_matrix2) I think.
    :param l1:
    :param l2:
    :return:
    """
    same_list = [True]  # Only true if every element of every list is the same.
    for i, line1 in enumerate(l1):
        line2 = l2[i]
        same_list.append(np.all(line1 == line2))
    return not np.all(same_list)  # If np.all(same_list) is True then they are all the same so they dont differ so return False.


def main(_):
    print("START shared_model.py")

    start = time.time()
    config = Config()
    log_tf_graph = True
    check_vectors_actually_trained_write_out = False
    # total_shared_epochs = 1
    total_shared_epochs = 1000
    print("Total shared epochs: %s" % total_shared_epochs)
    lib.init()
    config.relation = lib.getRelationTotal()
    config.entity = lib.getEntityTotal()
    config.batch_size = lib.getTripleTotal() // config.nbatches
    # model.train_transE()

    test_result_dir = "./test_results"
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                trainModel = JointModel(config=config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())

            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict], feed_dict)
                return predict

            ph = np.zeros(config.batch_size, dtype=np.int32)
            pt = np.zeros(config.batch_size, dtype=np.int32)
            pr = np.zeros(config.batch_size, dtype=np.int32)
            nh = np.zeros(config.batch_size, dtype=np.int32)
            nt = np.zeros(config.batch_size, dtype=np.int32)
            nr = np.zeros(config.batch_size, dtype=np.int32)

            ph_addr = ph.__array_interface__['data'][0]
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.testHead.argtypes = [ctypes.c_void_p]
            test_lib.testTail.argtypes = [ctypes.c_void_p]

            import copy
            entities_after_one_doc2vec_then_one_transE = None
            entities_after_one_doc2vec = None
            entities_before_everything = copy.deepcopy(sess.run(trainModel.ent_embeddings))

            if log_tf_graph:
                output_dir = os.path.join(os.getcwd(), 'logged_graph')  # Creates a directory `logged_graph`.
                file_writer = tf.summary.FileWriter(output_dir, sess.graph)

            # test_every = 100
            # 1000 test epochs, will test at quarters to see if it kinda converges
            # adds 30 mins for each test.
            # ~ 2 hours + 2 hours run time, check this at 8pm.
            test_epochs = {
                500, 750, 999
            }


            # TRAIN STEP.
            for i in range(total_shared_epochs):
                print("epoch: " + str(i))

                # Train doc2vec first.
                print("Training doc2vec I think")
                trainModel.train_doc2vec(sess)

                if entities_after_one_doc2vec is None:
                    entities_after_one_doc2vec = copy.deepcopy(sess.run(trainModel.ent_embeddings))

                # Train transE after.
                print("Training TransE")
                # train transE.
                # for times in range(config.trainTimes):
                res = 0.0
                for batch in range(config.nbatches):
                    lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                    res += train_step(ph, pt, pr, nh, nt, nr)
                    current_step = tf.train.global_step(sess, global_step)
                print("[transE] res: %s" % res)

                entities_after_one_doc2vec_then_one_transE = copy.deepcopy(sess.run(trainModel.ent_embeddings))

                if i in test_epochs:
                    # Save values so we can test on these values.
                    print("saved transE model")
                    saver.save(sess, trainModel.transe_embd_out_dir + '/model.ckpt')
                    print("Testing at epoch: %s" % i)
                    cmd_args = ["python3", "shared_model_test.py"]
                    fh_log = open(os.path.join(test_result_dir, "test_results_epoch_%s.log.txt" % i), 'w')
                    fh_err = open(os.path.join(test_result_dir, "test_results_epoch_%s.err.txt" % i), 'w')
                    subprocess.Popen(cmd_args, stderr=fh_err, stdout=fh_log).wait()
                    fh_log.close()
                    fh_err.close()
                    print("Finished testing at epoch: %s, results in %s/test_results_epoch_%s.log.txt" % (i, test_result_dir, i))


            # END TRAIN STEP.

            print("Before and after one trans E are differ: ", end="")
            print(numpy_list_of_lists_differ(entities_before_everything, entities_after_one_doc2vec))

            print("After one transE and then one doc2vec are differ: ", end="")
            print(numpy_list_of_lists_differ(entities_after_one_doc2vec, entities_after_one_doc2vec_then_one_transE))

            if check_vectors_actually_trained_write_out:
                with open('entities_before_everything.txt', 'w') as fh:
                    for i in entities_before_everything:
                        fh.write(str(i) + "\n")

                with open('entities_after_one_transE.txt', 'w') as fh:
                    for i in entities_after_one_doc2vec_then_one_transE:
                        fh.write(str(i) + "\n")

                with open('entities_after_one_tranE_and_one_doc2vec.txt', 'w') as fh:
                    for i in entities_after_one_doc2vec:
                        fh.write(str(i) + "\n")
            print("Time taken mins = %s" % ((time.time() - start) // 60))

if __name__ == "__main__":
    tf.app.run()