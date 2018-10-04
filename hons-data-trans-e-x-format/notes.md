All the files in here:
- entity2id.txt
- relation2id.txt
- test2id.txt
- train2id.txt
- valid2id.txt

are all in the format `e1 e2 rel` this is the data format the the code in `tensorflow-trans-x` expects.
We can use this to train `trans-e` and a joint `doc2vec` model.

To make our path data in a similar format for the doc2vec model we need to load up the adj matrix dump in the relevant `hons-data` folder.

The path files however (`./create_paths`), are in the format `e1, r1, e2, r2, e3, r4, e4 ...` etc.
They are the actual paths in the form `entity`, `relation`, `entity`. 
We will use this to train the `doc2vec` model.

If we want to use the code in `KB2E/PTransE` we need to convert it into `mid_1 mid_2, rel_word_rep` I think.