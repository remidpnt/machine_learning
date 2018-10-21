import tensorflow as tf
from sys import getsizeof
import tqdm
import numpy as np
from random import randint
import re
import io
from tensorflow.python.lib.io import file_io
from os import listdir
from os.path import isfile, join
import datetime
from own_functions import attention

################################################################################ Parameters

maxSeqLength = 249
epotchSize = 10000000000
shuffle_count = 300
batchSize = 256
lstmUnits = 128
numClasses = 2
iterations = 1#100000
lr = 0.00012
attention_size = 100


logs_path_train = "gs://nlp-remi/" + 'logs/train__' + datetime.datetime.now().isoformat()
logs_path_test = "gs://nlp-remi/" + 'logs/test__' + datetime.datetime.now().isoformat()
logs_path_model = "gs://nlp-remi/" + 'logs/model__' + datetime.datetime.now().isoformat()


f = io.BytesIO(file_io.read_file_to_string('gs://nlp-remi/idsMatrix.npy'))
ids = np.load(f)

f2 = io.BytesIO(file_io.read_file_to_string('gs://nlp-remi/vectorsPlusOne.npy'))
vectors = np.load(f2)

X_init = tf.placeholder(tf.float32, shape=(2196020, 300))
wordVectors = tf.Variable(X_init)




################################################################################ Data part

ids_train = np.concatenate((ids[0:10000,:], ids[12500:22500,:]), axis=0)
ids_test = np.concatenate((ids[10000:12500,:], ids[22500:,:]), axis=0)

ids_train = np.take(ids_train,np.random.permutation(ids_train.shape[0]),axis=0,out=ids_train)
ids_test = np.take(ids_test,np.random.permutation(ids_test.shape[0]),axis=0,out=ids_test)



dataset_train = tf.data.Dataset.from_tensor_slices(ids_train).cache().shuffle(shuffle_count).repeat(epotchSize).batch(batchSize).prefetch(5) # Make sure you always have 1 batch ready to serve
dataset_test  = tf.data.Dataset.from_tensor_slices(ids_test).cache().shuffle(shuffle_count).repeat(epotchSize).batch(batchSize).prefetch(5) # Make sure you always have 1 batch ready to serve

iterator_train = dataset_train.make_one_shot_iterator()
iterator_test  = dataset_test.make_one_shot_iterator()


conditionTrain = tf.placeholder(tf.int64, shape=[], name="condition")
iterator       = tf.cond(conditionTrain>0, lambda: iterator_train.get_next(), lambda: iterator_test.get_next())
dropout_prob_keep = tf.cond(conditionTrain>0, lambda: 0.75, lambda: 1.0)

input, output = tf.split(iterator, [249,1], axis=1)

################################################################################ Graph

sess = tf.Session()

labels = tf.cast(output, tf.int32)#[bs,1]
labels_one_hot=tf.reshape(tf.one_hot(labels,depth=2),[batchSize,2])
input_data = tf.cast(input, tf.int32)
data = tf.nn.embedding_lookup(wordVectors,input_data)


def LSTM():
    return tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.BasicLSTMCell(lstmUnits), output_keep_prob=dropout_prob_keep)

def tupple():
    return tf.contrib.rnn.LSTMStateTuple(h=tf.Variable(initial_value=tf.random_normal( (24, lstmUnits), stddev=0.1), dtype=tf.float32 ),c=tf.Variable(initial_value=tf.random_normal( (24, lstmUnits), stddev=0.1),dtype=tf.float32 ))

output, output_ =tf.nn.bidirectional_dynamic_rnn(
                    LSTM(),
                    LSTM(),
                    inputs=data,
                    dtype=tf.float32,
                    time_major = False
                    )


output_forward, output_backward = output  # A chaque fois, on a une forme: [bs, time, LSTM.shape] soit ici: (24,250,lstmUnits)

last = (output_forward + output_backward)/2
print("Last = ", last)



last_LSTM_output_forward = tf.gather(output_forward,[maxSeqLength-1],axis=1)# On selectionne le dernier output
last_LSTM_output_forward = tf.reshape(last_LSTM_output_forward, [batchSize,lstmUnits])

last_LSTM_output_backward = tf.gather(output_backward,[0],axis=1)# On selectionne le premier output
last_LSTM_output_backward = tf.reshape(last_LSTM_output_backward, [batchSize,lstmUnits])

attention_vect = attention(output, attention_size, time_major=False, return_alphas=False)
max = tf.reduce_max(last,axis=1)
mean = tf.reduce_mean(last,axis=1)

print("As a size, we have for attention_vect,max, mean, last_LSTM_output_forward,last_LSTM_output_backward: ", attention_vect , max, mean, last_LSTM_output_forward, last_LSTM_output_backward )
concat = tf.concat([attention_vect,max, mean, last_LSTM_output_forward,last_LSTM_output_backward], axis=1)

prediction = tf.contrib.layers.fully_connected(concat, numClasses, activation_fn=tf.identity)
prediction=tf.nn.softmax(prediction)
correctPred = tf.equal( tf.cast(tf.argmax(prediction,1), tf.int32), labels)

################################################################################ Cost function

accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
print("prediction, labels",prediction, labels_one_hot)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( logits=prediction, labels=labels_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

################################################################################ Tensorboard

tf.summary.scalar("Loss", loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(logs_path_train, sess.graph)
writer_test = tf.summary.FileWriter(logs_path_test, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer(), feed_dict={X_init: vectors})

################################################################################ Training part

for i in tqdm.tqdm(range(iterations), total=iterations ):


   #Write summary to Tensorboard

   if (i % 25 == 0):
       summary = sess.run(merged, feed_dict={conditionTrain:1})
       writer.add_summary(summary, i)

       summary_test = sess.run(merged, feed_dict={conditionTrain:0})
       writer_test.add_summary(summary_test, i)

   else:
       #Next Batch of reviews
       sess.run(optimizer, feed_dict={conditionTrain:1})


   #Save the network every 10,000 training iterations
   # if (i % 10000 == 0 and i != 0):
   #     save_path = saver.save(sess, logs_path_model, global_step=i)
   #     print("saved to %s" % save_path)


writer.close()
writer_test.close()


#tensorboard --logdir=gs://nlp-remi/logs/ --port 8000 --reload_interval=5
# gcloud ml-engine jobs submit training plants_preproc`date '+%Y%m%d_%H%M%S'` --package-path trainer     --module-name trainer.main_bidirectional  --config mlengine/preproc-config.yaml   --job-dir gs://nlp-remi/     --region europe-west1
