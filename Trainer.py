import numpy as np
import tensorflow as tf
import datetime

"""
    Shape(row, column)
"""

# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

"""
    Load data and get all the chars in text.
"""
# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(text1)
# exit(0)
text = open('clean').read()
# words = text.split()
# chars = np.unique(words)
# print(len(chars))
chars = sorted(list(set(text)))
char_size = len(chars)
print('char_size : ' + str(char_size))
# words = sorted(words)
# word_size = len(words)
# print('word_size : ' + str(word_size))
"""
    create dictionary to link each char to an id, and vice versa
"""
char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))
# word2id = dict((w, i) for i, w in enumerate(words))
# id2word = dict((i, w) for i, w in enumerate(words))


"""
    Text data have to be arranged into set of sections of {len_per_section} characters text 
    with the next character following the section as the output of the section.
    
    Then from the starting of the previous section, {skip} characters are skipped to start the next
    section that will form the input of the following input.
    
    Considering {len_per_section} to be 50 and skip = 2.
    section_1 = text[n:n+50]        =>      next_char_1 = text[n+50]
    section_2 = text[n+2:n+2+50]    =>      next_char_2 = text[n+2+50]
    ......
    ....
"""
len_per_section = 50
skip = 2
sections = []
next_chars = []


for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])
"""
for i in range(0, word_size - len_per_section):
    sections.append(words[i:i+len_per_section])
    next_word.append(words[i+len_per_section])
exit(0)
"""
"""
    Create two vectors of zeros to
    
    X   => to store sections, 3 dimension.
            1-D to store a char,
            2-D to store a specific section of characters
            3-D to store all the sections
    
    y   =>  to store the next chars, 2 dimension.
            1-D to store a char,
            2-D to store all the next chars each for a section
            
    dtype   =>  int
"""
X = np.zeros((len(sections), len_per_section, char_size), dtype=int)
y = np.zeros((len(sections), char_size), dtype=int)

"""
    Go through the sections, grab the characters and one-hot encode them.
"""
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1
"""
Var
"""
batch_size = 200
max_steps = 20
log_every = 1
save_every = 5
hidden_nodes = 100
"""
    Directory to store a trained model
"""
checkpoint_directory = 'ckpt/model'  # + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

"""
    Directory to store tensorboard summaries
"""
tensorboard_directory = 'temp/tensorboard'

if tf.gfile.Exists(tensorboard_directory):
    tf.gfile.DeleteRecursively(tensorboard_directory)
tf.gfile.MakeDirs(tensorboard_directory)

"""
    Define variable needed for the operations and outline the flow of data through this operations of computations.
    Basically outlining a computation graph to be later used.  
"""
graph = tf.Graph()
with graph.as_default():

    global_step = tf.Variable(0)

    with tf.name_scope('train_input'):
        # placeholders (no data in it during graph const), but will hold data during a session
        """
        1D  :   Store batch
        2D  :   Store input chars section
        3D  :   Store char
        """
        data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size], name="X")    # input data

        """
            1D  :   Store batch
            3D  :   Store output char
            """
        labels = tf.placeholder(tf.float32, [batch_size, char_size], name="Y")    # output data

    with tf.name_scope("Weights"):
        """
            Initialise weights and biases.
            Weights initialised with random values from a truncated normal distribution.
            Biases initialised to zero.
        """
        with tf.name_scope("update"):
            w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32), name='wui')
            tf.summary.histogram('weights_i', w_ii)
            w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32), name='wio')
            tf.summary.histogram('weights_o', w_io)
            b_i = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32), name='bi')
            tf.summary.histogram('bias', b_i)
        with tf.name_scope("reset"):
            w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32), name='wri')
            tf.summary.histogram('weights_i', w_fi)
            w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32), name='wfo')
            tf.summary.histogram('weights_o', w_fo)
            b_f = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32), name='bf')
            tf.summary.histogram('bias', b_f)
        with tf.name_scope("output"):
            w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32), name='woi')
            tf.summary.histogram('weights_i', w_oi)
            w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32), name='woo')
            tf.summary.histogram('weights_o', w_oo)
            b_o = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32), name='bo')
            tf.summary.histogram('bias', b_o)
        with tf.name_scope("Cell"):
            w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32), name='wci')
            tf.summary.histogram('weights_i', w_ci)
            w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32), name='wco')
            tf.summary.histogram('weights_o', w_co)
            b_c = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32), name='bc')
            tf.summary.histogram('bias', b_c)

    """
        lstm(i, o, s):
        Take in Inputs i and Outpus o, and Previous State and compute a new
        state and output.
        _____________________________________________________        
            Input   i   :   shape=(batch_size, char_size)
            Output  o   :   shape=(batch_size, hidden_layers)
            State   s   :   shape=(batch_size, hidden_layers)
        _____________________________________________________
    """
    def lstm(i, o, s, name):
        with tf.name_scope(str(name) + "LSTM"):
            # the scalars are all between zero and one inclusive.
            with tf.name_scope("Gates"):
                # compute a scalar that decides what to remember about the previously seen characters
                # and what to include about the new character.
                with tf.name_scope("update_gate"):
                    update_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
                    # tf.summary.scalar('input_gate', input_gate)

                # compute a scalar that decides what to discard about the previous output.
                with tf.name_scope("reset_gate"):
                    reset_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
                    # tf.summary.scalar('forget_gate', forget_gate)

                    # compute a scalar that decides what to include in the new output.
                with tf.name_scope("output_gate"):
                    output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
                    # tf.summary.scalar('output_gate', output_gate)
            with tf.name_scope("compute_cell"):
                memory_cell = tf.matmul(i, w_ci) + reset_gate * tf.matmul(o, w_co) + b_c

            with tf.name_scope("Update_state"):
                s = update_gate * s + (1 - update_gate) * memory_cell

            o = output_gate * tf.tanh(s)
            # tf.summary.scalar('state', s)

            return o, s

    # initial output and state to zero
    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])

    # loop through all the sections.
    for i in range(len_per_section):

        """
            data[:, i, :]   :   i(th) section from the given batch
        """
        output, state = lstm(data[:, i, :], output, state, i)

        """
            outputs_all_i   :   stores the outputs from the lstm
            labels_all_i    :   stores the characters that follow, Which are the correct labels
        """

        if i == 0:  # if first section
            outputs_all_i = output  # make current output the start
            labels_all_i = data[:, i + 1, :]    # make next input as the start

        elif i != len_per_section - 1:  # not first or last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)   # append the current output
            labels_all_i = tf.concat([labels_all_i, data[:, i + 1, :]], 0)  # append the next input

        else:   # the last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)   # append the current output
            labels_all_i = tf.concat([labels_all_i, labels], 0)     # append the final labels

    with tf.name_scope('ouput_weight_bias'):
        w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1), name='w')
        tf.summary.histogram('W', w)
        b = tf.Variable(tf.zeros([char_size]), name='b')
        tf.summary.histogram('B', w_co)

    with tf.name_scope('logits'):
        logits = tf.matmul(outputs_all_i, w) + b
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_all_i))
    tf.summary.scalar('cross_entropy', loss)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_predictions'):
            correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_all_i, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_directory, graph)
"""
    create a session and train the model on the data
"""
with tf.Session(graph=graph) as sess:

    # initialise all variables
    tf.global_variables_initializer().run()

    offset = 0

    saver = tf.train.Saver()

    X_length = len(X)

    # retrain on the data every epoch
    for step in range(max_steps):

        # pass the data to the model in batches
        for batch in range(X_length//batch_size + 1):

            offset = offset % X_length

            if offset <= (X_length - batch_size):
                batch_data = X[offset: offset + batch_size]
                batch_labels = y[offset: offset + batch_size]
                offset += batch_size
            else:
                to_add = batch_size - (X_length - offset)
                batch_data = np.concatenate((X[offset:X_length], X[0: to_add]))
                batch_labels = np.concatenate((y[offset:X_length], y[0: to_add]))
                offset = to_add

            _, acc, summary, training_loss = sess.run([optimizer, accuracy, merged, loss], feed_dict={data: batch_data, labels: batch_labels})

            if step % log_every == 0:
                print('training loss at step %d - batch %d: %.2f (%s)' % (step, batch, training_loss, datetime.datetime.now()))
                print('Accuracy at step %d - batch %s: %s' % (step, batch, acc))

                if batch % save_every == 0:
                    saver.save(sess, checkpoint_directory + '/model', global_step=step)

        train_writer.add_summary(summary, step)
