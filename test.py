import numpy as np
import pandas as pd
# from dataset import word_vectors # 从数据集引入词向量
word_vectors = np.load('wordVectors.npy')
# 模型参数设置
batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 30001
num_dimensions = 50
max_seq_num = 250
ids = np.load('test_ids.npy')
import tensorflow as tf

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_labels])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])

# data = tf.Variable(tf.zeros([batch_size,max_seq_num,num_dimensions]))#num_dimensions表示词向量的维数，此处为50 Dimensions for each word vector
data = tf.nn.embedding_lookup(word_vectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# dynamic_rnn的第一个输出可以被认为是最后的隐藏状态，该向量将重新确定维度，然后乘以一个权重加上bias,获得最终的label
weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

# 正确的预测函数以及正确率评估参数
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

df = pd.read_csv('test_data0.csv', encoding='utf-8')

for i in range(0,len(df['review']),24):
    arr = np.zeros([batch_size, max_seq_num])

    for j in range(batch_size):
        if i+j >= len(df['review']):
            break
        arr[j] = ids[i+j]
    next_batch = arr
    predic = tf.argmax(prediction, 1)
    predict = sess.run(predic,{input_data: next_batch})
    for m in range(len(predict)):

        if predict[m] == 0:
            f1 = open('result.txt','a')
            print(i+m,'positive')
            f1.write(str(i+m))
            f1.write(' positive\n')
            f1.close()
        elif predict[m] == 1:
            f1 = open('result.txt', 'a')
            print(i + m, 'negative')
            f1.write(str(i + m))
            f1.write(' negative\n')
            f1.close()







