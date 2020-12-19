

import numpy as np
import pandas as pd
text = open('record.txt','a',)
words_list = np.load('wordsList.npy')
print('载入word列表')
words_list = words_list.tolist()  # 转化为list
words_list = [word.decode('UTF-8') for word in words_list]
print(words_list)

print('载入文本向量')

# 检查数据
print(len(words_list))


# 构造整个训练集索引
# 需要先可视化和分析数据的情况从而确定并设置最好的序列长度

num_words = []
df = pd.read_csv('train.csv', encoding='utf-8')
for i in range(len(df['review'])):
    line = df['review'].loc[i]
    counter = len(line.split())
    num_words.append(counter)

num_files = len(num_words)
print('评论总数',num_files)
print('所有词的数量',sum(num_words))
lengh = sum(num_words)/len(num_words)
print('评论平均词长度',lengh)
#进行可视化

#matplotlib.use('qt4agg')


# 将文本生成一个索引矩阵，并且得到25000 x 300 的矩阵
import re
strip_special_chars = re.compile('[^A-Za-z0-9 ]+')
def cleanSentences(string):
    string = string.lower().replace("<br />"," ") #字符替换
    return re.sub(strip_special_chars,"",string.lower())

max_seq_num = 250
ids = np.zeros((num_files,max_seq_num),dtype='int32')
file_count = 0 # 文件数目
for i in range(len(df['review'])):
    if df['sentiment'].loc[i] == 'positive':
        print(i)
        indexCounter = 0
        line = df['review'].loc[i]
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            print(word)
            try:
                ids[file_count][indexCounter] = words_list.index(word)  # 该单词在词汇表中的索引值放入矩阵中
            except ValueError:
                ids[file_count][indexCounter] = 400001 # 未知的词
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count += 1
positive_count = file_count
print('positive_file:',positive_count)
text.write('positive_file:')
text.write(str(positive_count))

for i in range(len(df['review'])):
    if df['sentiment'].loc[i] == 'negative':
        print(i)
        indexCounter = 0
        line = df['review'].loc[i]
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            print(word)
            try:
                ids[file_count][indexCounter] = words_list.index(word)  # 该单词在词汇表中的索引值放入矩阵中
            except ValueError:
                ids[file_count][indexCounter] = 399999  # 未知的词
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count += 1
negative_count = file_count - positive_count
print('negative_file:',negative_count)
text.write('\nnegative_file:')
text.write(str(negative_count))

text.close()

# 保存到文件
np.save('train_ids',ids)  #保存为npy文件
