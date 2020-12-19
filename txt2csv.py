import pandas as pd
f1 = open('result.txt','r')
df = pd.read_csv('submission0.csv', encoding='utf-8')

for i in range(10000):
    p = f1.readline()
    s = p.split()
    print(s)
    df['sentiment'].loc[i] = s[1]

df.to_csv('submission.csv', encoding='utf-8',index=False)
f1.close()