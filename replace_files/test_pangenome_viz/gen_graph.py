import matplotlib.pyplot as plt
import numpy as np

node_id=[]
nx=[]
ny=[]
with open('sgd_out.txt') as f:
    for line in f:
        words=line.split()
        # print(words)
        node_id.append(int(str(words[0])))
        nx.append(float(words[2]))
        ny.append(float((words[3])))

arr=[]
with open('vis_id.txt') as f:
    for line in f:
        words=line.split()
        for i in range(0, len(words), 2):
            arr.append(int(words[i])/2)

print(arr)

for i in range(len(arr)-1):
    plt.plot([nx[int(arr[i])], nx[int(arr[i+1])]], [ny[int(arr[i])], ny[int(arr[i+1])]])

plt.show()