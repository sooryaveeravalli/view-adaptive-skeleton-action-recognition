import glob
from scipy.io import loadmat
import numpy as np
performer = []
for i in range(1,41):
        if i < 10:
                performer.append('P00'+str(i))
        else:
                performer.append('P0'+str(i))
        print(performer[i-1])

train_data = [None]*56578
test_data = [None]*56578
train_len = [None]*56578
test_len = [None]*56578
train_label = [None]*56578
test_label = [None]*56578
train_count = 0
test_count = 0

# data = loadmat('./skeletonfiles/S008C001P036R002A006.skeleton.mat')
# print(len(data['allbodyinfo'][0]))
# print(len(data['allbodyinfo'][0][0][0][0]))
# print(len(data['allbodyinfo'][0][0][0][0][0][11][0]))
# print(data['allbodyinfo'][0][0][0][0][0][11][0][0][0][0][0])

for i in range(0,40):
        print("===============================================================            " + str(i) +"           ===================================================================")
        name = [f for f in glob.glob('./skeletonfiles/*'+performer[i]+'*')]
        if len(name) != 0:
                for j in range(len(name)):
                        print(j)
                        m = 1
                        file_name = str(name[j])
                        data = loadmat(file_name)
                        noofframes = len(data['allbodyinfo'][0])
                        temp = np.zeros((noofframes,50,3))
                        for k in range(noofframes):
                                if len(data['allbodyinfo'][0][k][0][0]) != 0:
                                        for l in range(25):
                                                temp[k,l,0] = data['allbodyinfo'][0][k][0][0][0][11][0][l][0][0][0]
                                                temp[k,l,1] = data['allbodyinfo'][0][k][0][0][0][11][0][l][1][0][0]
                                                temp[k,l,2] = data['allbodyinfo'][0][k][0][0][0][11][0][l][2][0][0]
                                        if len(data['allbodyinfo'][0][k][0][0]) == 2:
                                                if m == 1:
                                                    m = 0
                                                    print('extra')
                                                for l in range(25):
                                                    temp[k,l+25,0] = data['allbodyinfo'][0][k][0][0][1][11][0][l][0][0][0]
                                                    temp[k,l+25,1] = data['allbodyinfo'][0][k][0][0][1][11][0][l][1][0][0]
                                                    temp[k,l+25,2] = data['allbodyinfo'][0][k][0][0][1][11][0][l][2][0][0]
                        if i == 2  or ( i > 4 and i < 12 and i != 7 and i != 8) or ( i > 18 and i < 24 ) or i == 25 or i == 28 or i == 29 or i == 31 or i == 32 or i == 35 or i == 36 or i == 38 or i == 39:
                            test_data[test_count] = temp
                            test_len[test_count] = noofframes
                            if int(file_name[26]) == 0:
                                test_label[test_count] = int(file_name[27])
                            else:
                                test_label[test_count] = int(file_name[26:28])
                            test_count += 1
                        else:
                            train_data[train_count] = temp
                            train_len[train_count] = noofframes
                            if int(file_name[26]) == 0:
                                train_label[train_count] = int(file_name[27])
                            else:
                                train_label[train_count] = int(file_name[26:28])
                            train_count += 1

train_data = train_data[0:train_count]
train_label = train_label[0:train_count]
train_len = train_len[0:train_count]

test_data = test_data[0:test_count]
test_label = test_label[0:test_count]
test_len = test_len[0:test_count]

np.save('train_ntus_data.npy', train_data)
np.save('train_ntus_label.npy', train_label)
np.save('train_ntus_len.npy', train_len)

np.save('test_ntus_data.npy', test_data)
np.save('test_ntus_label.npy', test_label)
np.save('test_ntus_len.npy', test_len)

