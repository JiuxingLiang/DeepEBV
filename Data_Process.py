import numpy as np
import os
from scipy.io import loadmat


# from array import array
# from util import seq_matrix

def seq_matrix(seq_list, dim):  # One Hot Encoding

    tensor = np.zeros((len(seq_list), dim, 4))

    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            # if s == 'A' or s == 'a':
            #    tensor[i][j] = [1, 0, 0, 1, 0, 1, 0, 0]
            # if s == 'T' or s == 't':
            #    tensor[i][j] = [0, 1, 0, 0, 1, 0, 0, 0]
            # if s == 'C' or s == 'c':
            #    tensor[i][j] = [0, 0, 1, 0, 0, 1, 0, 1]
            # if s == 'G' or s == 'g':
            #    tensor[i][j] = [0, 0, 0, 1, 0, 0, 1, 1]
            # if s == 'N':
            #    tensor[i][j] = [0, 0, 0, 0, 0, 0, 0, 0]
            if s == 'A' or s == 'a':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T' or s == 't':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C' or s == 'c':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G' or s == 'g':
                tensor[i][j] = [0, 0, 0, 1]
            if s == 'N':
                tensor[i][j] = [0, 0, 0, 0]
            j += 1
    return tensor


def fasta_to_matrix():
    seq_name_file = ['Data/DNA_neg_Test.mat',
                     'Data/DNA_pos_Test.mat']  # data file

    print(seq_name_file)
    dim = 2000
    print('starting')

    for name in seq_name_file:
        if 'pos_Train' in name:
            print(name)
            y = []
            seq = []

            liness_pos = loadmat(name)
            lines = liness_pos['DNA_pos_Train']
            for line in lines:
                seq.append(line)  #

            print('pos__Train_starting!')
            global Data_pos_Train
            Data_pos_Train = seq_matrix(seq, dim)
            # global Data_pos_Train
            print('pos__Train_ending!')

        if 'neg_Train' in name:
            print(name)
            y = []
            seq = []

            Liness_neg = loadmat(name)
            lines = Liness_neg['DNA_neg_Train']
            for line in lines:
                seq.append(line)
            print('neg_Train_starting!')
            global Data_neg_Train
            Data_neg_Train = seq_matrix(seq, dim)
            # global Data_neg_Train
            print('neg_Train_ending!')

        if 'neg_Test' in name:
            print(name)
            y = []
            seq = []
            Liness_neg = loadmat(name)
            lines = Liness_neg['DNA_neg_Test']
            for line in lines:
                seq.append(line)
            print('neg_Test_starting!')
            global Data_neg_Test
            Data_neg_Test = seq_matrix(seq, dim)
            # global Data_neg_Test
            print('neg_Test_ending!')

        if 'pos_Test' in name:
            print(name)
            y = []
            seq = []
            Liness_neg = loadmat(name)
            lines = Liness_neg['DNA_pos_Test']
            for line in lines:
                seq.append(line)
            print('pos_Test_starting!')
            global Data_pos_Test
            Data_pos_Test = seq_matrix(seq, dim)
            # global Data_pos_Test
            print('pos_Test_ending!')

    # global Data_pos_Test
    # global Data_neg_Train
    # global Data_neg_Test
    # global Data_pos_Train

    # Data_Train = np.concatenate([Data_pos_Train, Data_neg_Train])  # Train data
    # Label_Train = np.concatenate([np.ones(len(Data_pos_Train)), np.zeros(len(Data_neg_Train))])  # Trian_label
    Data_Test = np.concatenate([Data_pos_Test, Data_neg_Test])  # Test data
    Label_Test = np.concatenate([np.ones(len(Data_pos_Test)), np.zeros(len(Data_neg_Test))])  # Test label

    # np.save('Data/Data_Train', Data_Train)
    # np.save('Data/Label_Train', Label_Train)
    np.save('Data/Data_Test', Data_Test)
    np.save('Data/Label_Test', Label_Test)


if __name__ == '__main__':
    fasta_to_matrix()