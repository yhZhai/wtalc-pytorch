import numpy as np
import os




# a = np.load('ActivityNet1.2-Annotations/labels_all.npy', allow_pickle=True)
a = np.load('labels_all.npy', allow_pickle=True)

print('a')



# def gen_classlist():
#     output = []
#     with open(os.path.join("data_list", "anet13_class.txt"), 'r') as f:
#         for line in f:
#             class_name = line.replace('\n', '').split(' ')[1]
#             # class_name = np.array(class_name)
#             output.append(class_name)
#     output = np.array(output)
#     np.save(os.path.join("ActivityNet1.3-Annotations", "classlist.npy"), output)
#     print('a')
#
# if __name__ == '__main__':
#     gen_classlist()

