import numpy as np
import os

def get_fps(video_name):
    fps = 30
    with open(os.path.join('ActivityNet1.3-Annotations', 'misc_fps.txt'), 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            if line[0] == video_name:
                fps = float(line[1])
                break
    return fps


#a = np.load('ActivityNet1.2-Annotations/labels_all.npy', allow_pickle=True)
#b = np.load('ActivityNet1.2-Annotations/labels.npy', allow_pickle=True)

#for i, j in zip(a, b):
#    if not i == j:
#        print(i, j)

# a = np.load('labels_all.npy', allow_pickle=True)
#
#print('a')

# fps_sum = 0
# count = 0
# with open(os.path.join("data_list", "anet1.3_val_fps.txt"), 'r') as f:
#     for line in f:
#         line = line.replace('\n', '').split(' ')
#         fps_sum += float(line[1])
#         count += 1
# print(fps_sum / count)


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

