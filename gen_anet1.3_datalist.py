import os
import json
import numpy as np

def load_json():
    file_name = os.path.join("data_list", "activity_net.v1-3.min.json")
    file = open(file_name, 'r')
    json_file = json.load(file)
    database = json_file["database"]
    return database



def gen_datalist():
    train_file = os.path.join("data_list", "anet_1.3_i3d_train_feature.txt")
    val_file = os.path.join("data_list", "anet_1.3_i3d_val_feature.txt")
    video_name_array = []
    subset_array = []
    with open(train_file, 'r') as f:
        for line in f:
            line = line.split(' ')
            video_name_array.append(line[1])
            # subset_array.append("training")
    with open(val_file, 'r') as f:
        for line in f:
            line = line.split(' ')
            video_name_array.append(line[1])
            # subset_array.append("validation")
    database = load_json()
    duration_array = []
    segments_list = set()
    for video_name in video_name_array:
        # duration_array.append(database[video_name]['duration'])

        for seg in database[video_name]['annotations']:
            segments_list.add(seg['label'])
    segments_list = np.array(sorted(list(segments_list)))

    np.save("classlist.npy", segments_list)
    # duration_array = np.array(duration_array)
    # np.save("duration.npy", duration_array)
    print('a')
    # video_name_array = np.array(video_name_array)
    # subset_array = np.array(subset_array)
    # np.save("subset.npy", subset_array)
    # np.save("videoname.npy", video_name_array)

    # path2 = "/Users/yhzhai/LocalDocument/Data/anet_all/anet"
    # path3 = "/Users/yhzhai/LocalDocument/Data/anet_all/anet1.3"
    # train2 = os.listdir(os.path.join(path2, "train"))
    # train3 = os.listdir(os.path.join(path3, "train"))
    # eva2 = os.listdir(os.path.join(path2, "val"))
    # eva3 = os.listdir(os.path.join(path3, "val"))
    # video_name_array = []
    # for i in train2 + train3:
    #     video_name = i[:11]
    #     video_name_array.append(video_name)
    # video_name_array = np.array(video_name_array)
    # np.save("videoname.npy", video_name_array)




if __name__ == '__main__':
    # load_json()
    # load_json()
    gen_datalist()