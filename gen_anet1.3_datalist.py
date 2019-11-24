import os
import json

def load_json():
    file_name = os.path.join("data_list", "activity_net.v1-3.min.json")
    file = open(file_name, 'r')
    json_file = json.load(file)
    database = json_file["database"]
    print('a')


def gen_datalist():
    path2 = "G:\\anet"
    path3 = "G:\\anet1.3"
    train2 = os.listdir(os.path.join(path2, "train"))
    train3 = os.listdir(os.path.join(path3, "train"))
    eva2 = os.listdir(os.path.join(path2, "val"))
    eva3 = os.listdir(os.path.join(path3, "val"))
    with open("anet1.3_i3d_train.txt", 'w') as f:
        for i in train2:
            name = i[:11]
            path = os.path.join(path2, "train")
            f.write("{} {}\n".format(path, name))
        for i in train3:
            name = i[:11]
            path = os.path.join(path3, "train")
            f.write("{} {}\n".format(path, name))


if __name__ == '__main__':
    # load_json()
    gen_datalist()