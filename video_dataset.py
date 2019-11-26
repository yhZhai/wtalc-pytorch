import numpy as np
import glob
import utils
import time
import torch
import os


class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_annotations = self.dataset_name + '-Annotations/'
        if self.dataset_name != "ActivityNet1.3":
            self.path_to_features = '%s-%s-JOINTFeatures.npy' % (args.dataset_name, args.feature_type)

            self.features = np.load(self.path_to_features, encoding='bytes', allow_pickle=True)
        self.segments = np.load(self.path_to_annotations + 'segments.npy', allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy', allow_pickle=True)  # Specific to Thumos14
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy', allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + 'subset.npy', allow_pickle=True)
        self.video_name = np.load(self.path_to_annotations + 'videoname.npy', allow_pickle=True)
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]

        self.train_test_idx()
        self.classwise_feature_mapping()

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if 'thumos' in self.dataset_name.lower():
                if s == 'validation':
                    self.trainidx.append(i)
                else:
                    self.testidx.append(i)
            else:
                if s == 'training':
                    self.trainidx.append(i)
                else:
                    self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category:
                        idx.append(i);
                        break;
            self.classwiseidx.append(idx)

    def get_feature_at_index(self, index: int):
        video_name = self.video_name[index]
        with open(os.path.join(self.path_to_annotations, "misc_list.txt"), 'r') as f:
            for line in f:
                line = line.split(' ')
                if line[1] == video_name:
                    path = line[0]
                    break
        RGB_feature = np.load(os.path.join(path, "{}_RGB.npy".format(video_name)))
        Flow_feature = np.load(os.path.join(path, "{}_Flow.npy".format(video_name)))
        if len(RGB_feature.shape) == 1:
            RGB_feature = np.expand_dims(RGB_feature, 0)
            Flow_feature = np.expand_dims(Flow_feature, 0)
        return np.concatenate([RGB_feature, Flow_feature], 1)

    def load_data(self, n_similar=3, is_training=True):
        if is_training == True:
            features = []
            labels = []
            idx = []

            # Load similar pairs
            rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
            for rid in rand_classid:
                rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
                idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                idx.append(self.classwiseidx[rid][rand_sampleid[1]])

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size - 2 * n_similar)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            return np.array([utils.process_feat(self.get_feature_at_index(i), self.t_max) for i in idx]), np.array(
                [self.labels_multihot[i] for i in idx])

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.get_feature_at_index(self.testidx[self.currenttestidx])

            if self.currenttestidx == len(self.testidx) - 1:
                done = True;
                self.currenttestidx = 0
            else:
                done = False;
                self.currenttestidx += 1

            return np.array(feat), np.array(labs), done
