import numpy as np
import time
from scipy.signal import savgol_filter
import sys
import scipy.io as sio
import os


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]

def get_fps(video_name):
    fps = 30
    with open(os.path.join('ActivityNet1.3-Annotations', 'misc_fps.txt'), 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            if line[0] == video_name:
                fps = float(line[1])
                break
    return fps


def smooth(v):
    # return v
    l = min(351, len(v));
    l = l - (1 - l % 2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 1)  # savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)


def filter_segments(segment_predict, videonames, ambilist):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        fps = get_fps(vn)
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) * fps / 16)), int(round(float(a[3]) * fps / 16)))
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))
                if IoU > 0:
                    ind[i] = 1
    s = [segment_predict[i, :] for i in range(np.shape(segment_predict)[0]) if ind[i] == 0]
    return np.array(s)


def getLocMAP(predictions, th, annotation_path, args):
    train_set = 'validation' if 'thumos' in annotation_path.lower() else 'training'
    val_set = 'test' if 'thumos' in annotation_path.lower() else 'validation'
    gtsegments = np.load(annotation_path + '/segments.npy', allow_pickle=True)
    # gtlabels = np.load(annotation_path + '/labels.npy', allow_pickle=True)
    gtlabels = np.load(annotation_path + '/labels.npy', allow_pickle=True)
    videoname = np.load(annotation_path + '/videoname.npy', allow_pickle=True);
    videoname = np.array([v for v in videoname])
    subset = np.load(annotation_path + '/subset.npy', allow_pickle=True);
    subset = np.array([s for s in subset])
    classlist = np.load(annotation_path + '/classlist.npy', allow_pickle=True);
    classlist = np.array([c for c in classlist])
    duration = np.load(annotation_path + '/duration.npy', allow_pickle=True)
    ambilist = annotation_path + '/Ambiguous_test.txt'
    if args.feature_type == 'UNT':
        factor = 10.0 / 4.0
    else:
        factor = 27.6 / 16.0

    # TODO

    ambilist = list(open(ambilist, 'r'))
    ambilist = [a.strip('\n').split(' ') for a in ambilist]

    # keep training gtlabels for plotting
    gtltr = []
    for i, s in enumerate(subset):
        if subset[i] == train_set and len(gtsegments[i]):
            gtltr.append(gtlabels[i])
    gtlabelstr = gtltr

    # Keep only the test subset annotations
    gts, gtl, vn, dn = [], [], [], []
    for i, s in enumerate(subset):
        if subset[i] == val_set:
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            dn.append(duration[i])
    gtsegments = gts
    gtlabels = gtl
    videoname = vn
    duration = dn

    # keep ground truth and predictions for instances with temporal annotations
    gts, gtl, vn, pred, dn = [], [], [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            pred.append(predictions[i])
            dn.append(duration[i])
    gtsegments = gts
    gtlabels = gtl
    videoname = vn
    predictions = pred

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, classlist))

    # process the predictions such that classes having greater than a certain threshold are detected only
    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = - p;
        [pp[:, i].sort() for i in range(np.shape(pp)[1])];
        pp = -pp
        c_s = np.mean(pp[:int(np.shape(pp)[0] / 8), :], axis=0)
        ind = c_s > 0.0
        c_score.append(c_s)
        new_pred = np.zeros((np.shape(p)[0], np.shape(p)[1]), dtype='float32')
        predictions_mod.append(p * ind)
    predictions = predictions_mod

    detection_results = []
    for i, vn in enumerate(videoname):
        detection_results.append([])
        detection_results[i].append(vn)

    ap = []
    for c in templabelidx:
        segment_predict = []
        # Get list of all predictions for class c
        for i in range(len(predictions)):
            tmp = smooth(predictions[i][:, c])
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * 0.9
            vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                aggr_score = np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]
                if e[j] - s[j] >= 2:
                    segment_predict.append([i, s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
                    detection_results[i].append(
                        [classlist[c], s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
        segment_predict = np.array(segment_predict)
        segment_predict = filter_segments(segment_predict, videoname, ambilist)

        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        # Create gt list
        segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in
                      range(len(gtsegments[i])) if str2ind(gtlabels[i][j], classlist) == c]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            flag = 0.
            for j in range(len(segment_gt)):
                vn = videoname[int(segment_gt[j][0])]
                fps = get_fps(vn)
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(int(round(segment_gt[j][1] * fps / 16)), int(round(segment_gt[j][2] * fps / 16)))
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(len(set(gt).union(set(p))))
                    if IoU >= th:
                        flag = 1.
                        del segment_gt[j]
                        break
            tp.append(flag)
            fp.append(1. - flag)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp) == 0:
            prc = 0.
        else:
            prc = np.sum((tp_c / (fp_c + tp_c)) * tp) / gtpos
        ap.append(prc)

    return 100 * np.mean(ap)


def getLocMAPs(predictions, iou_thresholds, annotation_path, args):
    train_set = 'validation' if 'thumos' in annotation_path.lower() else 'training'
    val_set = 'test' if 'thumos' in annotation_path.lower() else 'validation'
    gtsegments = np.load(annotation_path + '/segments.npy', allow_pickle=True)
    # gtlabels = np.load(annotation_path + '/labels.npy', allow_pickle=True)
    gtlabels = np.load(annotation_path + '/labels.npy', allow_pickle=True)
    videoname = np.load(annotation_path + '/videoname.npy', allow_pickle=True);
    videoname = np.array([v for v in videoname])
    subset = np.load(annotation_path + '/subset.npy', allow_pickle=True);
    subset = np.array([s for s in subset])
    classlist = np.load(annotation_path + '/classlist.npy', allow_pickle=True);
    classlist = np.array([c for c in classlist])
    duration = np.load(annotation_path + '/duration.npy', allow_pickle=True)
    ambilist = annotation_path + '/Ambiguous_test.txt'
    if args.feature_type == 'UNT':
        factor = 10.0 / 4.0
    else:
        factor = 27.6 / 16.0


    ambilist = list(open(ambilist, 'r'))
    ambilist = [a.strip('\n').split(' ') for a in ambilist]

    # keep training gtlabels for plotting
    gtltr = []
    for i, s in enumerate(subset):
        if subset[i] == train_set and len(gtsegments[i]):
            gtltr.append(gtlabels[i])
    gtlabelstr = gtltr

    # Keep only the test subset annotations
    gts, gtl, vn, dn = [], [], [], []
    for i, s in enumerate(subset):
        if subset[i] == val_set:
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            dn.append(duration[i])
    gtsegments = gts
    gtlabels = gtl
    videoname = vn
    duration = dn

    # keep ground truth and predictions for instances with temporal annotations
    gts, gtl, vn, pred, dn = [], [], [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            pred.append(predictions[i])
            dn.append(duration[i])
    gtsegments = gts
    gtlabels = gtl
    videoname = vn
    predictions = pred

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, classlist))

    # process the predictions such that classes having greater than a certain threshold are detected only
    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = - p;
        [pp[:, i].sort() for i in range(np.shape(pp)[1])];
        pp = -pp
        c_s = np.mean(pp[:int(np.shape(pp)[0] / 8), :], axis=0)
        ind = c_s > 0.0
        c_score.append(c_s)
        new_pred = np.zeros((np.shape(p)[0], np.shape(p)[1]), dtype='float32')
        predictions_mod.append(p * ind)
    predictions = predictions_mod

    detection_results = []
    for i, vn in enumerate(videoname):
        detection_results.append([])
        detection_results[i].append(vn)

    # ap = []
    ap_list = [[] for _ in range(len(iou_thresholds))]
    for c in templabelidx:
        segment_predict = []
        # Get list of all predictions for class c
        for i in range(len(predictions)):
            tmp = smooth(predictions[i][:, c])
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * 1.0
            vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                aggr_score = np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]
                if e[j] - s[j] >= 2:
                    segment_predict.append([i, s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
                    detection_results[i].append(
                        [classlist[c], s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
        segment_predict = np.array(segment_predict)
        segment_predict = filter_segments(segment_predict, videoname, ambilist)

        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in
                      range(len(gtsegments[i])) if str2ind(gtlabels[i][j], classlist) == c]
        gtpos = len(segment_gt)

        # Compare predictions and gt

        # tp, fp = [], []
        tp_list = [[] for _ in range(len(iou_thresholds))]
        fp_list = [[] for _ in range(len(iou_thresholds))]
        for i in range(len(segment_predict)):
            flag = 0.
            flag_list = [0 for _ in range(len(iou_thresholds))]
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(int(round(segment_gt[j][1] * factor)), int(round(segment_gt[j][2] * factor)))
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(len(set(gt).union(set(p))))
                    if IoU >= iou_thresholds[0]:
                        flag = 1.
                        flag_list[0] = 1.
                        for k in range(1, len(iou_thresholds)):
                            if IoU >= iou_thresholds[k]:
                                flag_list[k] = 1
                    if flag == 1:
                        del segment_gt[j]
                        break
            for j in range(len(tp_list)):
                tp_list[j].append(flag_list[j])
                fp_list[j].append(1. - flag_list[j])
            # tp.append(flag)
            # fp.append(1. - flag)
        tp_c = [np.cumsum(tp) for tp in tp_list]
        fp_c = [np.cumsum(fp) for fp in fp_list]
        for i in range(len(tp_list)):

            if sum(tp_c[i]) == 0:
                prc = 0.
            else:
                prc = np.sum((tp_c[i] / (fp_c[i] + tp_c[i])) * tp_list[i]) / gtpos
            ap_list[i].append(prc)

    return [100 * np.mean(ap) for ap in ap_list]


def getDetectionMAP(predictions, annotation_path, args):
    iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # iou_list = [0.5]
    dmap_list = getLocMAPs(predictions, iou_list, annotation_path, args)
    # dmap_list = []
    # for iou in iou_list:
    #     print('Testing for IoU %f' % iou)
    #     dmap_list.append(getLocMAP(predictions, iou, annotation_path, args))

    return dmap_list, iou_list
