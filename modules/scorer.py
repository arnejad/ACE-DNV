from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# function extracted from  https://github.com/elmadjian/OEMC 

def count_event(preds, gt):   
    event_preds = []
    event_gt = []
    i = 0
    while i < len(gt):
        g_0 = g_n = int(gt[i])
        ini, end = i, i
        while g_0 == g_n and i < len(gt):
            g_n = int(gt[i])
            end = i
            i += 1
        if ini == end:
            i += 1
            continue
        pred_event = np.array(preds[ini:end], dtype=int)
        event_preds.append(np.bincount(pred_event).argmax())
        event_gt.append(g_0)

    return event_preds, event_gt


def print_results(sample_preds, sample_gt, event_preds, event_gt):
    target_names = ["Fixation", "Gaze Pursuit", "Gaze Shift", "Gaze Following"]
    # target_names = ["Fixation", "Gaze Pursuit", "Gaze Shift"]
    # target_names = [ "Gaze Shift", "Gaze Following"]

    print('SAMPLE-LEVEL metrics\n===================')
    print(metrics.classification_report(sample_gt, sample_preds, digits=4))
    f1_ss = metrics.classification_report(sample_gt, sample_preds, digits=4, output_dict=True)
    metrics.ConfusionMatrixDisplay.from_predictions(sample_gt, 
                            sample_preds, display_labels=target_names, 
                            cmap='Purples', normalize='pred', values_format='.2f')
    print('EVENT-LEVEL metrics\n===================')
    print(metrics.classification_report(event_gt, event_preds, digits=4))
    metrics.ConfusionMatrixDisplay.from_predictions(event_gt,
                            event_preds, display_labels=target_names, 
                            normalize='pred', cmap='Greens', values_format='.2f')
    # plt.show()
    f1_se = metrics.classification_report(event_gt, event_preds, digits=4, output_dict=True)

    f1_e = [f1_se['0']['f1-score'], f1_se['1']['f1-score'], f1_se['2']['f1-score'], f1_se['3']['f1-score']]
    f1_s = [f1_ss['0']['f1-score'], f1_ss['1']['f1-score'], f1_ss['2']['f1-score'], f1_ss['3']['f1-score']]

    return f1_e, f1_s

def score(sample_preds, sample_gt):
    event_preds, event_gt = count_event(sample_preds, sample_gt)
    f1_e, f1_s = print_results(sample_preds, sample_gt, event_preds, event_gt)
    return f1_e, f1_s