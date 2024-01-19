from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from config import EXP, VISUALIZE

font = {'size': 16}


# function extracted from  https://github.com/elmadjian/OEMC 


def count_event(preds, gt):   
    # note: the function has a bug that it does not consider events with length of one sample and skips them. In our work, 
    # the events of length one in the ground-truth belong to the flags of removed events. So, this code works fine for us. But might not be 
    # completey accurate for other work.

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
    target_names = ["GFi", "GP", "GS", "GFo"]
    # target_names = ["Fixation", "Gaze Pursuit", "Gaze Shift"]
    if (EXP == "1-1" or EXP == "1-3"):
        target_names = [ "GS", "GFo"]
    elif(EXP == "1-2"):
        target_names = ["GFi", "GP", "GS"]
    elif (EXP == "2-2"):
        target_names = ["GFi+GFo", "GP", "GS"]

    
    print('SAMPLE-LEVEL metrics\n===================')
    print(metrics.classification_report(sample_gt, sample_preds, digits=4))
    scores_s = metrics.classification_report(sample_gt, sample_preds, digits=4, output_dict=True, zero_division=0)
    if VISUALIZE:
        metrics.ConfusionMatrixDisplay.from_predictions(sample_gt, 
                                sample_preds, display_labels=target_names, 
                                cmap='Purples', normalize='pred', values_format='.2f', text_kw=font)
    
    print('EVENT-LEVEL metrics\n===================')
    print(metrics.classification_report(event_gt, event_preds, digits=4))
    if VISUALIZE:
        metrics.ConfusionMatrixDisplay.from_predictions(event_gt,
                                event_preds, display_labels=target_names, 
                                normalize='pred', cmap='Greens', values_format='.2f', text_kw=font)
    # # plt.show()
    scores_e = metrics.classification_report(event_gt, event_preds, digits=4, output_dict=True, zero_division=0)

    # TODO There is definately better way to seporate the scores instead of this mess of ifs and elses
    if (EXP == "1-1" or EXP == "1-3"):

        f1_e = [scores_e['0']['f1-score'], scores_e['1']['f1-score'], scores_e['macro avg']['f1-score'], scores_e['weighted avg']['f1-score']]
        f1_s = [scores_s['0']['f1-score'], scores_s['1']['f1-score'], scores_s['macro avg']['f1-score'], scores_s['weighted avg']['f1-score']]
        supp_e = [scores_e['0']['support'], scores_e['1']['support'], scores_e['macro avg']['support'], scores_e['weighted avg']['support']]
        supp_s = [scores_s['0']['support'], scores_s['1']['support'], scores_s['macro avg']['support'], scores_s['weighted avg']['support']]
    elif (EXP == "2-2"):
        f1_e = [scores_e['0']['f1-score'], scores_e['1']['f1-score'], scores_e['2']['f1-score'], scores_e['macro avg']['f1-score'], scores_e['weighted avg']['f1-score']]
        f1_s = [scores_s['0']['f1-score'], scores_s['1']['f1-score'], scores_s['2']['f1-score'], scores_s['macro avg']['f1-score'], scores_s['weighted avg']['f1-score']]
        supp_e = [scores_e['0']['support'], scores_e['1']['support'], scores_e['2']['support'], scores_e['macro avg']['support'], scores_e['weighted avg']['support']]
        supp_s = [scores_s['0']['support'], scores_s['1']['support'], scores_s['2']['support'], scores_s['macro avg']['support'], scores_s['weighted avg']['support']]
    elif (EXP=="1-2"):
        f1_e = [scores_e['0']['f1-score'], scores_e['1']['f1-score'], scores_e['2']['f1-score'], scores_e['macro avg']['f1-score'], scores_e['weighted avg']['f1-score']]
        f1_s = [scores_s['0']['f1-score'], scores_s['1']['f1-score'], scores_s['2']['f1-score'], scores_s['macro avg']['f1-score'], scores_s['weighted avg']['f1-score']]
        supp_e = [scores_e['0']['support'], scores_e['1']['support'], scores_e['2']['support'], scores_e['macro avg']['support'], scores_e['weighted avg']['support']]
        supp_s = [scores_s['0']['support'], scores_s['1']['support'], scores_s['2']['support'], scores_s['macro avg']['support'], scores_s['weighted avg']['support']]
    else:    
        f1_e = [scores_e['0']['f1-score'], scores_e['1']['f1-score'], scores_e['2']['f1-score'], scores_e['3']['f1-score'], scores_e['macro avg']['f1-score'], scores_e['weighted avg']['f1-score']]
        f1_s = [scores_s['0']['f1-score'], scores_s['1']['f1-score'], scores_s['2']['f1-score'], scores_s['3']['f1-score'], scores_s['macro avg']['f1-score'], scores_s['weighted avg']['f1-score']]
        supp_e = [scores_e['0']['support'], scores_e['1']['support'], scores_e['2']['support'], scores_e['3']['support'], scores_e['macro avg']['support'], scores_e['weighted avg']['support']]
        supp_s = [scores_s['0']['support'], scores_s['1']['support'], scores_s['2']['support'], scores_s['3']['support'], scores_s['macro avg']['support'], scores_s['weighted avg']['support']]

    return f1_e, f1_s, supp_e, supp_s

def score(sample_preds, sample_gt):
    event_preds, event_gt = count_event(sample_preds, sample_gt)
    f1_e, f1_s, supp_e, supp_s = print_results(sample_preds, sample_gt, event_preds, event_gt)
    return f1_e, f1_s, supp_e, supp_s