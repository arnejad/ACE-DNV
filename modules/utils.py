import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import utils


def print_scores(total_pred, total_label, test_loss, name):
    f1_fix = f1_score(total_pred, total_label, 0)*100
    f1_gazeP = f1_score(total_pred, total_label, 1)*100
    f1_sacc = f1_score(total_pred, total_label, 2)*100
    f1_gazeF = f1_score(total_pred, total_label, 3)*100
    f1_avg = (f1_fix + f1_sacc + f1_gazeP + f1_gazeF)/4
    print('\n{} set: Average loss: {:.4f}, F1_FIX: {:.2f}%, F1_SACC: {:.2f}%, F1_GazePursuit: {:.2f}%, F1_GazeFollowing: {:.2f}%, AVG: {:.2f}%\n'.format(
        name, test_loss, f1_fix, f1_sacc, f1_gazeP, f1_gazeF, f1_avg
    ))
    return (f1_fix + f1_sacc + f1_gazeP + f1_gazeF)/4



def f1_score(preds, labels, class_id):
    '''
    preds: precictions made by the network
    labels: list of expected targets
    class_id: corresponding id of the class
    '''
    true_count = torch.eq(labels, class_id).sum()
    true_positive = torch.logical_and(torch.eq(labels, preds),
                                      torch.eq(labels, class_id)).sum().float()
    precision = torch.div(true_positive, torch.eq(preds, class_id).sum().float())
    precision = torch.where(torch.isnan(precision),
                            torch.zeros_like(precision).type_as(true_positive),
                            precision)
    recall = torch.div(true_positive, true_count)
    f1 = 2*precision*recall / (precision+recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive),f1)
    return f1.item()


def finalReportPrint(exp, f1_samp_avg, f1_event_avg):
    print("Sample-level F1-Score weighted average:")
    if exp == "2-2":
        print("GFi+GFo: " + str(f1_samp_avg[0]) + " GP: " + str(f1_samp_avg[1]) + " GS: " + str(f1_samp_avg[2]) + " W AVG: " + str(f1_samp_avg[4]))
    elif (exp == "1-1" or exp == "1-3"):
        print("GS: " + str(f1_samp_avg[0]) + " GFo: " + str(f1_samp_avg[1]) + " W AVG: " + str(f1_samp_avg[3]))
    elif exp == "1-2":
        print("GFi: " + str(f1_samp_avg[0]) + " GP: " + str(f1_samp_avg[1]) + " GS: " + str(f1_samp_avg[2]) +  " W AVG: " + str(f1_samp_avg[4]))
    else:
        print("GFi: " + str(f1_samp_avg[0]) + " GP: " + str(f1_samp_avg[1]) + " GS: " + str(f1_samp_avg[2]) + " GFo: " + str(f1_samp_avg[3]) + " W AVG: " + str(f1_samp_avg[5]))


    print("Event-level F1-Score weighted average:")
    if exp == "2-2":
        print("GFi+GFo: " + str(f1_event_avg[0]) + " GP: " + str(f1_event_avg[1]) + " GS: " + str(f1_event_avg[2]) + " W AVG: " + str(f1_event_avg[4]))
    elif (exp == "1-1" or exp == "1-3"):
        print("GS: " + str(f1_event_avg[0]) + " GFo: " + str(f1_event_avg[1]) + " W AVG: " + str(f1_event_avg[3]))
    elif exp == "1-2":
        print("GFi: " + str(f1_event_avg[0]) + " GP: " + str(f1_event_avg[1]) + " GS: " + str(f1_event_avg[2]) +  " W AVG: " + str(f1_event_avg[4]))
    else:
        print("GFi: " + str(f1_event_avg[0]) + " GP: " + str(f1_event_avg[1]) + " GS: " + str(f1_event_avg[2]) + " GFo: " + str(f1_event_avg[3]) + " W AVG: " + str(f1_event_avg[5]))
