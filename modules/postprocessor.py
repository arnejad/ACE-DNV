from cmath import inf
from config import POSTPROCESS_WIN_SIZE

def postprocess(events):
    halfWin = int(POSTPROCESS_WIN_SIZE/2)
    res = list(events)
    for i in range(halfWin, len(events)-halfWin):
        #create histogram of existing events
        if events[i] != 'PotSAC':
            counts = []
            names = []
            for j in range(i-halfWin, i+halfWin):
                if events[j] in names:
                    indx = names.index(events[j])
                    counts[indx] = counts[indx] + 1

                else:
                    names.append(events[j])
                    counts.append(1)

            # check if the repetitions of the current is event is higher than the others
            if counts[names.index(events[i])] == max(counts): #avoid choosing the other event by accident if both were max
                res[i] = events[i]
            else:
                ind = counts.index(max(counts))
                res[i] = names[ind]

    return res