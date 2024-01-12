import torch
import numpy as np
# from DeepModels.loss import getPerformance
from DeepModels.torchtools import EarlyStopping, modVal
from scorer import score

def train(net, trainloader, validloader, testloader, TBwriter, args):
    cond = EarlyStopping(patience=75, mode='max', delta=1e-2, path2save=args.path2save, verbose=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

    validTrack = trackPerf()
    trainTrack = trackPerf()
    for eps in range(0, args.epochs):
        net.train()
        Y = []
        L = []
        GT= []
        ceL = []
        gDL = []
        for bt, data in enumerate(trainloader):
            optimizer.zero_grad()
            train_data, W, target = data
            y, loss, loss_ce, loss_dl = net(train_data.cuda().to(args.prec),
                                            target.cuda().to(args.prec),
                                            W.cuda().to(args.prec))
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()

            L.append(loss.detach().cpu().item())
            ceL.append(loss_ce)
            gDL.append(loss_dl)
            pd = np.argmax(y.cpu().detach().numpy(), axis=2)
            gt = target.cpu().detach().numpy()
            GT.append(gt.reshape(-1))
            Y.append(pd.reshape(-1))

        perf = getPerformance(np.hstack(GT), np.hstack(Y), calc_evt=False)
        perf['loss'] = np.mean(L)
        perf['loss_ce'] = np.mean(ceL)
        perf['loss_dl'] = np.mean(gDL)
        trainTrack.addEntry(eps, 0, perf)

        # Reduce learning rate linearly as training progresses
        for param_group in optimizer.param_groups:
            param_group['lr'] = modVal(trainTrack.getPerf(eps, 'kappa'), [0, 1], [args.lr, 1e-2*args.lr], 'linear')

        if validloader:
            # Valid evaluation
            net.eval()
            with torch.no_grad():
                Y = []
                L = []
                GT= []
                ceL = []
                gDL = []
                for bt, data in enumerate(validloader):
                    valid_data, W, target = data
                    y, loss, loss_ce, loss_dl = net(valid_data.cuda().to(args.prec),
                                                    target.cuda().to(args.prec),
                                                    W.cuda().to(args.prec))
                    pd = np.argmax(y.cpu().detach().numpy(), axis=2)
                    gt = target.cpu().detach().numpy()
                    Y.append(pd.reshape(-1))
                    GT.append(gt.reshape(-1))
                    L.append(loss.detach().cpu().numpy())
                    ceL.append(loss_ce)
                    gDL.append(loss_dl)

                scr_e, scr_s = score(np.hstack(GT), np.hstack(Y))
                # perf = getPerformance(np.hstack(GT), np.hstack(Y), calc_evt=False)
                # perf['loss'] = np.mean(L)
                # perf['loss_ce'] = np.mean(ceL)
                # perf['loss_dl'] = np.mean(gDL)
                # validTrack.addEntry(eps, 0, perf)

            # Record best model
            cond(eps, validTrack.getPerf(eps, 'kappa'),
                 net.state_dict() if torch.cuda.device_count() == 1 else net.module.state_dict())

            # Increase delta condition for model evaluation. This ensures that
            # only large changes in validation metric warrants a testing update
            cond.delta = modVal(validTrack.getPerf(eps, 'kappa'), [0, 1], [1e-2, 3e-2], 'linear')

            print('eps: {}. k: {}. p: {}. r: {}. k_evt: {}'.format(
                    eps,
                    validTrack.getPerf(eps, 'kappa'),
                    validTrack.getPerf(eps, 'prec'),
                    validTrack.getPerf(eps, 'recall'),
                    validTrack.getPerf(eps, 'kappa_evt')))

            # Calculate test performance
            perf_test = test(net, testloader, args, cond.update_flag)[1]

            # Update tensorboard
            update_tensorboard(TBwriter, trainTrack, validTrack, perf_test, eps)
        else:
            # Calculate test performance
            perf_test = test(net, testloader, args, 0)[1]

            # No testing set. Validate on testing subject.
            cond(eps, perf_test.getPerf(0, 'kappa'),
                 net.state_dict() if torch.cuda.device_count() == 1 else net.module.state_dict())

            print('eps: {}. k: {}. p: {}. r: {}. k_evt: {}'.format(
                    eps,
                    perf_test.getPerf(0, 'kappa'),
                    perf_test.getPerf(0, 'prec'),
                    perf_test.getPerf(0, 'recall'),
                    perf_test.getPerf(0, 'kappa_evt')))

            # Update tensorboard
            update_tensorboard(TBwriter, trainTrack, [], perf_test, eps)
        if cond.early_stop:
            break

    TBwriter.close()
    return validTrack, cond.best_model

def update_tensorboard(TBwriter, trainTrack, validTrack, testTrack, eps):
    if validTrack:
        TBwriter.add_scalars('loss', {'train': trainTrack.getPerf(eps, 'loss'),
                                      'valid': validTrack.getPerf(eps, 'loss'),
                                      'test': testTrack.getPerf(0, 'loss')}, eps)

        TBwriter.add_scalars('loss/CE', {'train': trainTrack.getPerf(eps, 'loss_ce'),
                                      'valid': validTrack.getPerf(eps, 'loss_ce'),
                                      'test': testTrack.getPerf(0, 'loss_ce')}, eps)

        TBwriter.add_scalars('loss/gDL', {'train': trainTrack.getPerf(eps, 'loss_dl'),
                                      'valid': validTrack.getPerf(eps, 'loss_dl'),
                                      'test': testTrack.getPerf(0, 'loss_dl')}, eps)

        TBwriter.add_scalars('kappa', {'train': trainTrack.getPerf(eps, 'kappa'),
                                       'valid': validTrack.getPerf(eps, 'kappa'),
                                       'test': testTrack.getPerf(0, 'kappa')}, eps)

        TBwriter.add_scalars('iou', {'train': trainTrack.getPerf(eps, 'iou'),
                                       'valid': validTrack.getPerf(eps, 'iou'),
                                       'test': testTrack.getPerf(0, 'iou')}, eps)

        TBwriter.add_scalars('iou/fix', {'train': trainTrack.getPerf(eps, 'iou_class')[0],
                                       'valid': validTrack.getPerf(eps, 'iou_class')[0],
                                       'test': testTrack.getPerf(0, 'iou_class')[0]}, eps)

        TBwriter.add_scalars('iou/pur', {'train': trainTrack.getPerf(eps, 'iou_class')[1],
                                       'valid': validTrack.getPerf(eps, 'iou_class')[1],
                                       'test': testTrack.getPerf(0, 'iou_class')[1]}, eps)

        TBwriter.add_scalars('iou/sac', {'train': trainTrack.getPerf(eps, 'iou_class')[2],
                                       'valid': validTrack.getPerf(eps, 'iou_class')[2],
                                       'test': testTrack.getPerf(0, 'iou_class')[2]}, eps)
    else:
        TBwriter.add_scalars('loss', {'train': trainTrack.getPerf(eps, 'loss'),
                                      'test': testTrack.getPerf(0, 'loss')}, eps)

        TBwriter.add_scalars('loss/CE', {'train': trainTrack.getPerf(eps, 'loss_ce'),
                                      'test': testTrack.getPerf(0, 'loss_ce')}, eps)

        TBwriter.add_scalars('loss/gDL', {'train': trainTrack.getPerf(eps, 'loss_dl'),
                                      'test': testTrack.getPerf(0, 'loss_dl')}, eps)

        TBwriter.add_scalars('kappa', {'train': trainTrack.getPerf(eps, 'kappa'),
                                       'test': testTrack.getPerf(0, 'kappa')}, eps)

        TBwriter.add_scalars('iou', {'train': trainTrack.getPerf(eps, 'iou'),
                                       'test': testTrack.getPerf(0, 'iou')}, eps)

        TBwriter.add_scalars('iou/fix', {'train': trainTrack.getPerf(eps, 'iou_class')[0],
                                       'test': testTrack.getPerf(0, 'iou_class')[0]}, eps)

        TBwriter.add_scalars('iou/pur', {'train': trainTrack.getPerf(eps, 'iou_class')[1],
                                       'test': testTrack.getPerf(0, 'iou_class')[1]}, eps)

        TBwriter.add_scalars('iou/sac', {'train': trainTrack.getPerf(eps, 'iou_class')[2],
                                       'test': testTrack.getPerf(0, 'iou_class')[2]}, eps)
    return []

def test(net, testloader, args, talk=False):
    '''
    Testing script. Since testing without the ground truth for any class
    produces erroneous results, we concat all trials belonging to a person to
    evaluate testing performance.
    '''
    testTrack = trackPerf()
    net.eval()
    Y = []
    L = []
    GT= []
    ID = []
    ceL = []
    gDL = []
    with torch.no_grad():
        for seqIdx, data in enumerate(testloader):
            test_data, W, target, id_trx = data
            y = net(test_data.cuda().to(args.prec),
                                            target.cuda().to(args.prec),
                                            W.cuda().to(args.prec))
            pd = np.argmax(y.cpu().detach().numpy(), axis=2)
            gt = target.cpu().detach().numpy()
            Y.append(pd.reshape(-1))
            GT.append(gt.reshape(-1))
            ID.append(id_trx.numpy().reshape(-1))
        f1_e, f1_s, supp_e, supp_s = score(np.hstack(GT), np.hstack(Y))
        # perf = getPerformance(np.hstack(GT), np.hstack(Y), calc_evt=False)
        # perf['loss'] = np.mean(L)
        # perf['loss_ce'] = np.mean(ceL)
        # perf['loss_dl'] = np.mean(gDL)
        # validTrack.addEntry(eps, 0, perf)

    return f1_e, f1_s, supp_e, supp_s

class trackPerf():
    def __init__(self):
        self.epoch = []
        self.bt = []
        self.loss = []
        self.loss_ce = []
        self.loss_dl = []
        self.kappa = []
        self.kappa_evt = []
        self.kappa_class_evt = []
        self.iou = []
        self.iou_class = []
        self.prec = []
        self.recall = []
        self.perf_list = []

    def addEntry(self, eps, bt, perf):
        self.epoch.append(eps)
        self.bt.append(bt)
        self.loss.append(perf['loss'])
        self.loss_ce.append(perf['loss_ce'])
        self.loss_dl.append(perf['loss_dl'])
        self.kappa.append(perf['kappa'])
        self.kappa_evt.append(perf['kappa_evt'])
        self.kappa_class_evt.append(perf['kappa_class_evt'])
        self.iou.append(perf['iou'])
        self.iou_class.append(perf['iou_class'])
        self.prec.append(perf['prec'])
        self.recall.append(perf['recall'])

    def getPerf(self, eps, attr):
        listeps = np.array(self.epoch)
        loc = listeps == eps
        temp = getattr(self, attr)
        temp = np.stack(temp, axis=0)
        temp = temp[loc, np.newaxis] if len(temp.shape) == 1 else temp[loc, :]
        op = np.mean(temp, axis=0)
        op = np.ma.round(op, 3)
        return op