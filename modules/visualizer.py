import matplotlib.pyplot as plt
from os import path, mkdir
from config import OUT_DIR


def panelInitializer():
    plt.rcParams["figure.figsize"] = [19, 20]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(4)
    return fig, axs

def panel(axs, nxtPatch, finalRes):
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[3].clear()
    axs[0].set_title('patch dist')
    axs[1].set_title('gaze distance')
    axs[2].set_title('env change')
    axs[3].set_title('gaze patch')
    axs[0].plot(finalRes[:,0])
    axs[1].plot(finalRes[:,1])
    axs[2].plot(finalRes[:,2])
    axs[3].imshow(nxtPatch)
    

def gazeCoordsPlotSave(gazes):
    fig,ax=plt.subplots(figsize=(20,5))
    plt.plot(gazes[:,1]) 
    plt.plot(gazes[:,2]) 

    if not path.exists(OUT_DIR): mkdir(OUT_DIR)
    if not path.exists(OUT_DIR+'/figs/'): mkdir(OUT_DIR+'/figs/')

    plt.savefig(OUT_DIR+'/figs/gazeCoords.png')
    plt.close()