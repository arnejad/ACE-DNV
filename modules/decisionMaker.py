import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x




def dataPrep(x,y):
    print("data prepration")



def train(model, x,y):
    print("training")


def validate(model, x,y):
    print("evaluating")


def execute(x,y):

    x, y = dataPrep(x,y)

    #define model

    train(x,y)

    validate(x,y)
    