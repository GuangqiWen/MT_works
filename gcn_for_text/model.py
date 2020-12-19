import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn(nn.Module):
    def __init__(self, X_size, A_hat, args, bias=True): 
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=True).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, 512))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        #self.weight.data.normal_(0,var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(512, 256))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0)) 
        self.weight2.data.normal_(0,var2)
      
        self.bias = nn.parameter.Parameter(torch.FloatTensor(512))
        self.bias.data.normal_(0,var)
        self.bias2 = nn.parameter.Parameter(torch.FloatTensor(256))
        self.bias2.data.normal_(0,var2)
        
        # 总是过拟合 试试dropout
        self.dropout = nn.Dropout(p=0.2) 
        # 一层FC
        #self.fc0 = nn.Linear(300, 100)
        self.fc1 = nn.Linear(256, args.num_classes)  
       
    def forward(self, X): 
        X = torch.mm(X, self.weight)        
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))
        #self.dropout(X)
        X = torch.mm(X, self.weight2)
        self.dropout(X)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = F.relu(torch.mm(self.A_hat, X))
        #self.dropout(X)
        #print(X.shape)
        #X = self.fc0(X)
        return self.dropout(self.fc1(X))
