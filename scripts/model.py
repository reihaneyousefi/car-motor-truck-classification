
import torch
import torch.nn as nn
import torch.functional as F

# import matplotlib.pyplot as plt
# from torchviz import make_dot
from torchsummary import summary
import torchvision.models as models


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class modelResnet(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(modelResnet , self).__init__() 
        self.resnet = models.resnet50(pretrained = True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_featu = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_featu,num_class)
    

    
    def forward(self,x):
        return self.resnet(x)
    

class modelMobilenet(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(modelMobilenet , self).__init__() 
        self.mobilenet = models.mobilenet_v2(pretrained = True)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        self.mobilenet.classifier[-1] = nn.Linear(self.mobilenet.classifier[-1].in_features, num_class)
        
    
    def forward(self,x):
        return self.mobilenet(x)
    


class MyEnsemble(nn.Module):
    def __init__(self , modelA , modelB , num_Class = 6):
        super(MyEnsemble,self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.fc1 = nn.Linear(num_Class*2 , num_Class)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)

        out = torch.cat((out1 , out2),dim=1)

        x = self.fc1(out)
        return x






class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
                
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)

    def forward(self, x):
        return self.resnet(x)

class MobileNetModel(nn.Module):
    def __init__(self):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
                
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, 1024)

    def forward(self, x):
        return self.mobilenet(x)


class EnsembleModel_overfit(nn.Module):
    def __init__(self, resnet_model, mobilenet_model, num_classes=3):
        super(EnsembleModel_overfit, self).__init__()
        self.resnet_model = resnet_model
        self.mobilenet_model = mobilenet_model
        self.fc =  nn.Sequential(
                                    nn.LayerNorm(2048),  
                                    nn.Dropout(0.5),    
                                    
                                    nn.Linear(2048, 1024), 
                                    nn.ReLU(inplace=True),
                                    nn.LayerNorm(1024),
                                    nn.Dropout(0.5),
                                    
                                    nn.Linear(1024, 512),        
                                    nn.ReLU(inplace=True),
                                    nn.LayerNorm(512),
                                    nn.Dropout(0.5),
                                    
                                    nn.Linear(512, num_classes)  
                                )


    def forward(self, x):
        resnet_features = self.resnet_model(x)
        mobilenet_features = self.mobilenet_model(x)
        features = torch.cat((resnet_features, mobilenet_features), dim=1)
        output = self.fc(features)
        return output





if '__main__' == __name__:
    
    img = torch.rand(size=(1 ,3,150,150))

    # model = model_ReLU( )
    # model = modelResnet()
    # model = modelMobilenet()
    # print(model)
    # print(model(img).shape)
    modela = MobileNetModel()
    modelb = ResNetModel()
    model = EnsembleModel_overfit(modela , modelb)
    out = model(img)
    print(out.shape)

    # summary(model.to(device) ,input_size= (3,150,150))
    # summary(model.to(device) ,input_size= (3,150,150))

