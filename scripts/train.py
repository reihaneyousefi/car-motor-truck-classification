import torch
from data_setup import creat_dataset
from model import  EnsembleModel_overfit , MobileNetModel , ResNetModel
import torchvision
from torchvision import models
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils import  get_accuracy , save_model
from time import time
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model import *
import os





NUM_CLASS =3
INPUT_CHANNELS = 3
NUM_EPOCHS = 15

device = "cuda" if torch.cuda.is_available() else "cpu"

start = time()

# batch_sizes = [4 , 8]
learning_rates =[0.001 ]

modela = MobileNetModel().to(device)
modelb = ResNetModel().to(device)

models = [["ensembleOverFit",EnsembleModel_overfit(modela ,modelb, num_classes=NUM_CLASS)] ]

for model_cnf in models:
    for learning_rate in learning_rates:
        step = 0
        traindata , testdata , valdata = creat_dataset(path_train=r"cmc\train" , path_test=r"cmc\test" , path_val=r"cmc\val" , BATCH_SIZE=8 , imbalance=True)


        model = model_cnf[1].to(device)

        writer = SummaryWriter(f"runs/{model_cnf[0]}")

        loss_fun = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        Scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt , factor = 0.2 , patience=10 , verbose = True)

        img_b , target_b = next(iter(traindata))

        writer.add_graph(model , img_b.to(device))
        
        grid = torchvision.utils.make_grid(img_b)
        writer.add_image('images', grid, 0)

        train_loader = traindata

        for epoch in tqdm(range(NUM_EPOCHS)):
            losses = []
            start_Epoch = time()

            model.train()
            lossBatch = 0

            for index_batch, (img,target) in enumerate(traindata):
                opt.zero_grad()
                img = img.to(device)
                                
                target = target.to(device)
                pred = model(img)

                loss = loss_fun(pred , target)
                losses.append(loss.item())
                lossBatch += loss

               
                loss.backward()
                opt.step()
                
                if (index_batch +1) % 200 == 0:
                    meanloss = lossBatch / 200
                    print(f"loss for this batch {index_batch + 1} = {meanloss}  ") 
                    Scheduler.step(meanloss)
                    lossBatch = 0
                

                # _, predictions = pred.max(1)
                # num_correct = (predictions == target).sum()
                # running_train_acc = float(num_correct)/float(img.shape[0])
            end_Epoch = time()
            writer.add_scalar("epochTime" , round(end_Epoch-start_Epoch , 3) , global_step=step)
            model.eval()
            ac_train  = get_accuracy(traindata , model ,  device)
            ac_test  = get_accuracy(testdata , model , device)

            # accuracies.append(running_train_acc)
            writer .add_scalar('Training Loss',sum(losses)/len(losses),global_step=step)#change loss into losses
            writer.add_scalar('training Accuracy',ac_train, global_step=step)
            writer.add_scalar('test Accuracy',ac_test, global_step=step)
            
            step +=1
            writer.add_hparams({'lr':learning_rate ,'bsize':8},
                                {'accuracy':ac_train,
                                        # 'accuracy':sum
                                        # (accuracies)/len(accuracies),
                                'loss':sum(losses)/len(losses)})


            ac_val  = get_accuracy(valdata , model ,  device)
            print(f"accuracy for train {ac_train} , accuracy for test {ac_test} , accuracy for val {ac_val}  ")


            print(f" time for training {epoch+1} epoch {end_Epoch-start_Epoch:.3f} sec")

            save_model(model , opt , epoch , f"weights/model_{model_cnf[0]}.pt")
                
writer.close()

end = time()
print(f"whole time for training {end-start:.3f}sec")

