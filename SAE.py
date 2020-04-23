import os
import time
import glob
import torch

from utils.constants import *
from utils.NeuralNetwork import *
from utils.Plot import *
from utils.basic import *

def train_autocoder(optimizer, criterion, net, device, 
                    save_path=SAVEPATH, epoch_num=EPOCH_NUM, ae=None):
    iterations = 0
    start = time.time()
    
    best_loss = 1e5; best_snapshot_path = ''
    header = '  Time Epoch Iteration  Progress    (%Epoch)  Loss'
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f}'.split(','))
    
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    print(header)

    train_loader = data_loader('train',True)
    if ae != None:
        ae = nn.Sequential(*list(ae.children())[:-2])
    
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            iterations += 1
            # get the inputs
            inputs, _ = data
            inputs = inputs.to(device)

            if ae != None:
                inputs = ae(inputs.float())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            
            loss = criterion(outputs, inputs) 
            loss.backward()
            optimizer.step()  
        
            if loss.item() < best_loss:
                # found a model with better validation set accuracy
                best_loss = loss.item()
                snapshot_prefix = os.path.join(save_path, 'best_snapshot_' + net._class_name())
                best_snapshot_path = snapshot_prefix + '_loss_{:.4f}__iter_{}_model.pt'.format(loss.item(), iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(net, best_snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != best_snapshot_path:
                        os.remove(f)

            if iterations % LOG_EVERY == 0:
                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(train_loader),
                    100. * (1+i) / len(train_loader), loss.item()))
        
    print('Finished Training')
    return best_snapshot_path

def run(times, validation, gpu, window_width, plot, epoch):
    acc_col = []

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device('cuda:{}'.format(gpu))
        print("Using GPU for training")
    else:
        device = torch.device('cpu')

    for i in range(times):
        print('\n----------------------------- EXPERIMENT %d -----------------------------' % (i+1))
        net = AutoEncoder(NUM_FEATURES_USED,AE1_DIM,1).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.002)
        criterion = nn.MSELoss()

        ae1_fp = train_autocoder(optimizer, criterion, net, device, epoch_num=20)
        ae1 = torch.load(ae1_fp)

        net = AutoEncoder(AE1_DIM,AE2_DIM,1).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.002)
        
        ae2_fp = train_autocoder(optimizer, criterion, net, device, epoch_num=20, ae=ae1)
        ae2 = torch.load(ae2_fp)

        net = StackedAutoEncoder(ae1, ae2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()
        net, fp = train(optimizer, criterion, net, validation, device, epoch)
        acc, y_true, y_pred = test(net, fp, validation, device)
        acc_col.append(acc)

        if i == times - 1 and plot:
            heatmap(y_true, y_pred, "SAE"+str(NUM_FEATURES_USED))
    
    if times > 1:
        print(np.round(acc_col,2))
        print("Average accuracy of %d experiments is: %.3f %%" % (times, np.mean(acc_col)))
    else:
        print("Accuracy is: %.3f %%" % np.mean(acc_col))

    print("Accuracy is: %.3f %%" % acc)

    


    