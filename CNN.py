import torch

from utils.constants import *
from utils.NeuralNetwork import *
from utils.Plot import *
from utils.basic import *
    
torch.backends.cudnn.benchmark = True

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
        fc1_input_dim = int(window_width/2 - 1) 
        net = Conv1DNet(fc1_input_dim).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        net, fp = train(optimizer, criterion, net, validation, device, epoch)
        acc, y_true, y_pred = test(net, fp, validation, device)
        acc_col.append(acc)

        if i == times - 1 and plot:
            heatmap(y_true, y_pred, "ConV1D_"+str(NUM_CLASSES)+"_")
    
    if times > 1:
        print(np.round(acc_col,2))
        print("Average accuracy of %d experiments is: %.3f %%" % (times, np.mean(acc_col)))
    else:
        print("Accuracy is: %.3f %%" % np.mean(acc_col))
    