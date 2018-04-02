
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import pdb
from tqdm import tqdm
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='Bio embeddings to glove transoformation Network')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()

#Use CUDNN
cudnn.benchmark = True
use_cuda = False

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def test(val_data, val_outputs, net):
    '''
    Testing
    '''
	net.eval()  #Switch to eval mode
	criterion = nn.MSELoss()
	correct_pred = 0
	total_pred = 0
	tot_batches = val_data.shape[0] // args.batch_size
	count = 0
	for batch_id in tqdm(range(0, tot_batches)):
		if batch_id != tot_batches:
		    x_batch = val_data[batch_id * args.batch_size: (batch_id + 1) * args.batch_size, ]
		    y_batch = val_outputs[batch_id * args.batch_size: (batch_id + 1) * args.batch_size, ]
		else:
		    x_batch = val_data[batch_id * args.batch_size:, ]
		    y_batch = val_outputs[batch_id * args.batch_size:, ]

		x_batch = torch.autograd.Variable(x_batch, volatile=True)
		y_batch = torch.autograd.Variable(y_batch, volatile=True)
		if use_cuda:
      		y_batch = y_batch.cuda(async=True)
      		x_batch = x_batch.cuda()  #Keep test batches volatile as they dont need to be pinned
      
        outputs = net(x_batch)
        count += 1
        if batch_id == 0:
        	val_loss = criterion(outputs, y_batch)
        else:
        	val_loss += criterion(outputs, y_batch)

    avg_loss = val_loss / float(count)
    net.train() #Revert to Train mode before resuming training
    return avg_loss



class Net(nn.Module):
    def __init__(self, bio_embedding_dim = 300, glove_embedding_dim = 300):
        super(Net, self).__init__()
        #Input Batch_size X bio_embedding_dim
        #Expected output Batch_size X glove_embedding_dim

        #Input 1X1X512
        self.fc1 = nn.Linear(bio_embedding_dim, 512)
        #self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        #self.drop3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, glove_embedding_dim)
#        self.softmax = nn.Softmax()


        #Using Glorot's weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0] # number of rows
                fan_in = size[1] # number of columns
                variance = np.sqrt(2.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)


    def forward(self, x):

        x = F.relu(self.fc1(x)) #Relu here added post assignment
        #x = self.drop2(x)
        x = F.relu(self.fc2(x)) #Relu here added post assignment
        #x = self.drop3(x)
        x = self.fc3(x)
#        x = self.softmax(x)
 
        return x

def train(train_data, train_outputs):
	net = Net()
	if use_cuda:
		net = nn.DataParallel(net) #Attempt at multi GPU
		net = net.cuda() #Ship model to GPU
	#net = Net().cuda() #Ship model to GPU

	criterion = nn.MSELoss()
	if use_cuda:
		criterion = criterion.cuda()  #Ship loss to GPU

	#optimizer = optim.SGD(net.parameters(), lr=0.01 ,momentum = 0.9, weight_decay = 5e-4)
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	best_test_loss = 1000000
	train_data, train_outputs = shuffle(train_data, train_outputs, random_state=0)  # seed 0
	#Carve 20% val data
	val_data = train_data[:int(train_data.shape[0]*0.2)]
	val_outputs = train_outputs[:int(train_outputs.shape[0]*0.2)]
	train_data = train_data[int(train_data.shape[0]*0.2):]
	train_outputs = train_outputs[int(train_outputs.shape[0]*0.2):]

	for epoch in range(args.epochs):
	    '''
	    Training
	    '''
	    train_data, train_outputs = shuffle(train_data, train_outputs)  
	    train_loss = 0  #Stores loss and accumulates it for all batches
	    i_batch = 0
	    epoch_train_loss = 0
	    print_interval = 500
	    tot_batches = train_data.shape[0] // args.batch_size
	    for batch_id in tqdm(range(0, tot_batches)):
	        if batch_id != tot_batches:
	            x_batch = train_data[batch_id * args.batch_size: (batch_id + 1) * args.batch_size, ]
	            y_batch = train_outputs[batch_id * args.batch_size: (batch_id + 1) * args.batch_size, ]

	        else:
	            x_batch = train_data[batch_id * args.batch_size:, ]
	            y_batch = train_outputs[batch_id * args.batch_size:, ]


	      
	      	x_batch = torch.autograd.Variable(x_batch)
	      	y_batch = torch.autograd.Variable(y_batch)

	      	if use_cuda:
	      		y_batch = y_batch.cuda(async=True)          #Used when batches have been pinned to GPU memory
	      		x_batch = x_batch.cuda()

	      optimizer.zero_grad()   #Needed to set initial gradients to 0

	      #Forward pass
	      outputs = net(x_batch)
	      #Loss computation
	      loss = criterion(outputs, y_batch)
	      #Backward pass
	      loss.backward()
	      #Weight updates
	      optimizer.step()


	      #Cast into floats
	      outputs = outputs.float()
	      loss = loss.float()


	      train_loss = train_loss + loss.data[0]
	      epoch_train_loss = epoch_train_loss + loss.data[0]
	      if (i_batch + 1) % print_interval == 0:
	        print("Epoch "+str(epoch)+" Batch "+str(i_batch)+" Loss: "+str(train_loss/float(print_interval)))  #Print the average loss over past print_interval batches
	        train_loss = 0  #Reset loss

	      i_batch = i_batch + 1


	    if (epoch+1) % 10 == 0:
	      val_loss = test(val_data, val_outputs, net)
	      print('Epoch: '+str(epoch)+': Test Loss is: '+str(test_acc))
	      if val_loss < best_test_loss:
	        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'val_loss': val_loss,}, 'bio_to_glove_projection_model_checkpoint.tar')
	        best_test_loss = val_loss

	    if (epoch+1) % 30 == 0: #Halve the learning rate every 30 epochs
	          for param_group in optimizer.param_groups:
	            param_group['lr'] /= 2.0



	print('Training Complete')

def main():
	'''
	Main Function
	Process data and feed to train
	'''

	X = np.load('data/pubmed_vectors_train.npy')
	Y = np.load('data/glove_vectors_train.npy')
	train(X, Y)

if __name__ == "__main__":
    main()
