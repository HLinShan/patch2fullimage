import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  torch import optim
import torch
from matplotlib.pyplot import imshow

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.pool=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv2(x)))  # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x=x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#

#
# params = list(net.parameters())
# print(len(params))
# print(params[0].size()) # conv1's .weight
#
# input = Variable(torch.randn(1, 1, 32, 32))
# output = net(input)
# # output = output.view(-1)
# print output
#
# net.zero_grad()
# output.backward()
#
# # label
# target=Variable(torch.range(1,10))
#
# target = target.view(-1,10)
#
# print target
# criterion=nn.MSELoss()
# loss=criterion(output,target)
# print "loss",loss

# data
import torchvision
import torchvision.transforms as transforms
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

trainset = torchvision.datasets.CIFAR10(root='./data1', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data1', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net
net=Net()
net.cuda()
# optimizer
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train
'''
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print 'Finished Training'
torch.save(net,'ls_net_model/model.pkl')
'''

# # test
dataiter = iter(testloader)
images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print'GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(4))
net=torch.load('ls_net_model/model.pkl').cuda()
print "loading net"

outputs = net(Variable(images).cuda())

# the outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class

# So, let's get the index of the highest energy
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s'% classes[predicted[j][0]] for j in range(4)))

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images).cuda())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print 'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images).cuda())
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i].cuda()
        class_correct[label] += c[i]
        class_total[label] += 1

print  classes[0]
#
for i in range(10):
    print classes[i]
    print class_correct[i]
    print class_total[i]
    # acc=float(class_correct[i]/class_total[i])
    # print 'Accuracy of %5s : %2d %%' % (classes[i], 100 *acc)