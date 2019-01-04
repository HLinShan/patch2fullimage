# resnet50 cam class3  test cam test
from __future__ import division, print_function
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import os, sys

reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.optim as optim
import cv2
import random
from collections import OrderedDict
from pooling import WildcatPool2d,ClassWisePool

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        print(cam.shape) #3 49
        cam = cam.reshape(h, w)
        print(cam.shape)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam

def compute_AUCs(gt_np, pred_np):
    AUROCs = []
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        #         self.probs = F.softmax(self.preds)[0]
        #         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        #        gcam -= gcam.min()
        #        gcam /= gcam.max()
        #        gcam = cv2.resize(gcam, (1024, 1024))#(self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


def plotCM(classes, y_true, y_pred, savname):
    '''
    classes: a list of class names
    '''
    matrix = confusion_matrix(y_true, y_pred)
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([' '] + classes, rotation=90)
    ax.set_yticklabels([' '] + classes)
    # save
    plt.savefig(savname)


# ====== prepare dataset ======
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, train_or_valid="train", augm=False, angle=[-25, 25]):

        #        data_path = sys.argv[1]
        self.train_or_valid = train_or_valid
        image_names = []
        image_short_names=[]
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]

                label = items[1]
                label = [int(i) for i in label]
                # if label == [0] or label == [1]:
                if label == [0]:
                    label = [0]
                elif label == [2]:
                # elif label == [2] or label == [3]:
                    label = [1]
                else:# label == [4]:
                    # label = [2]
                    continue
                # label = np.eye(N_CLASSES, dtype=int)[label[0]]  # onehot
                image_short_names.append(items[0])
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_short_names=image_short_names
        self.image_names = image_names
        self.labels = labels
        self.augm = augm
        self.angle = angle
        # neg 0
        self.label_weight_neg = len(self.labels) / (len(self.labels) - np.sum(self.labels, axis=0))
        # pos 1
        self.label_weight_pos = len(self.labels) / (np.sum(self.labels, axis=0))

        self.augm = augm

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image_short_name=self.image_short_names[index]
        image = cv2.imread(image_name)
        # print(image_name)

        # cv2.imshow('before', image)
        image = cv2.resize(image, (224, 224))
        # image = cv2.resize(image, (896,1152))

        if self.augm:
            image = augment(image, self.angle)
        image = normalize_between(image, 0, 1)
        # cv2.imshow('after', image)
        cv2.waitKey()
        image = np.transpose(image, [2, 0, 1])

        label = self.labels[index]
        label_inverse = np.ones(len(label)) - label
        # label_inverse = np.ones(1) - label
        weight = np.add((label_inverse * self.label_weight_neg), (label * self.label_weight_pos))
        return torch.FloatTensor(image), torch.LongTensor(label), torch.from_numpy(weight).type(torch.FloatTensor),image_name,image_short_name

    def __len__(self):
        return len(self.labels)


def normalize_between(img, bottom, top):
    '''
    Normalizes between two numbers: bottom ~ top
    '''
    minimum = np.amin(img, keepdims=True).astype(np.float32)
    maximum = np.amax(img, keepdims=True).astype(np.float32)
    scale_factor = (top - bottom) / (maximum - minimum)
    final_array = (img - minimum) * scale_factor + bottom
    final_array = np.clip(final_array, bottom, top)

    return final_array


def augment(img, angle=[-45, 45], center=None, scale=1.0, p=0.5):
    '''
    augmentation
    '''
    # Random Flipped Horizontally
    if random.random() < p:
        img = cv2.flip(img, 1)
    # Random Flipped Vertically
    if random.random() < p:
        img = cv2.flip(img, 0)
    # Random Rotate
    if random.random() < p:
        (h, w) = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        angle = random.randint(angle[0], angle[1])
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (w, h))
    return img


def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learning rate: %f" % (param_group['lr']))



# construct model
class ResNet50(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    # init the structure of the net
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # self.features=self.resnet50.features
#        for p in self.resnet50.conv1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.bn1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.layer1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.parameters():
#            p.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = 512*4
        self.fc = nn.Linear(num_ftrs, num_classes)

    #     the forward the function
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        xm = self.resnet50.layer4(x) #64 2048 7 *7
        # tensor
        xm=F.sigmoid(xm)


        x = self.avgpool(xm) # 64 2048 1 1
        x = x.view(x.size(0), -1) # 64 2048
#        print(x.size())
        x = self.fc(x)
        # x =F.softmax(x)#64* 2048  -> 64*3
#        x = F.relu(x, inplace=True)
        return xm,x #heatmap  outputtensor  pro


if __name__ == '__main__':
    #   patch image data roi image
    DATA_DIR = './roi-img'
    TRAIN_DIR_ROI = './roi-img/roi_images_sample'
    TEST_DIR_ROI = './roi-img/roi_images'
    TRAIN_ROI_LIST = './roi-img/roi_images_sample/train_roi_images_sample.txt'
    TEST_ROI_LIST = './roi-img/roi_images/test_roi_images.txt'
    HEATMAP_IMAGE_LIST = './roi-img/roi_images_sample/heatmap.txt'

    #   full image  data
    # DATA_DIR = './full-img'
    # TRAIN_DIR_PATCH = './full-img'
    # TEST_DIR_PATCH = './full-img'
    # TRAIN_IMAGE_LIST = './full-img/train_full_images.txt'
    # VALID_IMAGE_LIST = './full-img/test_full_images.txt'

    #    VALID_IMAGE_LIST = './data/data_entry_test.txt'
    #    HEATMAP_IMAGE_LIST = './data/data_entry_boxonly_test_.txt'
    SAVE_DIRS = 'ResNet50_pretrain_224aug_Class2_clca_benign_malignant_roi'
    N_CLASSES = 2
    BATCH_SIZE = 64
    LR = 0.0001  # * 0.1 * 0.1
    CKPT_NAME = 'ResNet50_pretrain'  # pkl name for saving
    PKL_DIR = 'pkl/' + SAVE_DIRS + '/'
    LOG_DIR = 'logs/' + SAVE_DIRS + '/'
    STEP = 50000
    TRAIN = True
    TEST = False
    Generate_Heatmap = False
    Pre = False
    correct_pre = 0
    OUTPUT_DIR = 'output/' + SAVE_DIRS + '/'
    if os.path.isdir(OUTPUT_DIR):
        pass
    else:
        os.mkdir(OUTPUT_DIR)

    CKPT_PATH = PKL_DIR + CKPT_NAME + '_' + str(0) + '.pkl'  # pretrain model for loading

    # prepare training set
    print('prepare training set...')
    train_dataset = ChestXrayDataSet(data_dir=TRAIN_DIR_ROI,
                                     image_list_file=TRAIN_ROI_LIST,
                                     train_or_valid="train",
                                     augm=True
                                     )
    # prepare validation set
    print('prepare validation set...')
    valid_dataset = ChestXrayDataSet(data_dir=TEST_DIR_ROI,
                                     image_list_file=TEST_ROI_LIST,
                                     train_or_valid="valid"
                                     )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    heatmap_dataset = ChestXrayDataSet(data_dir=TRAIN_DIR_ROI,
                                     image_list_file=HEATMAP_IMAGE_LIST,
                                     train_or_valid="valid"
                                     )
    heatmap_loader = DataLoader(dataset=heatmap_dataset, batch_size=1, shuffle=False)


    # initialize and load the model
    print('initialize and load the model...')

    model = ResNet50(N_CLASSES)
    #    print(model.state_dict().keys())
    #    asas
    if Pre:
        model_dict = model.state_dict()
        pretrain_dict = torch.load(PKL_DIR + 'ResNet50_pretrain' + '.pkl')
        pretrain_dict = pretrain_dict['state_dict']
        pretrained_dict = {k.split('module.')[1]: v for k, v in pretrain_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain checkpoint")
    ste = 1
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoints = torch.load(CKPT_PATH)
        checkpoint = checkpoints['state_dict']
        ste = checkpoints['step']
        state_dict = {k.split('module.')[1]: v for k, v in checkpoint.items()}
        #        print(state_dict.keys())
        #        asas
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint: %s" % CKPT_PATH)
    else:
        print("=> no checkpoint found")

    if TRAIN:
        if os.path.isdir(PKL_DIR):
            pass
        else:
            os.mkdir(PKL_DIR)
        if os.path.isdir(LOG_DIR):
            pass
        else:
            os.mkdir(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)
        # ====== start training =======
        print('start training...')
        cudnn.benchmark = True
        # Get optimizer with correct params.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
        #        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9)
        #         define the changing of the learning rate
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
        model = torch.nn.DataParallel(model).cuda()
        model.train()

        running_loss = 0.0

        total_train_length = len(train_dataset)
        total_valid_length = len(valid_dataset)
        perm = np.random.permutation(np.arange(total_train_length))
        cur = 0
        correctsum = 0.
        #        for epoch in range(epoc, EPOCH):
        for step in range(ste, STEP + 1):
            #            for cur in np.arange(0, total_train_length, BATCH_SIZE):
            augment_img = []
            augment_label = []
            augment_weight = []

            #                print('load training data')
            end = cur + BATCH_SIZE
            p_indexs = perm[cur: end]
            cur = int(cur + BATCH_SIZE)
            if cur > len(train_dataset) - int(BATCH_SIZE):
                cur = 0
                perm = np.random.permutation(np.arange(len(train_dataset)))
            for p in p_indexs:
                single_img, single_label, single_weight,img_name,img_short_name = train_dataset[p]
                augment_img.append(single_img)
                augment_label.append(single_label)
                augment_weight.append(single_weight)

            inputs_sub = torch.stack(augment_img)
            labels_sub = torch.stack(augment_label)
            weights_sub = torch.stack(augment_weight)

            #                print('zero the parameter gradients...')
            optimizer.zero_grad()
            inputs_sub, labels_sub = Variable(inputs_sub.cuda()), Variable(labels_sub.cuda())
            weights_sub = Variable(weights_sub.cuda())
            weights = np.zeros(N_CLASSES)

            #                print('forward + backward + optimize...')
            #           softmax predict
            feature_map,outputs = model(inputs_sub)
            outps = F.softmax(outputs, 1)
            _, predicted = torch.max(outps.data, 1)
            predicted_np = predicted.cpu().numpy()
            target_np = labels_sub.data.squeeze().cpu().numpy()
            # print(predicted_np, target_np)
            correct = np.sum(predicted_np == target_np)
            correctsum += float(correct)/float(BATCH_SIZE)
            # print("outputs,train",outputs.shape)

            #            labels_sub = labels_sub.view(-1)
            labels_np = labels_sub.data.cpu().numpy()
            # print(labels_np.shape)
            #            weights[0] = len(labels_np)/(np.sum(labels_np == 0)+1)
            #            weights[1] = len(labels_np)/(np.sum(labels_np == 1)+1)
            #            weights[2] = len(labels_np)/(np.sum(labels_np == 2)+1)
            #            weights[3] = len(labels_np)/(np.sum(labels_np == 3)+1)
            #            weights[4] = len(labels_np)/(np.sum(labels_np == 4)+1)
            #            weights =Variable( torch.from_numpy(weights).type(torch.FloatTensor).cuda())

            outputs_np = outputs.data.cpu().numpy()
                       # criterion = nn.CrossEntropyLoss()#weight = weights)
            criterion = nn.NLLLoss()
            # bce_criterion = nn.BCELoss(size_average=True)
            #            print(labels_sub.size(), outputs.size())
            # print(outputs, labels_sub)
            loss = criterion(F.log_softmax(outputs, 1), labels_sub.squeeze())
            # loss = bce_criterion(outputs, labels_sub)
            #                loss = F.binary_cross_entropy(outputs, labels_sub, size_average=False)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data[0]

            if step % 20 == 0:
                running_loss = running_loss / 20
                print('[STEP:%d] loss: %.6f' % (step, running_loss))
                writer.add_scalar('Loss1', running_loss, step)
                correctsum /= 20.
                print("Acc of Train: %.4f"%correctsum)
                running_loss = 0.
                correctsum = 0.

            if step % 50 == 0:
                model.eval()
                # running_loss_val = 0.
                # test
                print('Validation Testing......')
                # 0 1 2
                CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
                print_learning_rate(optimizer)

                correct = []
                for p, (inputs_sub, labels_sub, weights_sub,img_name,img_short_name) in enumerate(valid_loader):
                    inputs_sub = Variable(inputs_sub.cuda())
                    labels_sub = Variable(labels_sub.cuda())
                    #                    labels_sub = labels_sub.view(-1)
                    feature_map, outputs = model(inputs_sub)
                    # zhunque label zhi
                    outps = F.softmax(outputs, 1)
                    _, predicted = torch.max(outps.data, 1)
                    predicted_np = predicted.cpu().numpy()
                    # _, label = torch.max(labels_sub.data, 1)
                    target_np = labels_sub.data.squeeze().cpu().numpy()
                    #                    print(predicted_np, target_np)
                    correct.append(int(predicted_np == target_np))
                    #                       print('compute val loss...')
                    # loss = bce_criterion(outputs, labels_sub)
                    # print(outputs, labels_sub)
                    # loss_val = F.nll_loss(F.log_softmax(outputs, 1), labels_sub.squeeze())
                    # running_loss_val += loss_val.data[0]
                # running_loss_val = running_loss_val / total_valid_length
                # print('[EPOCH:%d] loss_val: %.6f' % (step, running_loss_val))
                # writer.add_scalar('Loss_val', running_loss_val, step)
                correctsum = np.sum(correct) / total_valid_length
                print('Accuracy of the network on test images: %.4f' % (correctsum))
                writer.add_scalar('Acc_val', correctsum, step)
                model.train()
                # print statistics
                print('************************************')
                #            print('[EPOCH:%d] running_loss: %.8f' % (epoch, running_loss / (total_train_length / BATCH_SIZE)))
                torch.save({'state_dict': model.state_dict(), 'step': step},
                           PKL_DIR + CKPT_NAME + '_' + str(step) + '.pkl')
                print('Save [STEP:%d] statistics done!' % (step))
                print('************************************')
                if correctsum > correct_pre:
                    torch.save({'state_dict': model.state_dict(), 'step': step}, PKL_DIR + CKPT_NAME + '.pkl')
                    correct_pre = correctsum
                    print('Save best statistics done!')
                print('************************************')

        writer.close()
        print('Finished Training')

    if TEST:
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        total_valid_length = len(valid_dataset)
        # 0 1 2
        CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
        print(' initialize the ground truth and output tensor...')
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
        correct = []
        for p, (inputs_sub, labels_sub, weights_sub,img_name,img_short_name) in enumerate(valid_loader):
            if p < 2:
                print('the [%d/%d] testbatch' % (p, total_valid_length / 1))
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            feature_map, probs = model(input_img)
            probs = F.softmax(probs, 1)
            _, predicted = torch.max(probs.data, 1)
            predicted_np = predicted.cpu().numpy()
            _, label = torch.max(labels_sub, 1)
            target_np = label.cpu().numpy()
            #            print(predicted_np, target_np[0])
            #            asas
            correct.append(int(predicted_np == target_np))

            labels_sub = labels_sub.cuda()
            gt = torch.cat((gt, labels_sub), 0)
            pred = torch.cat((pred, probs.data), 0)

        correctsum = np.sum(correct) / total_valid_length
        print('Accuracy of the network on test images: %.4f' % (correctsum))
        # Plot Confusion Matrix
        _, gtmax = torch.max(gt, 1)
        _, predmax = torch.max(pred, 1)
        plotCM(CLASS_NAMES, gtmax.cpu().numpy(), predmax.cpu().numpy(), 'Confusion Matrix/' + SAVE_DIRS + '.png')
        # Compute AUROC
        print('Compute validation dataset avgAUROC...')
        gt_npy = gt.cpu().numpy()
        pred_npy = pred.cpu().numpy()
        np.save('./npy/' + SAVE_DIRS + '_gt.npy', gt_npy)
        np.save('./npy/' + SAVE_DIRS + '_pred.npy', pred_npy)
        AUROCs = compute_AUCs(gt_npy, pred_npy)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for idx in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[idx], AUROCs[idx]))


    if Generate_Heatmap:
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        total_heatmap_length = len(heatmap_loader)
        #        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        CLASS_NAMES = ['B', 'M', 'N']
        color_map = [(0, 0, 255), (0, 255, 0)]
        font = cv2.FONT_HERSHEY_COMPLEX

        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())






        for p, (input_var, target,weight,img_name,img_short_name) in enumerate(heatmap_loader):



            # label_name = CLASS_NAMES[int(target)]

            if p < 2:
                print('the [%d] heatmap img' % p)
            input_img = Variable(input_var.cuda(), volatile=True)
            probs_tensor, outp = model(input_img)
            _, predicted = torch.max(outp.data, 1)
            _,label=torch.max(target.data,1)
            predicted_np = predicted.cpu().numpy()

            # get softmax para

            print(label)
            print(predicted)
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

            probs_tensor_np = probs_tensor.data.cpu().numpy()


            #            print('generate heatmap...')
            prob,idx=outp.sort(0,True)
            probs_np = outp.data.cpu().numpy()
            idx=idx.cpu().numpy()

            heatmaps = []
            activate_classes = [0, 1, 2]
            print("1,",probs_tensor_np.shape)
            print("2,",weight_softmax.shape)

            CAMs = returnCAM(probs_tensor_np, weight_softmax, predicted_np)
            print(predicted_np)

            # print('output CAM.jpg for the top1 prediction: %s' % CLASS_NAMES[idx[0]])
            print(img_name)
            img = cv2.imread(img_name[0])
            height, width, _ = img.shape
            img_short=img_short_name[0].split(',')[0].split('.')[0]

            print(img_short.split('.')[0])

            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5


            outname = os.path.join(OUTPUT_DIR,  img_short+'_' + 'new' + '.png')
            cv2.imwrite(outname, result)




