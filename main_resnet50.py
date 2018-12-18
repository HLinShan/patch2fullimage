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

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1]
                label = [int(i) for i in label]
                if label == [0] or label == [1]:
                    label = [0]
                elif label == [2] or label == [3]:
                    label = [1]
                elif label == [4]:
                    label = [2]
                label = np.eye(N_CLASSES, dtype=int)[label[0]]  # onehot
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

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
        weight = np.add((label_inverse * self.label_weight_neg), (label * self.label_weight_pos))
        return torch.FloatTensor(image), torch.FloatTensor(label), torch.from_numpy(weight).type(torch.FloatTensor)

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
        self.features=self.resnet50.features
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
        xm=F.sigmoid(xm)


        x = self.avgpool(xm) # 64 2048 1 1
        x = x.view(x.size(0), -1) # 64 2048
#        print(x.size())
        x = self.fc(x) #64* 2048  -> 64*3
        x =F.softmax(x)
#        x = F.relu(x, inplace=True)
        return xm,x #heatmap  output pro


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
    SAVE_DIRS = 'ResNet50_pretrain_224aug_Class3_pathology_roi'
    N_CLASSES = 3
    BATCH_SIZE = 64
    LR = 0.0001  # * 0.1 * 0.1
    CKPT_NAME = 'ResNet50_pretrain'  # pkl name for saving
    PKL_DIR = 'pkl/' + SAVE_DIRS + '/'
    LOG_DIR = 'logs/' + SAVE_DIRS + '/'
    STEP = 50000
    TRAIN = False
    TEST = False
    Generate_Heatmap = True
    Pre = True
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
    heatmap_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)


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
                single_img, single_label, single_weight = train_dataset[p]
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
            # print("outputs,train",outputs.shape)

            #            labels_sub = labels_sub.view(-1)
            labels_np = labels_sub.data.cpu().numpy()
            #            weights[0] = len(labels_np)/(np.sum(labels_np == 0)+1)
            #            weights[1] = len(labels_np)/(np.sum(labels_np == 1)+1)
            #            weights[2] = len(labels_np)/(np.sum(labels_np == 2)+1)
            #            weights[3] = len(labels_np)/(np.sum(labels_np == 3)+1)
            #            weights[4] = len(labels_np)/(np.sum(labels_np == 4)+1)
            #            weights =Variable( torch.from_numpy(weights).type(torch.FloatTensor).cuda())

            outputs_np = outputs.data.cpu().numpy()
            #            criterion = nn.CrossEntropyLoss()#weight = weights)
            bce_criterion = nn.BCELoss(size_average=True)
            #            print(labels_sub.size(), outputs.size())
            loss = bce_criterion(outputs, labels_sub)
            #                loss = F.binary_cross_entropy(outputs, labels_sub, size_average=False)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data[0]

            if step % 20 == 0:
                running_loss = running_loss / 20
                with open('train_running_loss.txt', 'a+') as f:
                    f.write(step + ' ' + running_loss)
                print('[STEP:%d] loss: %.6f' % (step, running_loss))
                writer.add_scalar('Loss1', running_loss, step)
                running_loss = 0.

            if step % 50 == 0:
                model.eval()
                running_loss_val = 0.
                # test
                print('Validation Testing......')
                # 0 1 2
                CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
                print_learning_rate(optimizer)

                correct = []
                for p, (inputs_sub, labels_sub, weights_sub) in enumerate(valid_loader):
                    inputs_sub = Variable(inputs_sub.cuda())
                    labels_sub = Variable(labels_sub.cuda())
                    #                    labels_sub = labels_sub.view(-1)
                    outputs = model(inputs_sub)
                    # zhunque label zhi
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_np = predicted.cpu().numpy()
                    _, label = torch.max(labels_sub.data, 1)
                    target_np = label.cpu().numpy()
                    #                    print(predicted_np, target_np)
                    correct.append(int(predicted_np == target_np))
                    #                       print('compute val loss...')
                    loss = bce_criterion(outputs, labels_sub)
                    running_loss_val += loss.data[0]
                running_loss_val = running_loss_val / total_valid_length
                print('[EPOCH:%d] loss_val: %.6f' % (step, running_loss_val))
                writer.add_scalar('Loss_val', running_loss_val, step)
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
        f.close()
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
        for p, (inputs_sub, labels_sub, weights_sub) in enumerate(valid_loader):
            if p < 2:
                print('the [%d/%d] testbatch' % (p, total_valid_length / 1))
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            probs = model(input_img)
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
        CLASS_NAMES = ['B', 'M', 'Normal']
        color_map = [(0, 0, 255), (0, 255, 0)]
        font = cv2.FONT_HERSHEY_COMPLEX
        thresholds = [0.132260, 0.038287, 0.166762, 0.173932, 0.045059, 0.043594, 0.013012,
                      0.043318, 0.059798, 0.028319, 0.011727, 0.015320, 0.031429, 0.002289]
        ground = {}
        #        gmasks = {}
        prediction = {}
        for p, (input_var, target,weight) in enumerate(heatmap_loader):


            ill_name = CLASS_NAMES[int(target)]

            if p < 2:
                print('the [%d] heatmap img' % p)
            input_img = Variable(input_var.cuda(), volatile=True)
            probs_tensor, outp = model(input_img)
            probs_tensor_np = probs_tensor.data.cpu().numpy()
            probs_np = outp.data.cpu().numpy()

            #            print('generate heatmap...')

            heatmaps = []
            activate_classes = [0, 1, 2]
            #            activate_classes =np.where((np.squeeze(probs_np)[:8] > thresholds[:8])==True)[0]
            #            activate_classes = np.where((np.squeeze(probs_np)[:8] > 0.5)==True)[0]
            for activate_class in activate_classes:
                heatmaps.append(probs_tensor_np[0, activate_class, :, :])

            # for k, npy in zip(activate_classes, heatmaps):
            #     ill_name_pre = CLASS_NAMES[k]
            #     hmask = npy
            #     if np.isnan(hmask).any():
            #         continue
            #     if not img_name in prediction:
            #         prediction[img_name] = {}
            #     #                    gmasks[img_name] = {}
            #     if not ill_name_pre in prediction[img_name]:
            #         prediction[img_name][ill_name_pre] = {}
            #         prediction[img_name][ill_name_pre]['heatmap'] = []
            #         #                    gmasks[img_name][ill_name_pre] = []
            #     prediction[img_name][ill_name_pre]['heatmap'].append(hmask)
                #                m = mask.numpy()[0][k,:,:]
        #                asas
        #                gmasks[img_name][ill_name_pre].append(m)

        #        gtall_num = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        #        'Pneumothorax': 0}
        #        acc = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        #        'Pneumothorax': 0}
        #        afp = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        #        'Pneumothorax': 0}
        gtall_num = {'B': 0, 'M': 0, 'N': 0}
        acc =  {'B': 0, 'M': 0, 'N': 0}
        afp = {'B': 0, 'M': 0, 'N': 0}
        ior = {}
        for img_name in ground:
            imgOriginal = cv2.imread(os.path.join(DATA_DIR, img_name), 1)
            imgOriginal = cv2.resize(imgOriginal, (224, 224))
            w_ori = np.shape(imgOriginal)[1]
            h_ori = np.shape(imgOriginal)[0]
            gt_num = 0
            positive_num = 0
            for ill_name in ground[img_name]:
                gmaskb = np.zeros([h_ori, w_ori])
                for i in range(len(ground[img_name][ill_name])):
                    x = int(ground[img_name][ill_name][i][0])
                    y = int(ground[img_name][ill_name][i][1])
                    w = int(ground[img_name][ill_name][i][2])
                    h = int(ground[img_name][ill_name][i][3])
                    cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), color_map[1], 2)
                    cv2.putText(imgOriginal, ill_name, (x, y), font, 1, color_map[1], 1)
                    gmaskb[y:y + h, x:x + w] = 1
                gtall_num[ill_name] += 1
                if not ill_name in ior:
                    ior[ill_name] = {}
                    ior[ill_name]['count'] = 0
                    ior[ill_name]['ior'] = 0.0
                if img_name in prediction:
                    for ill_name_pre in prediction[img_name]:
                        if ill_name_pre == ill_name:
                            hmask = prediction[img_name][ill_name_pre]['heatmap'][0]
                            #                            gmask = gmasks[img_name][ill_name_pre][0]
                            #                            gmask = cv2.resize(gmask, (h_ori, w_ori))
                            #                            gmask[np.where(gmask!=0)] = 1
                            #                            gmmask = cv2.applyColorMap(np.uint8(255*gmask), cv2.COLORMAP_JET)
                            hmask[np.where(hmask < 0.5 * hmask.max())] = 0
                            hmask = cv2.resize(hmask, (h_ori, w_ori))
                            hmmask = (hmask - hmask.min()) / hmask.max()
                            hmmask = cv2.applyColorMap(np.uint8(255 * hmmask), cv2.COLORMAP_JET)
                            img = hmmask * 0.5 + imgOriginal
                            outname = os.path.join(OUTPUT_DIR, img_name + '_' + ill_name_pre + '.png')
                            cv2.imwrite(outname, img)
                            #                            img_gm = gmmask*0.5 + imgOriginal
                            #                            outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre + '_' +'.png')
                            #                            cv2.imwrite(outname, img_gm)
                            #                            gbmask = cv2.applyColorMap(np.uint8(255*gmaskb), cv2.COLORMAP_JET)
                            #                            img_gb = gbmask*0.5 + imgOriginal
                            #                            outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre + '__' +'.png')
                            #                            cv2.imwrite(outname, img_gb)

                            hmask[np.where(hmask != 0)] = 1
                            if np.sum(hmask) == 0:
                                continue
                            iobb = np.sum(hmask * gmaskb) / np.sum(hmask)
                            #                            iobb = np.sum(hmask*gmaskb)/(np.sum(hmask)+np.sum(gmaskb)-np.sum(hmask*gmaskb))
                            ior[ill_name]['ior'] += iobb
                            ior[ill_name]['count'] += 1
                            if iobb >= 0.1:
                                acc[ill_name] += 1
                            elif iobb < 0.1:
                                afp[ill_name] += 1

                            #                        if not ill_name_pre in ground[img_name]:
        #                            afp[ill_name_pre] += 1
        ACC = 0.0
        AFP = 0.0
        IOR = 0.0
        for ill_name in gtall_num:
            acc[ill_name] = float(acc[ill_name]) / float(gtall_num[ill_name])
            ACC += acc[ill_name]
            afp[ill_name] = float(afp[ill_name]) / float(gtall_num[ill_name])
            AFP += afp[ill_name]
            print('The ACC of {} with threshold {} is : {}'.format(ill_name, 0.1, acc[ill_name]))
            print('The AFP of {} with threshold {} is : {}'.format(ill_name, 0.1, afp[ill_name]))
            if ior[ill_name]['count'] == 0: continue
            ior[ill_name]['avgIoR'] = float(ior[ill_name]['ior']) / float(ior[ill_name]['count'])
            IOR += ior[ill_name]['avgIoR']
            print('The avgIoR of {} is : {}'.format(ill_name, ior[ill_name]['avgIoR']))
        print('The average of ACC is : {}'.format(ACC / 8.0))
        print('The average of AFP is : {}'.format(AFP / 8.0))
        print('The average of IOR is : {}'.format(IOR / 8.0))


