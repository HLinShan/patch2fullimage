from __future__ import division, print_function
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import cv2
import random
from collections import OrderedDict

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
    
# ====== prepare dataset ======
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, train_or_valid = "train", transform=None):

#        data_path = sys.argv[1]
        self.train_or_valid = train_or_valid
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:            
            for line in f:
                items = line.split()
                image_name= items[0]#+'.png'
                label = items[1]
                label = [int(i) for i in label]
                if label == [3]:
                    label = [1]
                elif label == [4]:
                    label = [2]
#                print(label)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        
        self.label_weight_neg = len(self.labels)/(len(self.labels)-np.sum(self.labels, axis=0))
        self.label_weight_pos = len(self.labels)/(np.sum(self.labels, axis=0))
        
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        label_inverse = np.ones(len(label)) - label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        return image, torch.LongTensor(label), torch.from_numpy(weight).type(torch.FloatTensor)
        
    def __len__(self):
        return len(self.labels)

def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learning rate: %f"%(param_group['lr']))
        
# construct model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.features = self.densenet121.features
        num_ftrs = self.densenet121.classifier.in_features
#        for p in self.resnet50.conv1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.bn1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.layer1.parameters():
#            p.requires_grad = False
#        for p in self.resnet50.parameters():
#            p.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, True)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
#        print(x.size())
        x = self.fc(x)
#        x = F.relu(x, inplace=True)
        return x

        
if __name__ == '__main__':
          
    DATA_DIR = './data'
    TRAIN_IMAGE_LIST = './data/train_patch_list.txt'
    VALID_IMAGE_LIST = './data/test_patch_list.txt'
#    VALID_IMAGE_LIST = './data/data_entry_test.txt'
#    HEATMAP_IMAGE_LIST = './data/data_entry_boxonly_test_.txt'
    SAVE_DIRS = 'DenseNet121_pretrain_aug4_lr0.001_Class3'
    N_CLASSES = 3
    BATCH_SIZE = 64
    LR = 0.001
    CKPT_NAME = 'DenseNet121_pretrain'#pkl name for saving
    PKL_DIR = 'pkl/'+SAVE_DIRS +'/'
    LOG_DIR = 'logs/' + SAVE_DIRS +'/'
    STEP = 1500
    TRAIN = True
    Generate_Heatmap = False
    correct_pre = 0
    OUTPUT_DIR = 'output/' + SAVE_DIRS +'/'
    if os.path.isdir(OUTPUT_DIR):
        pass
    else:
        os.mkdir(OUTPUT_DIR) 
    
    CKPT_PATH = PKL_DIR + CKPT_NAME + '_' + str(0) + '.pkl'#pretrain model for loading
        
    # prepare training set
    print('prepare training set...')
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=TRAIN_IMAGE_LIST,
                            train_or_valid="train",
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(224),
#                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
#                                transforms.ColorJitter(0.15, 0.15),
                                transforms.RandomRotation([-25, 25]),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                                ]))
    # prepare validation set
    print('prepare validation set...')
    valid_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=VALID_IMAGE_LIST,
                            train_or_valid="valid",
                            transform=transforms.Compose([
                                transforms.Resize(256),
                    		     transforms.CenterCrop(224),
                    		     transforms.ToTensor(),
                    		     transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
			]))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    
    # prepare validation set
#    print('prepare validation set...')
#    heatmap_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
#                            image_list_file=HEATMAP_IMAGE_LIST,
#                            train_or_valid="valid",
#                            bbox = True,
#                            transform=transforms.Compose([
#                                transforms.Resize(256),
#                    			transforms.CenterCrop(224),
#                    			transforms.ToTensor(),
#                    			transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
#			]))
#    heatmap_loader = DataLoader(dataset=heatmap_dataset, batch_size=1, shuffle=False)

    # initialize and load the model
    print('initialize and load the model...')

    model = DenseNet121(N_CLASSES)
#    print(model.state_dict().keys())
#    asas
#    model_dict = model.state_dict()
#    pretrain_dict = torch.load(PKL_DIR+'pretrain' +'.pkl')
#    pretrain_dict = pretrain_dict['state_dict']
#    pretrained_dict = {k.split('module.')[1]: v for k, v in pretrain_dict.items()}
#    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#    model_dict.update(pretrained_dict)
#    model.load_state_dict(model_dict)
#    print("=> loaded pretrain checkpoint")
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
        print("=> loaded checkpoint: %s"%CKPT_PATH)
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
        model = torch.nn.DataParallel(model).cuda()
        model.train()
        
        running_loss = 0.0
        
        total_train_length = len(train_dataset)
        total_valid_length = len(valid_dataset)
        perm = np.random.permutation(np.arange(total_train_length))
        cur = 0
#        for epoch in range(epoc, EPOCH):
        for step in range(ste, STEP+1):
#            for cur in np.arange(0, total_train_length, BATCH_SIZE):
            augment_img = []
            augment_label = []
            augment_weight = []
        
#                print('load training data') 
            end = cur + BATCH_SIZE
            p_indexs = perm[cur: end]
            cur = int(cur + BATCH_SIZE)
            if cur > len(train_dataset)-int(BATCH_SIZE):
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
            outputs = model(inputs_sub)
            
            labels_sub = labels_sub.view(-1)
            labels_np = labels_sub.data.cpu().numpy()
            weights[0] = len(labels_np)/(np.sum(labels_np == 0)+1)
            weights[1] = len(labels_np)/(np.sum(labels_np == 1)+1)
            weights[2] = len(labels_np)/(np.sum(labels_np == 2)+1)
#            weights[3] = len(labels_np)/(np.sum(labels_np == 3)+1)
#            weights[4] = len(labels_np)/(np.sum(labels_np == 4)+1)
            weights =Variable( torch.from_numpy(weights).type(torch.FloatTensor).cuda())
            
            outputs_np = outputs.data.cpu().numpy()
            criterion = nn.CrossEntropyLoss(weight = weights)
#            print(labels_sub.size())
            loss = criterion(outputs, labels_sub)
#                loss = F.binary_cross_entropy(outputs, labels_sub, size_average=False)
            loss.backward()
            optimizer.step()                
            scheduler.step()
            running_loss += loss.data[0]
        
            if step%20 == 0:
                running_loss = running_loss/20
                print('[STEP:%d] loss: %.6f' % (step, running_loss))
                writer.add_scalar('Loss1', running_loss, step)
                running_loss = 0.
        
            if step%100 == 0:
                model.eval()
                running_loss_val = 0.
                print('Validation Testing......')
                print_learning_rate(optimizer)
                correct = 0
                for p, (inputs_sub, labels_sub, weights_sub) in enumerate(valid_loader):
        
                    inputs_sub = Variable(inputs_sub.cuda())
                    labels_sub = Variable(labels_sub.cuda())
                    labels_sub = labels_sub.view(-1)
                    outputs = model(inputs_sub)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_np = predicted.cpu().numpy()
                    target_np = labels_sub.data.cpu().numpy()
                    correct += (predicted_np == target_np).sum()
#                       print('compute val loss...')
                    loss = criterion(outputs, labels_sub)
                    running_loss_val += loss.data[0]
                running_loss_val = running_loss_val/total_valid_length
                print('[EPOCH:%d] loss_val: %.6f' % (step, running_loss_val))
                writer.add_scalar('Loss_val', running_loss_val, step)
                correct = correct/total_valid_length
                print('Accuracy of the network on test images: %.4f' % (correct))
                writer.add_scalar('Acc_val', correct, step)
            
                model.train() 
                # print statistics
                print('************************************')
    #            print('[EPOCH:%d] running_loss: %.8f' % (epoch, running_loss / (total_train_length / BATCH_SIZE)))
                torch.save({'state_dict': model.state_dict(), 'step': step},PKL_DIR+CKPT_NAME+'_'+str(step)+'.pkl')
                print('Save [STEP:%d] statistics done!' % (step))
                print('************************************')
                if correct > correct_pre:
                    torch.save({'state_dict': model.state_dict(), 'step': step},PKL_DIR+CKPT_NAME+'.pkl')
                    correct_pre = correct
                    print('Save best statistics done!')
                print('************************************')
            
        writer.close()
        print('Finished Training')   
    
    if not Generate_Heatmap:
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        total_valid_length = len(valid_dataset)
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                           'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        print(' initialize the ground truth and output tensor...')
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
        correct = 0
        for p, (inputs_sub, labels_sub, weights_sub) in enumerate(valid_loader):
            if p<2:
                print('the [%d/%d] testbatch'%(p, total_valid_length/1))
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            probs = model(input_img)
            _, predicted = torch.max(probs.data, 1)
            predicted_np = predicted.cpu().numpy()
            target_np = labels_sub.cpu().numpy()
#            print(predicted_np, target_np[0])
#            asas
            correct += (predicted_np == np.squeeze(target_np)).sum()
            
        print('Accuracy of the network on test images: %.4f' % (correct/total_valid_length))
#            probs_np = probs.data.cpu().numpy()
#            target_np = labels_sub.cpu().numpy()
            
#            labels_sub = labels_sub.cuda()
#            gt = torch.cat((gt, labels_sub), 0)
#            pred = torch.cat((pred, probs.data), 0)
                    
#        print('Compute validation dataset avgAUROC...')    
#        gt_npy = gt.cpu().numpy()
#        pred_npy = pred.cpu().numpy()
#        np.save('./npy/'+SAVE_DIRS+'_gt.npy', gt_npy)
#        np.save('./npy/'+SAVE_DIRS+'_pred.npy', pred_npy)
#        AUROCs = compute_AUCs(gt_npy, pred_npy)
#        AUROC_avg = np.array(AUROCs).mean()
#        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
#        for idx in range(N_CLASSES):
#            print('The AUROC of {} is {}'.format(CLASS_NAMES[idx], AUROCs[idx]))
        
    if Generate_Heatmap:
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        gcam = GradCAM(model=model, cuda=True)
        #=================initialize the prediction and ground truth for bbox================
        prediction = {}
        ground = {}
        color_map = np.array([[255, 0, 0], [0, 255, 0]])#red: pred, G: gt
        font = cv2.FONT_HERSHEY_COMPLEX
        total_heatmap_length = len(heatmap_loader)
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        thresholds = [0.11, 0.027, 0.14, 0.14, 0.05, 0.054, 0.013,
                      0.05, 0.05, 0.027, 0.023, 0.015, 0.029, 0.002]  
        for p, (input_var, target, weight, img_name, ill_name, b_box) in enumerate(heatmap_loader):
            
            img_name = str(img_name).split('/')[-1].split("'")[0]
            ill_name = str(ill_name).split("'")[1]
            if not img_name in ground:
                ground[img_name] = {}
            if not ill_name in ground[img_name]:
                ground[img_name][ill_name] = []
            ground[img_name][ill_name].append(b_box)   
            
            print('the [%d] testimg'%p)
            input_img = Variable(input_var.cuda(), requires_grad=True)
            probs = gcam.forward(input_img)  
#            probs_np = probs.data.cpu().numpy()
            
            #---- Blend original and heatmap                   
            heatmap_output = []                    
            activate_classes = np.where((np.squeeze(probs)[:8] > thresholds[:8])==True)[0]               
            for activate_class in activate_classes:
                
                gcam.backward(idx=activate_class)
                output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv2")
#                output = cv2.resize(output, (224, 224))
                
                if np.sum(np.isnan(output)) > 0:
                    print("fxxx nan")
                heatmap_output.append(output)
            
            for k, npy in zip(activate_classes, heatmap_output):
                
                ill_name_pre = CLASS_NAMES[k]
                
                hmask  = npy
                if np.isnan(hmask).any():
                    continue                  
                
                if not img_name in prediction:
                    prediction[img_name] = {}
                if not ill_name_pre in prediction[img_name]:
                    prediction[img_name][ill_name_pre] = {}
                    prediction[img_name][ill_name_pre]['heatmap'] = []                     
                prediction[img_name][ill_name_pre]['heatmap'].append(hmask) 
                
        color = {CLASS_NAMES[0]: [255, 0, 0], CLASS_NAMES[1]: [0, 255, 0], CLASS_NAMES[2]: [0, 0, 255], 
                      CLASS_NAMES[3]: [240,255,0], CLASS_NAMES[4]: [128, 0, 128], CLASS_NAMES[5]: [244,164,96], 
                      CLASS_NAMES[6]: [119,136,153], CLASS_NAMES[7]: [50,205,50]}
        gtall_num = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        'Pneumothorax': 0}
        acc = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        'Pneumothorax': 0}
        afp = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        'Pneumothorax': 0}
        thresholds = {'Atelectasis':0.5, 'Cardiomegaly':0.01, 'Effusion':0.5, 
                      'Infiltration':0.5, 'Mass':0.5, 'Nodule':0.5, 
                      'Pneumonia':0.02, 'Pneumothorax':0.06} 
        best_ior = {'Atelectasis': 0.0, 'Cardiomegaly': 0.0, 'Effusion': 0.0, 'Infiltration': 0.0, 'Mass': 0.0, 'Nodule': 0.0, 'Pneumonia': 0.0,
        'Pneumothorax': 0.0}
        best_t = {'Atelectasis': 0, 'Cardiomegaly': 0, 'Effusion': 0, 'Infiltration': 0, 'Mass': 0, 'Nodule': 0, 'Pneumonia': 0,
        'Pneumothorax': 0}
        iou = {}
        for t in np.arange(0.01,1,0.01):
            for img_name in ground:                
                imgOriginal = cv2.imread(os.path.join(DATA_DIR, img_name), 1)
                w_ori = np.shape(imgOriginal)[1]
                h_ori = np.shape(imgOriginal)[0]
                gt_num = 0
                positive_num = 0
                for ill_name in ground[img_name]:
                    gmask = np.zeros([h_ori, w_ori])
                    for i in range(len(ground[img_name][ill_name])):
                        x = int(ground[img_name][ill_name][i][0])
                        y = int(ground[img_name][ill_name][i][1])
                        w = int(ground[img_name][ill_name][i][2])
                        h = int(ground[img_name][ill_name][i][3])
                    gmask[y:y+h, x:x+w] = 1
                    if not ill_name in iou:
                        iou[ill_name] = {}
                        iou[ill_name]['count'] = 0
                        iou[ill_name]['iou'] = 0.0
                    if img_name in prediction:
                        for ill_name_pre in prediction[img_name]:
                            if ill_name_pre == ill_name:
                                hmask = prediction[img_name][ill_name_pre]['heatmap'][0]
                                hmask[np.where(hmask<t)] = 0
                                hmask = cv2.resize(hmask, (h_ori, w_ori))   
                                hmask[np.where(hmask!=0)] = 1 
                                if np.sum(hmask) == 0:
                                    continue
                                IOU = np.sum(hmask*gmask)/(np.sum(hmask)+np.sum(gmask)-np.sum(hmask*gmask))
                                iou[ill_name]['iou'] += IOU
                                iou[ill_name]['count'] += 1
            for ill in best_ior:
                if iou[ill]['count'] == 0: continue
                IOU=iou[ill]['iou']/iou[ill]['count']
                if IOU>best_ior[ill]: 
                    best_ior[ill]=IOU
                    best_t[ill] = t
                iou[ill]['iou']=0.0
                iou[ill]['count']=0
        for ill in best_ior:
            print('[best_ior/best_threshold][{}/{}] of {}'.format(best_ior[ill],best_t[ill],ill))
        a=1/0
        ior = {}
        for img_name in ground:                
            imgOriginal = cv2.imread(os.path.join(DATA_DIR, img_name), 1)
            w_ori = np.shape(imgOriginal)[1]
            h_ori = np.shape(imgOriginal)[0]
            gt_num = 0
            positive_num = 0
            for ill_name in ground[img_name]:
                gmask = np.zeros([h_ori, w_ori])
                for i in range(len(ground[img_name][ill_name])):
                    x = int(ground[img_name][ill_name][i][0])
                    y = int(ground[img_name][ill_name][i][1])
                    w = int(ground[img_name][ill_name][i][2])
                    h = int(ground[img_name][ill_name][i][3])
                    cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), color_map[1], 2)
                    cv2.putText(imgOriginal, ill_name, (x, y), font, 1, color_map[1], 1)
                    gmask[y:y+h, x:x+w] = 1
                gtall_num[ill_name] += 1                
                if not ill_name in ior:
                    ior[ill_name] = {}
                    ior[ill_name]['count'] = 0
                    ior[ill_name]['ior'] = 0.0
                if img_name in prediction:
                    for ill_name_pre in prediction[img_name]:
                        if ill_name_pre == ill_name:
                            hmask = prediction[img_name][ill_name_pre]['heatmap'][0] 
                            hmask[np.where(hmask<thresholds[ill_name_pre])] = 0 
                            hmask = cv2.resize(hmask, (h_ori, w_ori))  
                            hmmask = hmask/hmask.max()                  
                            hmmask = cv2.applyColorMap(np.uint8(255*hmmask), cv2.COLORMAP_JET) 
                            img = hmmask*0.5 + imgOriginal
                            outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre +'.png')
                            cv2.imwrite(outname, img)   
                            
                            hmask[np.where(hmask!=0)] = 1
                            if np.sum(hmask) == 0:
                                continue
                            iobb = np.sum(hmask*gmask)/np.sum(hmask)
                            ior[ill_name]['ior'] += iobb
                            ior[ill_name]['count'] += 1
                            if iobb >= 0.1: acc[ill_name] += 1
                            elif iobb < 0.1: afp[ill_name] += 1           
                        
                            
                        if not ill_name_pre in ground[img_name]:
                            afp[ill_name_pre] += 1
        ACC = 0.0
        AFP = 0.0
        IOR = 0.0
        for ill_name in gtall_num:
            acc[ill_name] = float(acc[ill_name])/float(gtall_num[ill_name])
            ACC += acc[ill_name]
            afp[ill_name] = float(afp[ill_name])/float(gtall_num[ill_name])
            AFP += afp[ill_name]
            print('The ACC of {} with threshold {} is : {}'.format(ill_name, 0.1, acc[ill_name]))
            print('The AFP of {} with threshold {} is : {}'.format(ill_name, 0.1, afp[ill_name]))  
            if ior[ill_name]['count'] == 0:continue
            ior[ill_name]['avgIoR'] = float(ior[ill_name]['ior'])/float(ior[ill_name]['count'])
            IOR += ior[ill_name]['avgIoR']      
            print('The avgIoR of {} is : {}'.format(ill_name, ior[ill_name]['avgIoR']))  
        print('The average of ACC is : {}'.format(ACC/8.0))
        print('The average of AFP is : {}'.format(AFP/8.0))
        print('The average of IOR is : {}'.format(IOR/8.0))