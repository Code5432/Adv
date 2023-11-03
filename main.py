import argparse
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
from torch import optim

import Face.verification_model
from Face.celeba_data import create_dic
from Face.celeba_solver import Celeba_Solver
from Face.utils import TVLoss, rec_transform, save_image
from Face.stargan_model import AFF, MS_CAM
import matplotlib as plt
import pandas as pd
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import numpy as np
import torch.nn.functional as F
from Face.data_loader_semantic import get_loader
import time
import datetime
from Face.backbone import Backbone
from Face.Models.MobileFace import MobileFacenet
from Face.Models.Mobilenet import MobileNet
from Face.Models.MobilenetV2 import MobileNetV2
from Face.Models.SphereFace import sphere20a
from Face.Models.CosFace import sphere
from Face.Models.IR import IR_50
from Face.Models.ShuffleNetV2 import ShufflenetV2




sys.path.append('../')
from attacks import semantic_attack


def denorm1(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class End_Model(nn.Module):
    def __init__(self, net):
        super(End_Model, self).__init__()
        self.net = net

    def forward(self, x):
        face_feature = self.net(rec_transform(x))
        normed_face_feature = face_feature / torch.norm(face_feature, dim=1)
        return normed_face_feature


def main(config):


    ind = 0
    success_records = 0
    fail_records = 0
    success_rate = []
    attack_records = []

    save_path = config.save_path
    # threshold = config.threshold
    # if threshold > 0 and config.untargeted == True:
    #     threshold = -threshold
    # if config.test_threshold == 0:
    #     test_threshold = threshold
    # else:
    #     test_threshold = config.test_threshold

    solver = Celeba_Solver(config)
#%%
    #goal = 'impersonation'
    #Models = ['MobileFace', 'MobileNet', 'MobileNetV2', 'SphereFace']
    #name_model = 'MobileFace'

    # %%
    if config.name_model == 'MobileFace':
        model = MobileFacenet()
        threshold = 1.35#1.578
        model.cuda()
        model.load_state_dict(torch.load(
            './pretrain_models/mobileface.pth',
            map_location=torch.device("cuda")))
    elif config.name_model == 'CosFace':
        model = sphere()
        threshold = 1.507
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/cosface.pth',map_location=torch.device("cuda")))
    elif config.name_model == 'MobileNetV2':
        model = MobileNetV2(stride=1)  # stride=1
        threshold = 1.547
        model.cuda()
        model.load_state_dict(torch.load(
            './pretrain_models/Backbone_Mobilenetv2.pth',
            map_location=torch.device("cuda")))
    elif config.name_model == 'SphereFace':
        model =  sphere20a()
        threshold = 1.301#1.301
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/sphere20a.pth', map_location=torch.device("cuda")))
    elif config.name_model == 'IR50-ArcFace':
        model =  IR_50(input_size=(112, 112))#, loss='Softmax')#loss='ArcFace'
        threshold = 1.37 #1.39
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/Backbone_IR_50_Arcface.pth', map_location=torch.device("cuda")))
    elif config.name_model == 'IR50-Softmax':
        model =  IR_50(input_size=(112, 112))#, loss='Softmax')#loss='ArcFace'
        threshold = 1.315
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/Backbone_IR_50_softmax.pth', map_location=torch.device("cuda")))
    elif config.name_model == 'IR50-CosFace':
        model =  IR_50(input_size=(112, 112))#, loss='Softmax')#loss='ArcFace'
        threshold = 1.55
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/Backbone_IR_50_cosface.pth', map_location=torch.device("cuda")))
    elif config.name_model == 'IR50-SphereFace':
        model =  IR_50(input_size=(112, 112))#, loss='Softmax')#loss='ArcFace'
        threshold = 1.277
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/Backbone_IR_50_sphereface.pth', map_location=torch.device("cuda")))
    elif config.name_model == 'FaceNet':
        model = InceptionResnetV1(pretrained='vggface2')  # resize image 160
        threshold = 1.15
    elif config.name_model == 'ShuffleNet_V1_GDConv':
        model = ShufflenetV2()  # , loss='Softmax')#loss='ArcFace'
        threshold = 1.552
        model.cuda()
        model.load_state_dict(
            torch.load('./pretrain_models/Backbone_ShuffleNetV2.pth',
                       map_location=torch.device("cuda")))


    else:
        model = Backbone([112, 112])
        model.cuda()
        model.load_state_dict(torch.load('./pretrain_models/backbone_ir50_ms1m_epoch120.pth', map_location=torch.device("cuda")))
        threshold = 1.37
    # model = InceptionResnetV1(pretrained='vggface2')  # resize image 160
    model.eval()
    model.cuda()

#%%
    criterionL2 = torch.nn.MSELoss(reduction ='sum')
    """""
    adversary = semantic_attack.FP_CW_TV(config.lr, config.max_iteration,
                                         config.tv_lambda,
                                         threshold) #/ 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    #%% Read images
    test_idlist = []
    test_namelist = []
    if config.data_mode == 'demo':
        sub_folder = ''
        data_list = [0, 2, 10, 24, 28, 69]
        if config.untargeted == True:
            original_list = [0, 1, 2, 3, 4, 5]
        else:
            original_list = [1, 2, 3, 4, 5]
    elif config.data_mode == 'all':
        sub_folder = 'ori/'
        data_list = range(10)#5000
        original_list = range(5)#2500
    start_time = time.time()
    for temp_i in data_list: # may add name identity
        temp_path1 = config.celeba_image_dir + str(temp_i + 1) + '/' + sub_folder
        test_idlist.append(str(temp_i + 1))
        test_namelist.append(sorted(os.listdir(temp_path1))[0])

    dic_label, dic_image = create_dic(config.celeba_image_dir,
                                      config.attr_path, config.selected_attrs,
                                      config.celeba_crop_size,
                                      config.image_size, test_idlist,
                                      test_namelist, sub_folder)

    for index in original_list:
        if config.untargeted == True:
            i_target = index
        else:
            if config.data_mode == 'demo':
                i_target = 0
            elif config.data_mode == 'all':
                i_target = index + 5 #5000 2500
 # %% Read target image and get embeddings
        nn = test_namelist[i_target]
        x_trg = dic_image[test_namelist[i_target]]
        x_trg = x_trg.unsqueeze(0).cuda()
        # t_img_ori = rec_transform(denorm1(x_trg)).cuda()
#%% apply model to target image
        x_trg_embedding = model(x_trg)
        x_trg_embedding_const = torch.zeros_like(x_trg_embedding)
        x_trg_embedding_const.data = x_trg_embedding.clone()
        x_trg_embedding_const = F.normalize(x_trg_embedding_const).cpu()
# %% to save target image
        """""
        save_path_trg = 'results/target/'
        save_image(
            save_path_trg + str(index) + '_' + str(i_target) + '_' +'target.png', denorm1(x_trg))
        """
# %% get original label and original image
        c_org = dic_label[test_namelist[index]]
        c_org = c_org.unsqueeze(0)
        nn_ori = test_namelist[index]
        x_real = dic_image[test_namelist[index]]
        x_real = x_real.unsqueeze(0)

        c_org = c_org.cuda()
        x_real = x_real.cuda()
        x_real_constant = denorm1(x_real).clone().cuda()
#%%
        """
        save_path_ori = 'results/original/'
        save_image(
            save_path_ori + str(index) + '_' + str(i_target) + '_' +
            'original.png', x_real_constant)
        """
#%%
        delta = torch.zeros_like(c_org)
        delta = delta.cuda()
        x_real.requires_grad = True
        optimizer = optim.Adam([x_real], lr=0.01)
#%% Opitimize X to get X'. G(X',c) looks more similar than G(X,c).
        for z in range(4):
            denormed_adv = solver.enc1(delta, x_real, c_org)
            edit_final = solver.dec(denormed_adv)

            img_loss = criterionL2(edit_final, x_real_constant)
            face_loss = criterionL2(denorm1(x_real), x_real_constant)
            loss = img_loss + face_loss * 1.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            x_real.data = torch.clamp(x_real.data, -1, 1)
#%% apply model to original image
        new_x_real = x_real.clone().cuda()
        x_ori_embedding = model(new_x_real)
        x_ori_embedding1 = F.normalize(x_ori_embedding).cpu()
        #%%
        dic_attribute = dict(zip(['Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses', 'Heavy_Makeup', 'Rosy_Cheeks',
                                  'Chubby', 'Mouth_Slightly_Open', 'Bushy_Eyebrows', 'Wearing_Lipstick', 'Smiling',
                                  'Arched_Eyebrows', 'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes', 'Receding_Hairline', 'Pale_Skin'],
                                 range(17)))
#%% to calculate similarity
        image_list = []
        label_list = []
        Similarity_list = []
        Similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        c_celeba_list, attrs_name = solver.create_labels_g(c_org, config.c_dim, 'CelebA', config.selected_attrs)
        with torch.no_grad():
            for i , c_trg in enumerate(c_celeba_list):
                x_fake = solver.G_img(new_x_real, c_trg)
                x_fake_feature = model(x_fake)
                #x_fake_feature = F.normalize(x_fake_feature).cpu()
                S = Similarity(x_ori_embedding, x_fake_feature)
                S = torch.clamp(S, 0, 1)
                image_list.append(x_fake)
                label_list.append(c_trg)
                Similarity_list.append(S)
# %% save csv file

            df = pd.DataFrame(list(zip(attrs_name, label_list)),columns=['labels', 'c_trg'])
            df_attr = pd.DataFrame(list(zip(attrs_name, Similarity_list,label_list)),columns=['labels', 'Similarity','c_trg'])
            df_attr.sort_values(by=['Similarity'], inplace=True , ascending=True)
            selected_att1 = df_attr.iloc[0]['labels']
            selected_att2 = df_attr.iloc[1]['labels']
            selected_att3 = df_attr.iloc[2]['labels']
            selected_att4 = df_attr.iloc[3]['labels']


            c_trg = df_attr.iloc[0]['c_trg']
            c_trg2 = df_attr.iloc[1]['c_trg']
            c_trg3 = df_attr.iloc[2]['c_trg']
            c_trg4 = df_attr.iloc[3]['c_trg']
#%% to create second attribute
            # delta = c_trg.clone()
            # delta = delta.cuda()
            # attribute_idx = [dic_attribute[selected_att2]]
            # attribute_idx3 = [dic_attribute[selected_att3]]
            #
            # if delta[:, attribute_idx] == 0:
            #     delta[:, attribute_idx] = 1
            # else:
            #     delta[:, attribute_idx] = 0
            # c_trg1 = delta.cuda()
            # if c_trg1[:, attribute_idx3] == 0:
            #     c_trg1[:, attribute_idx3] = 1
            # else:
            #     c_trg1[:, attribute_idx3] = 0
            # c_trg3 = c_trg1.cuda()
            # # delta = torch.zeros_like(c_org)
            # # delta = delta.cuda()
            # # delta[:, attribute_idx] = delta[:, attribute_idx] + 1
        from numpy.random import rand
        # alpha = rand(100)
        # alpha.sort()
        # alpha =0.99
        selected_att_list = []
        for i in range (10):
            c_trg = df_attr.iloc[i]['c_trg']
            selected_att = df_attr.iloc[i]['labels']
            if i > 0:
                selected_att = df_attr.iloc[i]['labels']
                delta = delta.cuda()
                attribute_idx = [dic_attribute[selected_att]]
                if delta[:, attribute_idx] == 0:
                    delta[:, attribute_idx] = 1
                else:
                    delta[:, attribute_idx] = 0
                c_trg = delta.cuda()


            # y = solver.G(new_x_real, c_trg)
            # alpha = 0.99999
            # alpha =
            from random import randint

            f_conv = solver.enc_conv(new_x_real, c_trg)
            f_res = solver.enc_res(new_x_real, c_trg)
            selected_att_list.append(selected_att)
            # %% alpha generation using binary search
            # initial_alpha =0
            # max_alpha = 0.2
            # min_alpha =  0.3
            #
            # k = torch.zeros_like(f_conv) + initial_alpha
            # k_max = torch.zeros_like(f_conv) + max_alpha
            # k_min = torch.zeros_like(f_conv) + min_alpha
            # for z in range(10 ):
            # y = solver.dec(1 - torch.min(torch.max(k, k_min), k_max)) * f_conv + torch.min(torch.max(k, k_min), k_max) * f_res
            dist_score = []
            imgs = []
            #threshold =1.24
            #alpha1 = [0.7,0.6,0.5,0.9]
            #for alpha in [0.7, 0.6, 0.5, 0.9]:
            alpha1 = torch.rand(100)
            for alpha in alpha1:
                y = solver.dec( f_conv * alpha + f_res * (1 - alpha))
                x_adv_embedding = model(y)
                x_adv_embedding_const = torch.zeros_like(x_adv_embedding)
                x_adv_embedding_const.data = x_adv_embedding.clone()
                x_adv_embedding_const = F.normalize(x_adv_embedding_const).cpu()
                #%%
                dist_score1 = torch.dist(x_adv_embedding, x_trg_embedding, 2)

                dist_score.append(dist_score1.item())
                imgs.append(y)
            df_dist = pd.DataFrame(list(zip(alpha1, dist_score,imgs)),
                                   columns=['alpha', 'Distance', 'images_name' ])
            df_dist.sort_values(by=['Distance'], inplace=True, ascending=True)
            x_adv = df_dist.iloc[0]['images_name']
            # max = max(dist_score)

            # y = solver.dec(2 * f_conv * 0.3 + 2 * f_res * (1 - .3))
#%% final result after binary search alpha
            x_adv_embedding = model(x_adv)
            x_adv_embedding_const = torch.zeros_like(x_adv_embedding)
            x_adv_embedding_const.data = x_adv_embedding.clone()
            x_adv_embedding_const = F.normalize(x_adv_embedding_const).cpu()
            ori_dist = torch.dist(x_ori_embedding, x_trg_embedding, 2)

            #adv_dist = torch.dist(x_adv_embedding, x_trg_embedding, 2)
            adv_dist = torch.dist(x_adv_embedding_const, x_trg_embedding_const, 2)

            score_ori = Similarity(x_ori_embedding1, x_trg_embedding_const)
            score = Similarity(x_adv_embedding_const, x_trg_embedding_const)
            if adv_dist.item() < (threshold):
                print (adv_dist.item())
                break
            else:
                new_x_real = x_adv
                delta = c_trg.clone()
        # %%

        f = open("train.txt", "a")
        #sys.stdout = open("train.txt", "w")
        print('source id:', index, ', target id:', i_target, 'No of attributes: ', i,
              ', attribute :', selected_att_list, ', feature distance:', adv_dist.item(),
              ', attack result:', adv_dist.item() < threshold, ', ori dist:',ori_dist.item(), file=f)
        f.close()

#%%
        if adv_dist <threshold:
            success_records += 1
        else:
            fail_records += 1
        save_image(
            save_path + str(index) + '_' + str(i_target) + '_' + '_' + 'adv_' + str(
                dic_attribute[selected_att1]) + ',' + str(dic_attribute[selected_att2]) + ').png', x_adv)

#%% time
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print('Time: ', et)
    # rate_list = []
    if (success_records + fail_records) > 0:
        rate = (success_records /
                         (success_records + fail_records))
    print('success rate for each attribute:', rate)

    if not osp.exists('./' + save_path):
        os.makedirs('./' + save_path)
    f = open(save_path + 'record.txt', 'w')

    f.write(
        'source id, target id, attribute index, feature distance,  attack result'
        + '\n')

    for length in range(len(attack_records)):
        f.write(str(attack_records[length])[1:-1] + '\n')

    f.write('success rate for each attribute: ' + str(rate)[1:-1] + '\n')

    f.close()

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim',
                        type=int,
                        default=17,
                        help='dimension of domain labels (1st dataset)')
    parser.add_argument('--celeba_crop_size',
                        type=int,
                        default=112,
                        help='crop size for the CelebA dataset')
    parser.add_argument('--image_size',
                        type=int,
                        default=(112, 112),
                        help='image resolution')
    parser.add_argument('--g_conv_dim',
                        type=int,
                        default=128,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim',
                        type=int,
                        default=128,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num',
                        type=int,
                        default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num',
                        type=int,
                        default=6,
                        help='number of strided conv layers in D')

    parser.add_argument('--dataset',
                        type=str,
                        default='CelebA',
                        choices=['CelebA'],
                        help='which dataset to use, only support CelebA currently')
    parser.add_argument('--c2_dim',
                        type=int,
                        default=8,
                        help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--selected_attrs',
                        '--list',
                        nargs='+',
                        help='selected attributes for the CelebA dataset',
                        default=['Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses',
                            'Heavy_Makeup', 'Rosy_Cheeks', 'Chubby',
                            'Mouth_Slightly_Open', 'Bushy_Eyebrows',
                            'Wearing_Lipstick', 'Smiling', 'Arched_Eyebrows',
                            'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes',
                            'Receding_Hairline', 'Pale_Skin'
                        ])

    # Test configuration.
    parser.add_argument('--test_iters',
                        type=int,
                        default=200000,
                        help='test model from this step')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    # Directories.
    parser.add_argument('--celeba_image_dir',
                        type=str,
                        default='./id_112_name/',
                        help='path of face images') #aligned_id_divied_imgs:demo,id_112_name:all
    parser.add_argument('--attr_path',
                        type=str,
                        default='./list_attr_celeba.txt',
                        help='path of face attributes')
    parser.add_argument('--model_save_dir',
                        type=str,
                        default='./pretrain_models/',
                        help='path of pretrained stargan model')
    # Face Recognition
    parser.add_argument('--feature_dim', default=256, type=int,
                        help='feature dimensions for face verification')
    parser.add_argument('--load_path',
                        type=str,
                        default='./pretrain_models/res101_softmax.pth.tar',
                        help='path of pretrained face verification model')
    # Please don't change above setting.

    # You can change below setting.
    parser.add_argument('--max_iteration', type=int, default=200,
                        help='maximum iterations')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--tv_lambda', type=float, default=0.01, help='lambda for tv loss')
    # parser.add_argument('--threshold', type=float, default=1.03,
    #                     help='threshold for face verification, 1.244 for fpr=10e-3，0.597 for fpr=10e-4')

    parser.add_argument('--save_path', type=str, default='results/adv/', help='path to save the results')
    parser.add_argument('--interp_layer', type=str, default='0', choices=['0', '1', '2', '01', '02'], help='which layer to interpolate')
    parser.add_argument('--test_threshold', type=float, default=0)
    parser.add_argument('--untargeted', action='store_true', help='targeted or untargeted')
    parser.add_argument('--data_mode', type=str, default='all', choices=['demo', 'all'], help='demo mode for simple demo, all mode for paper reproduction')
    parser.add_argument('--name_model', type=str, default='ArcFace', choices=['FaceNet', 'ArcFace','MobileFace','SphereFace','CosFace','IR50-ArcFace', 'IR50-Softmax','IR50-CosFace','IR50-SphereFace','MobileNetV2'], help='choose target model')
    parser.add_argument('--level_atts', type=str, default='single', choices=['single', 'multi'], help='choose level of attributes')
    parser.add_argument('--adv_attribute', type=str, default='all', choices=[
                            'Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses',
                            'Heavy_Makeup', 'Rosy_Cheeks', 'Chubby',
                            'Mouth_Slightly_Open', 'Bushy_Eyebrows',
                            'Wearing_Lipstick', 'Smiling', 'Arched_Eyebrows',
                            'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes',
                            'Receding_Hairline', 'Pale_Skin', 'all'
                        ], help='which attribute to use')
    # Directories.
    # parser.add_argument('--celeba_image_dir', type=str, default='D:/data/Celeba_cls_aligned')  # /celeba/images-- celeba_hq\train\female
    # parser.add_argument('--attr_path', type=str, default='D:/data/celeba/train_11000_data.csv')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')



    config = parser.parse_args()
    print(config)
    main(config)