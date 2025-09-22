# -*- coding:utf-8 -*-

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import datetime
import numpy as np
import warnings
import yaml
from utils.Data_Load_with_Dail import *
warnings.filterwarnings("ignore")
from utils.random_seed import setup_seed
from utils.loss import Fuse_loss
from utils.metrics import Fuse_loss,KLLoss,CCLoss,SIMLoss,Aux_loss,CustomLoss
"""
下面是模型加载
"""
from utils.model.ARFAF import ARFAF_Net
from utils.model.Resnet import resnet50 as Resnet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def Load_Data(train_inf,val_inf,Data_root_path,train_batch_size=64,val_batch_size=64,num_threads=4):
    train_eii = DataSource(batch_size=train_batch_size, Datainf=train_inf, Data_root_path=Data_root_path)
    train_pipe = SourcePipeline(batch_size=train_batch_size, num_threads=num_threads, device_id=0, external_data=train_eii,
                                modeltype='train')
    train_iter = CustomDALIGenericIterator(len(train_eii) / train_batch_size, pipelines=[train_pipe],
                                           output_map=['Satellite', "BSVI", 'Satellite_170','Road' ,'Risk_Map', 'Risk_Label'],
                                           last_batch_padded=False,
                                           size=len(train_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)
    val_eii = DataSource(batch_size=val_batch_size, Datainf=val_inf, Data_root_path=Data_root_path)
    val_pipe = SourcePipeline(batch_size=val_batch_size, num_threads=num_threads, device_id=0, external_data=val_eii,
                                modeltype='val')
    val_iter = CustomDALIGenericIterator(len(val_eii) / val_batch_size, pipelines=[val_pipe],
                                           output_map=['Satellite', "BSVI", 'Satellite_170','Road' ,'Risk_Map', 'Risk_Label'],
                                           last_batch_padded=False,
                                           size=len(val_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)

    train_loader = train_iter
    val_loader = val_iter
    return train_loader,val_loader

if __name__ == '__main__':
    Data_root_path = 'Dataset/'
    train_inf = pd.read_csv(os.path.join(Data_root_path, 'train.csv'))
    val_inf = pd.read_csv(os.path.join(Data_root_path, 'val.csv'))
    """
    循环进行对config文件进行训练
    """
    config_path = 'configs/aux_hyperparam.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for cfg in config:
        """
        先定义模型
        """
        torch.cuda.empty_cache()
        config_name = config_path.split('/')[1].replace('.yaml', '')
        setup_seed(cfg['seed'])
        if config_name == 'ARFAF_ablation':
            model = ARFAF_Net(HFAF_Flag=cfg['model']['HFAF_Flag'],GCDCA_flag=cfg['model']['GCDCA_Flag'],MPGF_flag=cfg['model']['MPGF_Flag']).cuda()
            experiment_name = 'ARFAF'
            for key, value in cfg['model'].items():
                if value is not False and value != 'ARFAF':
                    experiment_name += "_" + key.replace('_Flag', '')
        elif config_name == 'hyperparam' or  config_name == 'new_hyperparam' or config_name == 'aux_hyperparam' or config_name =='3_aux_hyperparam' or config_name == 'mse_hyperparam':
            model = ARFAF_Net(HFAF_Flag=cfg['model']['HFAF_Flag'],GCDCA_flag=cfg['model']['GCDCA_Flag'],MPGF_flag=cfg['model']['MPGF_Flag']).cuda()
            experiment_name = 'ARFAF' + '_' +str(cfg['learning_rate']).split('.')[1] + '_' + str(cfg['weight_decay']).split('.')[1] + '_' + str(cfg['momentum']).split('.')[1]
        """
       定义损失函数、模型参数
       """
        risk_criterion = nn.MSELoss().cuda()
        # fuse_loss = Fuse_loss().cuda()
        fuse_loss = CustomLoss().cuda()
        # fuse_loss = nn.MSELoss().cuda()
        aux_criterion = Aux_loss().cuda()
        # fuse_loss = SSIM_MSE_Loss().cuda()

        KLd_loss = KLLoss().cuda()
        if cfg['optimizer'] == 'adamw':
            try:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(cfg['momentum'], 0.99),eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
            except:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),weight_decay=float(cfg['weight_decay']))
        elif cfg['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),weight_decay=float(cfg['weight_decay']))
        if cfg['scheduler_flag']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg['train_epoch'],eta_min=cfg['eta_min'])
        """
        定义损失函数，模型存放位置
        """
        best_Risk_level_acc = 0.0
        best_Risk_map_mse = 200.0
        best_Risk_kldiv = 200.0
        model_path = os.path.join('model_pth', config_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_weight_path = os.path.join(model_path, experiment_name + '.pth')
        """
        数据加载
        """
        train_loader, val_loader = Load_Data(train_inf=train_inf, val_inf=val_inf, Data_root_path=Data_root_path,train_batch_size=cfg['train_batch_size'],val_batch_size=cfg['val_batch_size'], num_threads=8)
        train_numbers, val_numbers = len(train_inf), len(val_inf)
        """
        训练
        """
        writer = SummaryWriter("exp/" + config_path.split('/')[1].replace('.yaml', '') + '/' + experiment_name,flush_secs=60)
        print('*********************{}开始训练*********************'.format(experiment_name))
        for epoch in range(cfg['train_epoch']):
            sum_train_loss = 0.0
            train_Risk_mse = 0.0
            train_kld_loss = 0.0
            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout, ncols=200, position=0)
            for step, batch in enumerate(train_bar):
                """
                每次开始前将梯度清零
                """
                optimizer.zero_grad()
                Satellite_img, BSVI_img, Satellite_170, Road_map, Risk_Map, Risk_Label = batch['Satellite'], batch[ 'BSVI'], batch['Satellite_170'], batch['Road'], \
                                                                                        batch['Risk_Map'], batch['Risk_Label']
                Risk_map_output = model(Satellite_img, Satellite_170, BSVI_img)
                p_Risk_map_output,aux_Risk_map_output = Risk_map_output[0],Risk_map_output[1]

                Risk_map_output = torch.mul(p_Risk_map_output, Road_map.squeeze(1))
                aux_Risk_map_output = torch.mul(aux_Risk_map_output, Road_map.squeeze(1))

                p_loss = fuse_loss(100 * Risk_map_output,100 * Risk_Map)
                aux_loss = aux_criterion(aux_Risk_map_output,Risk_Map)
                loss = p_loss + aux_loss
                # Risk_map_mse = risk_criterion(100*Risk_map_output, 100*Risk_Map)
                kldiv_loss = KLd_loss(100 * Risk_map_output, 100 * Risk_Map)
                Risk_map_mse = risk_criterion(100 * Risk_map_output, 100 * Risk_Map)
                # kldiv_loss = KLd_loss(Risk_map_output, Risk_Map)

                loss.backward()
                optimizer.step()
                """
                计算训练期间的指标值
                """
                sum_train_loss = sum_train_loss + loss.item()
                train_Risk_mse = train_Risk_mse + Risk_map_mse.item()
                train_kld_loss = train_kld_loss + kldiv_loss.item()
                train_bar.desc = '训练阶段==> 风险Loss:{:.3f} 风险mse:{:.3f}'.format(loss.item(),Risk_map_mse.item())
            if cfg['scheduler_flag']:
                scheduler.step()
            epoch_times = step + 1
            print(' [Train epoch {}] 训练阶段平均指标======>风险loss:{:.3f} 风险mse:{:.3f} '.format(epoch + 1,sum_train_loss / epoch_times,train_Risk_mse/epoch_times))
            writer.add_scalar('训练指标/总loss', round(sum_train_loss / epoch_times, 4), epoch + 1)
            writer.add_scalar('训练指标/风险mse', round(train_Risk_mse / epoch_times, 4), epoch + 1)
            writer.add_scalar('训练指标/kld_loss', round(train_kld_loss / epoch_times, 4), epoch + 1)
            """
            进入评估阶段
            """
            if (epoch + 1) % cfg['train_val_times'] == 0:
                sum_val_loss = 0.0
                val_Risk_mse = 0.0
                val_kld_loss = 0.0
                model.eval()
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout, ncols=200, position=0)
                    output_map = []
                    label_map = []
                    for step, batch in enumerate(val_bar):
                        optimizer.zero_grad()
                        Satellite_img, BSVI_img, Satellite_170, Road_map, Risk_Map, Risk_Label = batch['Satellite'], batch['BSVI'], batch['Satellite_170'], \
                                                                                                 batch['Road'], batch['Risk_Map'], batch['Risk_Label']
                        Risk_map_output = model(Satellite_img, Satellite_170, BSVI_img)
                        p_Risk_map_output, aux_Risk_map_output = Risk_map_output[0], Risk_map_output[1]
                        Risk_map_output = torch.mul(p_Risk_map_output, Road_map.squeeze(1))
                        aux_Risk_map_output = torch.mul(aux_Risk_map_output, Road_map.squeeze(1))
                        loss = fuse_loss(100 * Risk_map_output, 100 * Risk_Map)
                        Risk_map_mse = risk_criterion(100 * Risk_map_output, 100 * Risk_Map)
                        kldiv_loss = KLd_loss(100*Risk_map_output, 100*Risk_Map)
                        val_Risk_mse += Risk_map_mse.item()
                        sum_val_loss = sum_val_loss + loss.item()
                        val_kld_loss = val_kld_loss + kldiv_loss.item()
                        val_bar.desc = '验证阶段==> 风险Loss:{:.3f} 风险mse:{:.3f}'.format(loss.item(),Risk_map_mse.item())
                    epoch_times = step + 1
                    print(' [Train epoch {}] 验证阶段平均指标======>风险loss:{:.3f} 风险mse:{:.3f} '.format(epoch + 1,sum_val_loss / epoch_times,val_Risk_mse / epoch_times))
                    # writer.add_scalars('验证指标', { "风险loss": round(sum_val_loss / epoch_times, 3),"风险mse": round(val_Risk_mse / epoch_times, 3)}, epoch + 1)
                    writer.add_scalar('验证指标/总loss', round(sum_val_loss / epoch_times, 4), epoch + 1)
                    writer.add_scalar('验证指标/风险mse', round(val_Risk_mse / epoch_times, 4), epoch + 1)
                    writer.add_scalar('验证指标/kld_loss', round(val_kld_loss / epoch_times, 4), epoch + 1)

                    if (best_Risk_map_mse >=val_Risk_mse / epoch_times) and (best_Risk_kldiv >=val_kld_loss / epoch_times):
                        best_Risk_map_mse = val_Risk_mse / epoch_times
                        best_Risk_kldiv = val_kld_loss / epoch_times
                        torch.save(model.state_dict(), model_weight_path)
