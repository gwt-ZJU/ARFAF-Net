import os
import random
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from nvidia import dali
from nvidia.dali import tensors
from nvidia.dali import pipeline_def
from nvidia.dali.fn import experimental
# import nvidia.dali.fn as fn
from nvidia.dali import fn
from PIL import Image
from torchvision import transforms
from PIL import Image
import json
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from tqdm import tqdm
import sys
import datetime
import matplotlib.pyplot as plt
import cv2
import warnings
import pandas as pd
import re
warnings.filterwarnings("ignore")

def labelShuffling(dataFrame, groupByName = 'class_num'):
    np.random.seed(42)
    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]
        print("Num of the label is : ", labels[i])
        lst=lst._append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        print("Done")
    return lst

class DataSource(object):
    def __init__(self, Datainf,Data_root_path,Risk_Map='Risk_Map',shuffle=True,batch_size=64,load_type='train',seq_number=4):
        self.batch_size = batch_size
        self.load_type = load_type
        self.Risk_Map = Risk_Map
        # if self.load_type == 'train':
        #     Datainf = labelShuffling(Datainf,groupByName='交叉口类别')
        self.Number = len(Datainf)
        Datainf = Datainf.values.tolist()
        self.paths = list(zip(*(Datainf,)))
        self.shuffle = shuffle
        self.data_root_path = Data_root_path
        self.Satellite_file = os.path.join(self.data_root_path, 'Satellite')
        self.Satellite_170_file = os.path.join(self.data_root_path, 'Satellite_170')
        self.BSVI_file = os.path.join(self.data_root_path, 'BSVI')
        self.Road_file = os.path.join(self.data_root_path, 'Road')
        self.Risk_Map_file = os.path.join(self.data_root_path, self.Risk_Map)
        self.seq_number = seq_number
        if shuffle:
            random.shuffle(self.paths)
        pass

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.paths)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.paths):
            self.__iter__()
            raise StopIteration

        """
        数据存放定义
        """
        Satellite_img_list = []
        BSVI_img_list = []
        Road_img_list = []
        Satellite_170_img_list = []
        Risk_Map_list = []
        Risk_Label_list = []
        if self.load_type == 'cam':
            index_list = []
        for _ in range(self.batch_size):
            data_inf = self.paths[self.i % len(self.paths)][0]
            satellite_img_path = os.path.join(self.Satellite_file,str(int(data_inf[0]))+'.jpg')
            BSVI_img_path = os.path.join(self.BSVI_file, str(int(data_inf[0])) + '.jpg')
            satellite_170_img_path = os.path.join(self.Satellite_170_file, str(int(data_inf[0])) + '.jpg')
            Road_img_path = os.path.join(self.Road_file, str(int(data_inf[0])) + '.npy')
            Risk_Map_path = os.path.join(self.Risk_Map_file, str(int(data_inf[0])) + '.npy')

            Satellite_img_list.append(np.fromfile(satellite_img_path, dtype=np.uint8))
            BSVI_img_list.append(np.fromfile(BSVI_img_path, dtype=np.uint8))
            Road_img_list.append(np.load(Road_img_path).astype(np.float32))
            Satellite_170_img_list.append(np.fromfile(satellite_170_img_path, dtype=np.uint8))
            Risk_Map_list.append(np.load(Risk_Map_path).astype(np.float32))
            Risk_Label_list.append(np.array(int(data_inf[0])).astype(np.float32))
            self.i += 1
        if self.load_type == 'train' or self.load_type == 'val' :
            return (Satellite_img_list,BSVI_img_list,Satellite_170_img_list,Road_img_list,Risk_Map_list,Risk_Label_list)
        elif self.load_type == 'cam':
            return (Satellite_img_list,BSVI_img_list,Satellite_170_img_list,Road_img_list,Risk_Map_list,Risk_Label_list)


    def __len__(self):
        return len(self.paths)

    next = __next__

class SourcePipeline(Pipeline):
    def __init__(self,  batch_size, num_threads, device_id, external_data,modeltype,load_type='train'):
        super(SourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     exec_async=True,
                                                     exec_pipelined=True,
                                                     prefetch_queue_depth = 2,
                                                     )
        self.load_type = load_type
        self.input_Satellite_img = ops.ExternalSource()
        self.input_BSVI_img = ops.ExternalSource()
        self.intput_Satellite_170_img = ops.ExternalSource()
        self.intput_Road_img = ops.ExternalSource()
        self.input_Risk_Map = ops.ExternalSource()
        self.input_Risk_Label = ops.ExternalSource()

        if self.load_type == 'cam':
            self.input_index_list = ops.ExternalSource()

        self.external_data = external_data
        self.model_type = modeltype
        self.iterator = iter(self.external_data)
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.decode_for_depth = ops.decoders.Image(device="mixed", output_type=types.ANY_DATA)
        self.cat = ops.Cat(device="gpu",axis=2)
        self.tran = ops.Transpose(device="gpu",perm=[2,0,1])




        self.Mirror_probability = ops.CoinFlip(probability=0.5)

        self.train_resize = ops.Resize(device='gpu', resize_x=256, resize_y=192)
        self.Normalize = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               output_layout=types.NCHW,
                                               crop_pos_x=0,
                                               crop_pos_y=0,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])


        self.val_resize = ops.Resize(device='gpu', resize_x=256, resize_y=256, dtype=types.FLOAT)
        self.cat = ops.Cat(device="gpu", axis=0)
        self.Rotation_probability = ops.Uniform(range=(-10, 10)) #随机旋转
        self.Mirror_probability = ops.CoinFlip(probability=0.5) #镜像翻转
        # self.fuse_crop = ops.RandomResizedCrop(device="gpu", size=416, random_area=[0.75, 1.0])
        self.fuse_crop = ops.RandomResizedCrop(device="gpu", size=256, random_area=[0.8, 1.0])
        self.fuse_tran_HWC = ops.Transpose(device="gpu", perm=[1, 2, 0])
        self.fuse_tran_CHW = ops.Transpose(device="gpu", perm=[2, 0, 1])
        self.img_resize = ops.Resize(device='gpu', resize_x=256, resize_y=256, dtype=types.FLOAT)
        self.road_normalize = ops.CropMirrorNormalize(device="gpu",
                                                      dtype=types.FLOAT,
                                                      output_layout=types.NCHW,
                                                      crop_pos_x=0,
                                                      crop_pos_y=0,
                                                      mean=[0, 0, 0],
                                                      std=[1, 1, 1])
        self.seg_img_resize = ops.Resize(device='gpu', resize_x=512, resize_y=256, dtype=types.FLOAT)
        self.bsvi_img_resize = ops.Resize(device='gpu', resize_x=416, resize_y=256, dtype=types.FLOAT)
        self.BSVI_crop = ops.RandomResizedCrop(device="gpu", size=256, random_area=[0.8, 1.00])
        self.fuse_resize = ops.Resize(device='gpu', resize_x=512, resize_y=512, dtype=types.FLOAT)
        self.bsvi_img_val_resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)

    def define_graph(self):
        self.Satellite_img = self.input_Satellite_img()
        self.BSVI_img = self.input_BSVI_img()
        self.Satellite_170_img = self.intput_Satellite_170_img()
        self.Road_img = self.intput_Road_img()
        self.Risk_Map = self.input_Risk_Map()
        self.Risk_Label = self.input_Risk_Label()

        if self.load_type == 'cam':
            self.index_list = self.input_index_list()
            index_list = self.index_list.gpu()

        if self.model_type == 'train':
            Satellite_img = self.decode(self.Satellite_img)
            BSVI_img = self.decode(self.BSVI_img)
            Satellite_170_img = self.decode(self.Satellite_170_img)
            Road_img = self.Road_img.gpu()
            Risk_Map = self.Risk_Map.gpu()
            """
            # 开始对数据数据进行分开处理
            # 1.卫星图：只归一化
            # 2.街景图像：改变大小到
            # 3.街景语义图像：只归一化
            # 4.风险图：不变
            # 以上的训练和验证采用相同的操作
            #新的处理方式
            1.将遥感图 路网图 风险图 进行合并：然后进行 随机翻转 【左右、上下】 随机裁剪 随机旋转
            2.将街景全景图 执行 随机翻转、随机裁剪、随机旋转,语义图同样执行这些操作
            """
            # 1
            Satellite_img = self.Normalize(Satellite_img)
            Satellite_170_img = self.Normalize(Satellite_170_img)
            Road_img = Road_img[dali.newaxis]
            Risk_Map = Risk_Map[dali.newaxis]
            fuse_imgs = self.cat(Satellite_img,Satellite_170_img,Road_img,Risk_Map)
            fuse_imgs = self.fuse_resize(fuse_imgs)
            # fuse_imgs = self.fuse_tran_HWC(fuse_imgs)
            # Rotation_probability = self.Rotation_probability()
            # fuse_imgs = fn.rotate(fuse_imgs, angle=Rotation_probability, fill_value=0)
            # fuse_imgs = self.fuse_tran_CHW(fuse_imgs)
            # Mirror_probability = self.Mirror_probability()
            # fuse_imgs = fn.flip(fuse_imgs, horizontal=Mirror_probability)
            # Mirror_probability = self.Mirror_probability()
            # fuse_imgs = fn.flip(fuse_imgs, vertical=Mirror_probability)
            fuse_imgs = self.fuse_crop(fuse_imgs)
            Satellite_img,Satellite_170_img, Road_img, Risk_Map = fuse_imgs[0:3,:,:],fuse_imgs[3:6,:,:],fuse_imgs[6:7,:,:],fuse_imgs[7:8,:,:]
            Risk_Map = Risk_Map[0]
            # 2
            Rotation_probability = self.Rotation_probability()
            Mirror_probability = self.Mirror_probability()
            BSVI_img = fn.rotate(BSVI_img, angle=Rotation_probability, fill_value=0)
            BSVI_img = fn.flip(BSVI_img, horizontal=Mirror_probability)
            BSVI_img = self.bsvi_img_resize(BSVI_img)
            BSVI_img = self.Normalize(BSVI_img)
            BSVI_img = self.BSVI_crop(BSVI_img)
        if self.model_type == 'val':
            Satellite_img = self.decode(self.Satellite_img)
            BSVI_img = self.decode(self.BSVI_img)
            Satellite_170_img = self.decode(self.Satellite_170_img)
            Road_img = self.Road_img.gpu()
            Risk_Map = self.Risk_Map.gpu()
            """
            开始对数据数据进行分开处理
            1.卫星图：只归一化
            2.街景图像：改变大小到256*256
            3.街景语义图像：只归一化
            4.风险图：不变
            以上的训练和验证采用相同的操作
            """
            BSVI_img = self.Normalize(BSVI_img)
            BSVI_img = self.bsvi_img_val_resize(BSVI_img)
            Satellite_img = self.Normalize(Satellite_img)
            Satellite_170_img = self.Normalize(Satellite_170_img)
            Road_img = Road_img[dali.newaxis]
            Risk_Map = Risk_Map[dali.newaxis]
            fuse_imgs = self.cat(Satellite_img,Satellite_170_img, Road_img, Risk_Map)
            fuse_imgs = self.val_resize(fuse_imgs)
            Satellite_img,Satellite_170_img, Road_img, Risk_Map = fuse_imgs[0:3,:,:],fuse_imgs[3:6,:,:],fuse_imgs[6:7,:,:],fuse_imgs[7:8,:,:]
            Risk_Map = Risk_Map[0]
        Risk_Label = self.Risk_Label.gpu()[dali.newaxis]
        if self.load_type == 'train' or self.load_type == 'val':
            return (Satellite_img, BSVI_img, Satellite_170_img,Road_img,Risk_Map,Risk_Label)
        elif self.load_type == 'cam':
            return (Satellite_img, BSVI_img, Satellite_170_img,Road_img,Risk_Map,Risk_Label)


    def iter_setup(self):
        try:
            if self.load_type == 'train' or self.load_type == 'val':
                Satellite_img,BSVI_img, Satellite_170_img,Road_img,Risk_Map,Risk_Label  = self.iterator.next()
                self.feed_input(self.Satellite_img, Satellite_img)
                self.feed_input(self.BSVI_img, BSVI_img)
                self.feed_input(self.Satellite_170_img, Satellite_170_img)
                self.feed_input(self.Road_img, Road_img)
                self.feed_input(self.Risk_Map, Risk_Map)
                self.feed_input(self.Risk_Label, Risk_Label)
            elif self.load_type == 'cam':
                Satellite_img, BSVI_img, Satellite_170_img, Risk_Map, Risk_Label = self.iterator.next()
                self.feed_input(self.Satellite_img, Satellite_img)
                self.feed_input(self.BSVI_img, BSVI_img)
                self.feed_input(self.Satellite_170_img, Satellite_170_img)
                self.feed_input(self.Risk_Map, Risk_Map)
                self.feed_input(self.Risk_Label, Risk_Label)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class CustomDALIGenericIterator(DALIGenericIterator):
    def __init__(self, length,  pipelines,output_map,load_type='train', **argw):
        self._len = length # dataloader 的长度
        output_map = output_map
        self.load_type = load_type
        super().__init__(pipelines, output_map, **argw)

    def __next__(self):
        batch = super().__next__()
        return self.parse_batch(batch)

    def __len__(self):
        return self._len

    def parse_batch(self, batch):
        Satellite_img = batch[0]['Satellite']
        BSVI_img = batch[0]['BSVI']
        Satellite_170_img = batch[0]['Satellite_170']
        Road_img = batch[0]['Road']
        Risk_Map = batch[0]['Risk_Map']
        Risk_Label = batch[0]['Risk_Label']
        if self.load_type == 'cam':
            index_list = batch[0]['index_list']
            return {'Satellite':Satellite_img,"BSVI": BSVI_img, "Satellite_170": Satellite_170_img,'Road':Road_img,'Risk_Map':Risk_Map,'Risk_Label':Risk_Label}
        elif self.load_type == 'train' or self.load_type == 'val':
            return {'Satellite':Satellite_img,"BSVI": BSVI_img, "Satellite_170": Satellite_170_img,'Road':Road_img,'Risk_Map':Risk_Map,'Risk_Label':Risk_Label}

if __name__ == '__main__':
    Data_root_path = '../Dataset/'
    train_inf = pd.read_csv(os.path.join(Data_root_path,'train.csv'))
    batch_size  = 32
    num_threads = 12
    load_type = 'train'
    train_eii = DataSource(batch_size=batch_size, Datainf=train_inf, Data_root_path= Data_root_path,shuffle=False,load_type=load_type,seq_number=4)
    train_pipe = SourcePipeline(batch_size=batch_size, num_threads=num_threads, device_id=0, external_data=train_eii,
                                modeltype=load_type,load_type=load_type)
    if load_type == 'train' or load_type == 'val' :
        train_iter = CustomDALIGenericIterator(len(train_eii) / batch_size, pipelines=[train_pipe],
                                               output_map=['Satellite', "BSVI", 'Satellite_170','Road' ,'Risk_Map', 'Risk_Label'],
                                               last_batch_padded=False,
                                               size=len(train_eii),
                                               last_batch_policy=LastBatchPolicy.PARTIAL,
                                               auto_reset=True,
                                               load_type=load_type)
    train_bar = tqdm(total=int(train_iter._len), iterable=train_iter, file=sys.stdout)
    start_time = datetime.datetime.now()
    for epochs in range(1):
        for i, batch in enumerate(train_bar):
            Satellite_img,BSVI_img, Satellite_170_img,Road_map, Risk_Map, Risk_Label = batch['Satellite'],batch['BSVI'],batch['Satellite_170'],batch['Road'],batch['Risk_Map'],batch['Risk_Label']
#             # if load_type == 'train':
#             #     Scene_img,Sce ne_sequence,Label = batch['SGS'],batch['DLS'],batch['Label']
#             # elif load_type == 'cam':
#             #     Scene_img, Scene_sequence, Label, index_list = batch['SGS'], batch['DLS'], batch['Label'], batch['index_list']
#             #     print(index_list)
#             Scene_img = Satellite_img[0,  :, :, :].cpu().numpy()
#             Road_map = Road_map[0, 0, :, :].cpu().numpy()
#             Risk_Map = Risk_Map[0, :, :].cpu().numpy()
#             BSVI_img = BSVI_img[0, :, :, :].cpu().numpy()
#             Satellite_170_img = Satellite_170_img[0, :, :, :].cpu().numpy()
#             Scene_img = Scene_img.transpose(1,2, 0)
#             BSVI_img = BSVI_img.transpose(1, 2, 0)
#             Satellite_170_img = Satellite_170_img.transpose(1, 2, 0)
#             plt.subplot(2, 3, 1)
#             plt.imshow(Scene_img)
#             plt.subplot(2, 3, 2)
#             plt.imshow(Road_map, cmap='hot', interpolation='nearest')
#             plt.subplot(2, 3, 3)
#             plt.imshow(Risk_Map, cmap='hot', interpolation='nearest')
#             plt.subplot(2, 3, 4)
#             plt.imshow(BSVI_img, cmap='hot', interpolation='nearest')
#             plt.subplot(2, 3, 5)
#             plt.imshow(Satellite_170_img, cmap='hot', interpolation='nearest')
#             plt.show()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('使用Dali循环的耗时为{}'.format(elapsed_time.total_seconds()))