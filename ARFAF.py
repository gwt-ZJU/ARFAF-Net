import torch
from torch import flatten, nn
from Resnet import resnet50 as SFE
from HFAF import HFAF,HFCGF
from RFCP import MRI
from GFGF import GFGF
from MSFAE import MSFAE
from GCDCA import GCDCA_Cell


class MSFE(nn.Module):
    def __init__(self):
        super(MSFE, self).__init__()
        self.ST_Branch = SFE(include_top=False,in_dim=3)
        self.SV_Branch = SFE(include_top=False,in_dim=3)
        self.G_SV_Branch = SFE(include_top=False, in_dim=3,branch_type='G_ST')
    def forward(self,ST_img,SV_img,G_ST_img):
        st_1,st_2,st_3,st_4 = self.ST_Branch(ST_img)
        sv_1, sv_2, sv_3, sv_4 = self.SV_Branch(SV_img)
        G_ST_list = self.G_SV_Branch(G_ST_img)
        return [st_1,sv_1],[st_2,sv_2],[st_3,sv_3],[st_4,sv_4],G_ST_list


class ARFAF_Net(nn.Module):
    def __init__(self,HFAF_Flag=True,GCDCA_flag=True,MPGF_flag=True):
        super(ARFAF_Net, self).__init__()
        self.MSFE = MSFE()
        self.HFAF = HFAF(Flag=HFAF_Flag)
        self.MRI = MRI(GCDCA_flag=GCDCA_flag,MPGF_flag=MPGF_flag)

    
    def forward(self,ST_img,G_ST_img,SV_img):
        t_v_1,t_v_2,t_v_3,t_v_4,G_ST_list = self.MSFE(ST_img,SV_img,G_ST_img)
        Fuse_feature_1, Fuse_feature_2, Fuse_feature_3 = self.HFAF(t_v_1,t_v_2,t_v_3)
        Risk_map = self.MRI(t_v_4[0],Fuse_feature_1, Fuse_feature_2, Fuse_feature_3,G_ST_list)
        return Risk_map



if __name__ == '__main__':
    ST_img = torch.randn(4, 3, 256, 256).cuda()
    G_ST_img = torch.randn(4, 3, 256, 256).cuda()
    SV_img = torch.randn(4, 3, 256, 256).cuda()
    Seg_img = torch.rand(4,3,256, 512 ).cuda()
    model = ARFAF_Net(HFAF_Flag=True,GCDCA_flag=True,MPGF_flag=False).cuda()
    Risk_map,Risk_leve = model(ST_img,G_ST_img,SV_img)
    pass