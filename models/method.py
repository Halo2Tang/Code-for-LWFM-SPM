import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from models.matcher import HungarianMatcher, NNMatcher
from models.resnet.config_networks import ConfigureNetworks
import matplotlib.pyplot as plt
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
plt.rcParams['font.sans-serif'] = ['Times New Roman']

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)


class Method(nn.Module):
    def __init__(self,
                 args,
                 mode=None,
                 feature_size=3,
                 first_padding=3,
                 hyperpixel_pooling='average'):
        super().__init__()
        self.mode = mode
        self.args = args
        self.hyperpixel_ids = args.hyperpixel_ids

        self.networks = ConfigureNetworks(
            feature_size=feature_size,
            hyperpixel_ids=self.hyperpixel_ids,
            pretrained_resnet_path=None,
            first_padding=first_padding,
            hyperpixel_pooling=hyperpixel_pooling)
        self.feature_extractor = self.networks.get_feature_extractor()

        self.encoder_dim = self.feature_extractor.encoder_dim

        self.matcher_prob = self.args.matcher_prob
        if self.matcher_prob > 0:
            self.matchers = nn.ModuleList([
                 HungarianMatcher(dim, self.args.learnable_matcher) for dim in self.feature_extractor.hyperpixel_dims
            ])
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.l2norm = FeatureL2Norm()

        self.feature_size = feature_size
        self.args = args

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def encode(self, x):
        x = self.feature_extractor(x)
        x = torch.cat(x, dim=1)
        return x

    def corr(self, src, trg):
        corr = src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
        return corr

    def cca(self, spt, qry):
        spt = spt.squeeze(0)
        way = spt.shape[0]
        spt = self.normalize_feature(spt) # [25,960,3,3]

        num_qry = qry.shape[0]
        qry = self.normalize_feature(qry) # [75,960,3,3]

        # ----------------------------------cat--------------------------------------#
        hyperpixel_dims = self.feature_extractor.hyperpixel_dims
        spt_feats = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1).view(-1, *spt.size()[1:])  # [75x25,960,3,3]
        qry_feats = qry.unsqueeze(1).repeat(1, way, 1, 1, 1).view(-1, *qry.size()[1:])  # [75x25,960,3,3]
        spt_feats = torch.split(spt_feats, hyperpixel_dims, dim=1)
        qry_feats = torch.split(qry_feats, hyperpixel_dims, dim=1)
        corrs = []
        for i, (src, tgt) in enumerate(zip(spt_feats, qry_feats)):
            src = self.l2norm(src) # [75x25,c,3,3]
            tgt = self.l2norm(tgt) # [75x25,c,3,3]
            corr = self.corr(src, tgt) # [75x25,9,9]
            corrs.append(corr)

        corr = torch.stack(corrs, dim=1)  # [75x25,4,9,9]

        refined_corr = corr.view(num_qry, way, len(hyperpixel_dims), *[self.feature_size] * 4)

        corr_s = refined_corr.view(num_qry, way, len(hyperpixel_dims), self.feature_size * self.feature_size, self.feature_size,
                                   self.feature_size) # [num_qry, way, L, shw, qh, qw]
        corr_q = refined_corr.view(num_qry, way, len(hyperpixel_dims), self.feature_size, self.feature_size,
                                   self.feature_size * self.feature_size) # [num_qry, way, L, sh, sw, qhw]

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr_s = self.gaussian_normalize(corr_s, dim=3)
        corr_q = self.gaussian_normalize(corr_q, dim=5)

        # applying softmax for each side
        corr_s = F.softmax(corr_s / self.args.temperature_attn, dim=3) # spt hyperpixel softmax
        corr_s = corr_s.view(num_qry, way, len(hyperpixel_dims), self.feature_size, self.feature_size, self.feature_size, self.feature_size)
        corr_q = F.softmax(corr_q / self.args.temperature_attn, dim=5) # query hyperpixel softmax
        corr_q = corr_q.view(num_qry, way, len(hyperpixel_dims), self.feature_size, self.feature_size, self.feature_size, self.feature_size)

        # suming up matching scores
        attn_s = corr_s.sum(dim=[5, 6]) # [num_qry, way, L, sh, sw]
        attn_q = corr_q.sum(dim=[3, 4]) # [num_qry, way, L, qh, qw]

        # applying attention
        spt_attended = []
        for i, channel in enumerate(hyperpixel_dims):
            spt_attended.append(attn_s[:,:,i,:,:].unsqueeze(2).expand(-1,-1,channel,-1,-1))
        spt_attended = torch.cat(spt_attended, dim=2) # [num_qry, way, c, sh, sw]
        spt_attended = spt_attended * spt.unsqueeze(0) # [num_qry, way, c, sh, sw] * [1, way, c, sh, sw] = [num_qry, way, c, sh, sw]

        qry_attended = []
        for i, channel in enumerate(hyperpixel_dims):
            qry_attended.append(attn_q[:,:,i,:,:].unsqueeze(2).expand(-1,-1,channel,-1,-1))
        qry_attended = torch.cat(qry_attended, dim=2) # [num_qry, way, c, qh, qw]
        qry_attended = qry_attended * qry.unsqueeze(1) # [num_qry, way, c, qh, qw] * [num_qry, 1, c, qh, qw] = [num_qry, way, c, qh, qw]

        if self.matcher_prob > 0:
            similarity_matrixs = []
            spt_feats = spt_attended.flatten(0, 1).flatten(-2, -1).permute(0,2,1)
            qry_feats = qry_attended.flatten(0, 1).flatten(-2, -1).permute(0,2,1)
            spt_feats = torch.split(spt_feats, hyperpixel_dims, dim=-1)
            qry_feats = torch.split(qry_feats, hyperpixel_dims, dim=-1)
            for i, (spt_feat, qry_feat) in enumerate(zip(spt_feats, qry_feats)):
                spt_feat, qry_feat = self.matchers[i](spt_feat, qry_feat)
                spt_feat = spt_feat.view(num_qry, way, self.feature_size**2, -1)
                qry_feat = qry_feat.view(num_qry, way, self.feature_size**2, -1)
                similarity_matrix = F.cosine_similarity(spt_feat, qry_feat, dim=-1)
                similarity_matrix, _ = similarity_matrix.topk(1, dim=-1)
                similarity_matrix = similarity_matrix.sum(dim=-1)
                similarity_matrix *= self.matcher_prob
                similarity_matrixs.append(similarity_matrix)
            similarity_matrix, _ = torch.stack(similarity_matrixs, dim=1).max(dim=1) # [num_qry, way]

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        spt_attended_pooled = spt_attended.mean(dim=[-1, -2]) # [num_qry, way, c]
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2]) # [num_qry, way, c]
        if self.matcher_prob > 0:
            similarity_matrix = similarity_matrix.view(num_qry, self.args.shot, self.args.way).mean(dim=1)
            base_similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
            similarity_matrix += base_similarity_matrix
        else:
            similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_attended_pooled.mean(dim=1))
        else:
            return similarity_matrix / self.args.temperature

    # def visualize(self, t1, t2):
    #
    #     t1 = t1.view(75 * 5, 1024).cpu().detach()
    #     t = t1[0].unsqueeze(0)
    #     offset = 1
    #     for i in range(5, 75 * 5, 5):
    #         t = torch.cat([t, t1[i + offset].unsqueeze(0)], dim=0)
    #         offset = (offset + 1) % 5
    #     label1 = [1, 2, 3, 4, 5] * 15
    #
    #     t2 = t2.view(75 * 5, 1024).cpu().detach()
    #     offset = 0
    #     for i in range(0, 75 * 5, 5):
    #         t = torch.cat([t, t2[i + offset].unsqueeze(0)], dim=0)
    #         offset = (offset + 1) % 5
    #     label2 = [6, 7, 8, 9, 10] * 15
    #
    #     label = label1 + label2
    #
    #     ts = TSNE(n_components=2, learning_rate="auto", init="pca", random_state=33)
    #     result = ts.fit_transform(t)
    #     self.plot_embedding(result, label, '')

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    # def plot_embedding(self, data, label, title):
    #     # plt.rc('font',family='Times New Roman')
    #     x_min, x_max = np.min(data, 0), np.max(data, 0)
    #     data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    #     fig = plt.figure()  # 创建图形实例
    #     ax = plt.subplot(111)  # 创建子图
    #     # 遍历所有样本
    #
    #     for i in range(data.shape[0]):
    #         # 在图中为每个数据点画出标签
    #         plt.text(data[i, 0], data[i, 1], '.' if label[i] <= 5 else 'X', color=plt.cm.Set1(label[i] % 5),
    #                  fontdict={'weight': 'bold', 'size': 28 if label[i] <= 5 else 14})
    #     plt.xticks(fontproperties="Times New Roman")  # 指定坐标的刻度
    #     plt.yticks(fontproperties="Times New Roman")
    #     # plt.xticks()		# 指定坐标的刻度
    #     # plt.yticks()
    #     plt.title("Distribution of task" + str(self.args.visualfile) + " generated by ML", fontsize=14)
    #     plt.savefig("/data/data-home/chenderong/work/MCNet/visualizes/" + str(self.args.visualfile) + ".pdf",
    #                 format="pdf")
    #     # 返回值
    #     return fig
