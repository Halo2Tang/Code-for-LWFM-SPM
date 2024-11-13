import torch
import torch.nn as nn
import torch.nn.functional as F


from models.resnet.config_networks import ConfigureNetworks


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
            freeze_feature_extractor=False,
            first_padding=first_padding,
            hyperpixel_pooling=hyperpixel_pooling)
        self.feature_extractor = self.networks.get_feature_extractor()

        self.encoder_dim = self.feature_extractor.encoder_dim

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
        spt_feats = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1).view(-1, *spt.size()[1:])  # [75x25,960,3,3]
        qry_feats = qry.unsqueeze(1).repeat(1, way, 1, 1, 1).view(-1, *qry.size()[1:])  # [75x25,960,3,3]

        spt_feats = self.l2norm(spt_feats)
        qry_feats = self.l2norm(qry_feats)

        corr = self.corr(spt_feats, qry_feats)

        refined_corr = corr.view(num_qry, way, *[self.feature_size] * 4)
        
        corr_s = refined_corr.view(num_qry, way, self.feature_size * self.feature_size, self.feature_size,
                                   self.feature_size) # [num_qry, way, shw, qh, qw]
        corr_q = refined_corr.view(num_qry, way, self.feature_size, self.feature_size,
                                   self.feature_size * self.feature_size) # [num_qry, way, sh, sw, qhw]
        
        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr_s = self.gaussian_normalize(corr_s, dim=2)
        corr_q = self.gaussian_normalize(corr_q, dim=4)

        # applying softmax for each side
        corr_s = F.softmax(corr_s / self.args.temperature_attn, dim=2) # spt hyperpixel softmax
        corr_s = corr_s.view(num_qry, way, self.feature_size, self.feature_size, self.feature_size, self.feature_size)
        corr_q = F.softmax(corr_q / self.args.temperature_attn, dim=4) # query hyperpixel softmax
        corr_q = corr_q.view(num_qry, way, self.feature_size, self.feature_size, self.feature_size, self.feature_size)

        # suming up matching scores
        attn_s = corr_s.sum(dim=[4, 5]) # [num_qry, way, sh, sw]
        attn_q = corr_q.sum(dim=[2, 3]) # [num_qry, way, qh, qw]

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0) # [num_qry, way, 1, sh, sw] * [1, way, c, sh, sw] = [num_qry, way, c, sh, sw]
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1) # [num_qry, way, 1, qh, qw] * [num_qry, 1, c, qh, qw] = [num_qry, way, c, qh, qw]

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        spt_attended_pooled = spt_attended.mean(dim=[-1, -2]) # [num_qry, way, c]
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2]) # [num_qry, way, c]
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_attended_pooled.mean(dim=1))
        else:
            return similarity_matrix / self.args.temperature

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x