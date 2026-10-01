from collections import defaultdict

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import math
from utils.finch import FINCH
import numpy as np
from utils.augmentation import random_data_augmentation, strong_augmentation,weak_augmentation,random_flip,random_crop,color_jitter,random_erasing, horizontal_flip,add_gaussian_noise,add_uniform_noise,mixup
from utils.util import *
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def agg_func(protos):
    for key, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = torch.zeros_like(proto_list[0].data)
            for i in proto_list:
                proto += i.data
            protos[key] = proto / len(proto_list)
        else:
            protos[key] = proto_list[0]
    return protos


class DomainOperator(nn.Module):
    def __init__(self, num_domains, feat_dim, rank=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.rank = rank
        if rank is None:
            self.W = nn.ParameterList([
                nn.Parameter(torch.eye(feat_dim) + torch.randn(feat_dim, feat_dim) * 0.01)
                for _ in range(num_domains)
            ])
        else:
            self.U = nn.ParameterList([
                nn.Parameter(torch.randn(feat_dim, rank) * 0.01) for _ in range(num_domains)
            ])
            self.V = nn.ParameterList([
                nn.Parameter(torch.randn(feat_dim, rank) * 0.01) for _ in range(num_domains)
            ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(feat_dim)) for _ in range(num_domains)
        ])
        self.register_buffer('_eye', torch.eye(feat_dim), persistent=False)

    # def get_W(self, domain_id):
    #     if self.rank is None:
    #         return self.W[domain_id]
    #     return self.U[domain_id] @ self.V[domain_id].T
    
    def get_W(self, domain_id):
        if self.rank is None:
            return self.W[domain_id]
        I = torch.eye(self.feat_dim, device=self.U[domain_id].device)
        return I + self.U[domain_id] @ self.V[domain_id].T

    def transform_batch(self, content_batch, domain_id):
        W_d = self.get_W(domain_id)
        return content_batch @ W_d.T + self.b[domain_id]

    def invert_batch(self, f_batch, domain_id, eps=1e-4):
        W_d = self.get_W(domain_id)
        I = self._eye.to(device=W_d.device, dtype=W_d.dtype)  # [OPT] không re-alloc mỗi lần gọi
        W_d_inv = torch.linalg.inv(W_d + eps * I)
        return (f_batch - self.b[domain_id]) @ W_d_inv.T

    def identity_regularization(self, domain_id):
        """Kéo W_d về gần identity - an toàn khi operator chưa học đủ tốt."""
        W_d = self.get_W(domain_id)
        identity = self._eye.to(device=W_d.device, dtype=W_d.dtype)
        return ((W_d - identity) ** 2).mean() + (self.b[domain_id] ** 2).mean()


def synthesize_all_domains(domain_operator, f_batch, own_domain_id, num_domains):
    content = domain_operator.invert_batch(f_batch, own_domain_id)
    virtual = {}
    for d in range(num_domains):
        if d == own_domain_id:
            continue
        virtual[d] = domain_operator.transform_batch(content, d)
    return virtual


def build_domain_proto_grid(global_protos, domain_label, num_classes, feat_dim, device):
    grid = torch.zeros(num_classes, feat_dim, device=device)
    valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for cls in range(num_classes):
        key = (cls, domain_label)
        if key in global_protos:
            target = global_protos[key]
            target = target[0] if isinstance(target, list) else target
            grid[cls] = target.detach()
            valid[cls] = True
    return grid, valid


class feddope(FederatedModel):
    NAME = 'feddope'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(feddope, self).__init__(nets_list, args, transform)
        self.global_protos = {}
        self.prev_global_protos = {}
        self.intra_proto = []
        self.local_protos = {}
        self.local_protos_augment = {}
        self.infoNCET = args.infoNCET

        # ---------------- Domain Completion setup ----------------
        self.num_domains = len(args.client_domains)
        self.num_classes = args.num_classes
        self.feat_dim = 512
        operator_rank = getattr(args, 'domain_operator_rank', None)  # None = full matrix

        self.global_domain_operator = DomainOperator(self.num_domains, self.feat_dim, operator_rank)
        self.domain_operator_list = [copy.deepcopy(self.global_domain_operator) for _ in nets_list]

        self.domain_to_id = None
        self.current_epoch = 0

        self._proto_tensor_all = None   # (num_domains, num_classes, feat_dim)
        self._proto_valid_all = None    # (num_domains, num_classes)
        from utils.prototype_evolution import EvolutionManager
        self.evolution_manager = EvolutionManager(
            match_similarity_threshold=0.7,
            birth_candidate_sim_threshold=0.9,
            birth_min_clients=2,
            birth_min_rounds=2,
            merge_sim_threshold=0.97,
            merge_min_reliability=0.9,
            max_prototypes_per_key=3,
            update_lr=1.0
        )

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        global_op_w = self.domain_operator_list[0].state_dict()
        self.global_domain_operator.load_state_dict(global_op_w)
        for op in self.domain_operator_list:
            op.load_state_dict(global_op_w)

        if hasattr(self, 'client_domains'):
            unique_domains = sorted(set(self.client_domains))
            self.domain_to_id = {d: i for i, d in enumerate(unique_domains)}
            self.id_to_domain = {i: d for d, i in self.domain_to_id.items()}
            assert len(unique_domains) <= self.num_domains, \
                f"Số domain thực tế ({len(unique_domains)}) vượt quá args.num_domains ({self.num_domains})"

    def proto_aggregation_attn(self, local_protos_list, temperature=0.1):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for key in local_protos.keys():
                if key in agg_protos_label:
                    agg_protos_label[key].append(local_protos[key])
                else:
                    agg_protos_label[key] = [local_protos[key]]

        for key, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                stacked = torch.stack([p.detach() for p in proto_list])
                sim_matrix = F.cosine_similarity(stacked.unsqueeze(1), stacked.unsqueeze(0), dim=-1)
                attn_weights = torch.softmax(sim_matrix.sum(dim=1) / temperature, dim=0)
                proto = torch.sum(attn_weights.view(-1, 1) * stacked, dim=0)
                agg_protos_label[key] = proto
            else:
                agg_protos_label[key] = proto_list[0].data

        return agg_protos_label

    def _refresh_proto_cache(self):
        if len(self.global_protos) == 0:
            self._proto_tensor_all = None
            self._proto_valid_all = None
            return
        device = self.device
        proto_tensor = torch.zeros(self.num_domains, self.num_classes, self.feat_dim, device=device)
        proto_valid = torch.zeros(self.num_domains, self.num_classes, dtype=torch.bool, device=device)
        for (cls, domain_label), proto in self.global_protos.items():
            if domain_label not in self.domain_to_id:
                continue
            d = self.domain_to_id[domain_label]
            p = proto[0] if isinstance(proto, list) else proto
            proto_tensor[d, cls] = p.detach()
            proto_valid[d, cls] = True
        self._proto_tensor_all = proto_tensor
        self._proto_valid_all = proto_valid

    def hierarchical_info_loss_batch(self, f_batch, labels, exclude_domain_id):
        if self._proto_tensor_all is None:
            return torch.tensor(0.0, device=self.device)

        proto_tensor = self._proto_tensor_all   # (num_domains, num_classes, D)
        proto_valid = self._proto_valid_all     # (num_domains, num_classes)

        domain_mask = torch.ones(self.num_domains, dtype=torch.bool, device=f_batch.device)
        domain_mask[exclude_domain_id] = False

        protos = proto_tensor[domain_mask]      # (num_domains-1, num_classes, D)
        valid = proto_valid[domain_mask]        # (num_domains-1, num_classes)
        if protos.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        protos_flat = protos.reshape(-1, self.feat_dim)                       # (N, D)
        valid_flat = valid.reshape(-1)                                        # (N,)
        class_idx_flat = torch.arange(self.num_classes, device=f_batch.device).repeat(protos.shape[0])  # (N,)

        if valid_flat.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        f_norm = F.normalize(f_batch, dim=1)
        p_norm = F.normalize(protos_flat, dim=1)

        sim = (f_norm @ p_norm.T) / self.args.infoNCET   # (B, N)

        pos_mask = (class_idx_flat.unsqueeze(0) == labels.unsqueeze(1)) & valid_flat.unsqueeze(0)  # (B,N)
        neg_mask = (~pos_mask) & valid_flat.unsqueeze(0)

        exp_sim = torch.exp(sim) * valid_flat.unsqueeze(0)  # zero-out entries không hợp lệ

        sum_pos = (exp_sim * pos_mask).sum(dim=1)
        sum_all = (exp_sim * (pos_mask | neg_mask)).sum(dim=1)

        has_both = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if has_both.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        per_sample_loss = -torch.log(sum_pos[has_both] / (sum_all[has_both] + 1e-12) + 1e-12)
        return per_sample_loss.mean()

    def loc_update(self, priloader_list, epoch):
        self.current_epoch = epoch
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        print(self.online_clients)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], self.domain_operator_list[i],
                             priloader_list[i], self.client_domains[i])

        self.global_protos = self.evolution_manager.step(self.local_protos)
        self._refresh_proto_cache()  # [OPT] build cache 1 lần/round, dùng cho round kế tiếp
        self.aggregate_nets(None)
        self._aggregate_domain_operator()
        if epoch % 10 == 0:
            print(self.evolution_manager.genealogy_log[-5:])
        return None

    def _aggregate_domain_operator(self):
        online_clients = self.online_clients
        n = len(online_clients)
        global_sd = self.global_domain_operator.state_dict()
        first = True
        for cid in online_clients:
            sd = self.domain_operator_list[cid].state_dict()
            if first:
                first = False
                for k in sd:
                    global_sd[k] = sd[k] / n
            else:
                for k in sd:
                    global_sd[k] += sd[k] / n
        self.global_domain_operator.load_state_dict(global_sd)
        for op in self.domain_operator_list:
            op.load_state_dict(self.global_domain_operator.state_dict())

    def _train_net(self, index, net, domain_operator, train_loader, domain_label):
        net = net.to(self.device)
        domain_operator = domain_operator.to(self.device)
        domain_id = self.domain_to_id[domain_label]

        optimizer = optim.SGD(
            list(net.parameters()) + list(domain_operator.parameters()),
            lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        proto_grids = {}
        own_proto_grid, own_proto_valid = None, None
        if len(self.global_protos) > 0:
            for d in range(self.num_domains):
                if d != domain_id:
                    domain_str = self.id_to_domain[d]
                    proto_grids[d] = build_domain_proto_grid(
                        self.global_protos, domain_str, self.num_classes, self.feat_dim, self.device)
            own_proto_grid, own_proto_valid = build_domain_proto_grid(
                self.global_protos, domain_label, self.num_classes, self.feat_dim, self.device)

        ramp_rounds = getattr(self.args, 'synth_ramp_rounds', 20)
        ramp = min(1.0, self.current_epoch / max(1, ramp_rounds))

        iterator = tqdm(range(self.local_epoch))

        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)

                f = net.features(images)
                outputs = net.classifier(f)
                lossCE = criterion(outputs, labels)

                if len(self.global_protos) == 0:
                    loss_InfoNCE = torch.tensor(0.0, device=self.device)
                    loss_intra_proto = torch.tensor(0.0, device=self.device)
                    loss_virtual_ce = torch.tensor(0.0, device=self.device)
                    loss_virtual_infonce = torch.tensor(0.0, device=self.device)
                    loss_virtual_dpa = torch.tensor(0.0, device=self.device)
                    loss_identity_reg = torch.tensor(0.0, device=self.device)
                else:
                    loss_InfoNCE = self.hierarchical_info_loss_batch(f, labels, exclude_domain_id=domain_id)

                    f_detached = f.detach()
                    if own_proto_valid is not None and own_proto_valid.any():
                        own_targets = own_proto_grid[labels]          # (B, D)
                        own_mask = own_proto_valid[labels].unsqueeze(1)  # (B, 1)
                        proto_new = torch.where(own_mask, own_targets, f_detached)
                    else:
                        proto_new = f_detached
                    loss_intra_proto = 1 - F.cosine_similarity(
                        F.normalize(proto_new, dim=1), F.normalize(f, dim=1), dim=1).mean()

                    virtual_feats = synthesize_all_domains(domain_operator, f, domain_id, self.num_domains)

                    loss_virtual_ce = torch.tensor(0.0, device=self.device)
                    loss_virtual_infonce = torch.tensor(0.0, device=self.device)
                    loss_virtual_dpa = torch.tensor(0.0, device=self.device)
                    n_virtual = 0

                    for d, v_f in virtual_feats.items():
                        outputs_virtual = net.classifier(v_f)
                        loss_virtual_ce = loss_virtual_ce + criterion(outputs_virtual, labels)
                        
                        loss_virtual_infonce = loss_virtual_infonce + self.hierarchical_info_loss_batch(v_f, labels, exclude_domain_id=d)

                        if d in proto_grids:
                            grid, valid = proto_grids[d]
                            targets = grid[labels]
                            mask = valid[labels]
                            if mask.sum() > 0:
                                cos_sim = F.cosine_similarity(v_f[mask], targets[mask], dim=1)
                                loss_virtual_dpa = loss_virtual_dpa + (1 - cos_sim).mean()

                        n_virtual += 1

                    if n_virtual > 0:
                        loss_virtual_ce = loss_virtual_ce / n_virtual
                        loss_virtual_infonce = loss_virtual_infonce / n_virtual
                        loss_virtual_dpa = loss_virtual_dpa / n_virtual

                    loss_identity_reg = torch.tensor(0.0, device=self.device)
                    for d in range(self.num_domains):
                        if d != domain_id:
                            loss_identity_reg = loss_identity_reg + domain_operator.identity_regularization(d)
                    loss_identity_reg = loss_identity_reg / max(1, self.num_domains - 1)

                loss = (lossCE
                        + self.args.lamda_1 * loss_intra_proto
                        + self.args.lamda_2 * loss_InfoNCE
                        + self.args.lamda_2 * loss_virtual_infonce
                        + self.args.lamda_1 * loss_virtual_dpa
                        + ramp * getattr(self.args, 'lamda_virtual_ce', 0.5) * loss_virtual_ce
                        + getattr(self.args, 'lamda_identity_reg', 0.1) * loss_identity_reg)

                loss.backward()
                iterator.desc = ("P%d CE=%.3f InfoNCE=%.3f Intra=%.3f | VirCE=%.3f(ramp=%.2f) "
                                  "VirInfoNCE=%.3f VirDPA=%.3f IdReg=%.4f"
                                  % (index, lossCE, loss_InfoNCE, loss_intra_proto,
                                     loss_virtual_ce, ramp, loss_virtual_infonce,
                                     loss_virtual_dpa, loss_identity_reg))
                optimizer.step()

                if iter == self.local_epoch - 1:
                    for i in range(len(labels)):
                        key = (labels[i].item(), domain_label)
                        if key in agg_protos_label:
                            agg_protos_label[key].append(f[i, :])
                        else:
                            agg_protos_label[key] = [f[i, :]]

        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
