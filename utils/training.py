import os
from datetime import datetime

import torch
import torch.nn.functional as F
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple, List, Optional, Dict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils.logger import CsvWriter
from collections import Counter


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str, args) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs


# ============================================================================
# [NEW] Domain-Completion Test-Time Augmentation (TTA) - giữ nguyên logic gốc
# ============================================================================
def global_evaluate_domain_tta(model: FederatedModel, test_dl: DataLoader,
                                domains_list: List[str], setting: str, name: str,
                                args, avg_mode: str = 'prob') -> list:
    has_domain_op = (
        hasattr(model, 'global_domain_operator')
        and hasattr(model, 'domain_to_id')
        and model.domain_to_id is not None
    )

    net = model.global_net
    status = net.training
    net.eval()

    if not has_domain_op:
        net.train(status)
        return global_evaluate(model, test_dl, setting, name, args)

    domain_operator = model.global_domain_operator.to(model.device)
    domain_operator.eval()
    num_domains = model.num_domains

    accs = []
    for j, dl in enumerate(test_dl):
        domain_label = domains_list[j]
        if domain_label not in model.domain_to_id:
            correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
            for images, labels in dl:
                with torch.no_grad():
                    images, labels = images.to(model.device), labels.to(model.device)
                    outputs = net(images)
                    _, max5 = torch.topk(outputs, 5, dim=-1)
                    labels = labels.view(-1, 1)
                    top1 += (labels == max5[:, 0:1]).sum().item()
                    top5 += (labels == max5).sum().item()
                    total += labels.size(0)
            accs.append(round(100 * top1 / total, 2))
            continue

        domain_id_src = model.domain_to_id[domain_label]

        top1, total = 0.0, 0.0
        for images, labels in dl:
            with torch.no_grad():
                images = images.to(model.device)
                labels = labels.to(model.device).view(-1, 1)

                f = net.features(images)
                content = domain_operator.invert_batch(f, domain_id_src)

                if avg_mode == 'prob':
                    acc_probs = 0.0
                    for d in range(num_domains):
                        v_f = domain_operator.transform_batch(content, d)
                        logits_d = net.classifier(v_f)
                        acc_probs = acc_probs + F.softmax(logits_d, dim=-1)
                    avg_probs = acc_probs / num_domains
                else:
                    acc_logits = 0.0
                    for d in range(num_domains):
                        v_f = domain_operator.transform_batch(content, d)
                        acc_logits = acc_logits + net.classifier(v_f)
                    avg_probs = F.softmax(acc_logits / num_domains, dim=-1)

                _, max1 = torch.topk(avg_probs, 1, dim=-1)
                top1 += (labels == max1).sum().item()
                total += labels.size(0)

        accs.append(round(100 * top1 / total, 2))

    net.train(status)
    return accs


# ============================================================================
# [NEW] Checkpoint saving helper
# ============================================================================
def save_checkpoint(net: torch.nn.Module, ckpt_path: str, epoch: int,
                     accs: list, mean_acc: float,
                     tta_accs: Optional[list] = None,
                     tta_mean_acc: Optional[float] = None,
                     extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'accs': accs,
        'mean_acc': mean_acc,
        'tta_accs': tta_accs,
        'tta_mean_acc': tta_mean_acc,
        'timestamp': datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)


# ============================================================================
# [NEW] Extract per-image embeddings + global class prototypes -> .npz
#       Dùng để vẽ t-SNE (embedding + prototype trên cùng 1 không gian)
# ============================================================================
def extract_embeddings_and_prototypes(model: FederatedModel, test_dl: DataLoader,
                                       domains_list: List[str], save_path: str,
                                       max_per_loader: Optional[int] = None) -> str:
    """
    Chạy global_net qua toàn bộ test_dl (mỗi domain 1 loader), lấy:
      - embeddings: feature vector trước classifier (net.features)
      - labels: nhãn class tương ứng
      - domains: tên domain tương ứng (string) cho mỗi sample
    Sau đó tính prototype = mean-embedding theo từng cặp (class, domain)
    — tức mỗi (class, domain) có 1 prototype riêng, KHÔNG gộp domain lại
    như trước. Điều này cho phép t-SNE so sánh domain-shift của cùng 1 class
    giữa các domain khác nhau.
    Lưu tất cả vào 1 file .npz để script vẽ t-SNE dùng chung.

    max_per_loader: nếu set, chỉ lấy tối đa N sample/loader (tránh file quá nặng
                     khi tập test lớn, vẫn đủ để vẽ t-SNE).
    """
    net = model.global_net
    status = net.training
    net.eval()

    all_feats, all_labels, all_domains = [], [], []

    with torch.no_grad():
        for j, dl in enumerate(test_dl):
            domain_label = domains_list[j] if j < len(domains_list) else f'domain_{j}'
            collected = 0
            for images, labels in dl:
                images = images.to(model.device)
                feats = net.features(images)
                feats = feats.detach().cpu().numpy()
                labs = labels.detach().cpu().numpy()

                if max_per_loader is not None:
                    remaining = max_per_loader - collected
                    if remaining <= 0:
                        break
                    if feats.shape[0] > remaining:
                        feats = feats[:remaining]
                        labs = labs[:remaining]

                all_feats.append(feats)
                all_labels.append(labs)
                all_domains.extend([domain_label] * feats.shape[0])
                collected += feats.shape[0]

    net.train(status)

    if len(all_feats) == 0:
        raise RuntimeError("No embeddings collected — check that test_dl is non-empty.")

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.array(all_domains)

    # --- Prototype theo key (class, domain): mean embedding của từng cặp ---
    unique_pairs = sorted(set(zip(all_labels.tolist(), all_domains.tolist())))
    proto_list = []
    proto_labels = []
    proto_domains = []
    proto_counts = []
    for c, d in unique_pairs:
        mask = (all_labels == c) & (all_domains == d)
        n = int(mask.sum())
        if n == 0:
            continue
        proto_list.append(all_feats[mask].mean(axis=0))
        proto_labels.append(c)
        proto_domains.append(d)
        proto_counts.append(n)

    prototypes = np.stack(proto_list, axis=0)                 # (num_class*num_domain, dim)
    prototype_labels = np.array(proto_labels, dtype=np.int64)  # (P,)
    prototype_domains = np.array(proto_domains)                 # (P,) string
    prototype_counts = np.array(proto_counts, dtype=np.int64)  # (P,) số sample đóng góp

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        embeddings=all_feats,                   # (N, dim)
        labels=all_labels,                      # (N,)
        domains=all_domains,                    # (N,) string array
        prototypes=prototypes,                  # (P, dim) — P = num (class, domain) pairs found
        prototype_labels=prototype_labels,      # (P,) class id ứng với từng prototype
        prototype_domains=prototype_domains,    # (P,) domain name ứng với từng prototype
        prototype_counts=prototype_counts,      # (P,) số ảnh dùng để tính mean prototype đó
    )
    print(f"[Embedding] Saved {all_feats.shape[0]} embeddings (dim={all_feats.shape[1]}), "
          f"{prototypes.shape[0]} (class, domain) prototypes -> {save_path}")
    return save_path


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    # ------------------------------------------------------------------
    # [NEW] Thư mục lưu checkpoint / embedding — truyền qua args
    #   args.save_dir          : thư mục gốc (mặc định './output')
    #   args.checkpoint_dir    : override riêng, mặc định {save_dir}/checkpoints
    #   args.embedding_dir     : override riêng, mặc định {save_dir}/embeddings
    #   args.embed_max_per_domain : giới hạn số sample/domain khi extract (mặc định None = lấy hết)
    # ------------------------------------------------------------------
    save_dir = getattr(args, 'save_dir', '/users/grad/nphan/work/FedDAP_CVPR2026/visualization')
    checkpoint_dir = getattr(args, 'checkpoint_dir', os.path.join(save_dir, 'checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_tag = f"{private_dataset.NAME}_{model.args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_tag}_best.pth")
    embedding_save_path = os.path.join(checkpoint_dir, f"{run_tag}_embeddings.npz")

    best_acc = -1.0
    best_epoch = -1

    print("🔹 Running Domain Skew Setting")
    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_pacs':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True,
                                                        p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_vlcs':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_domainnet':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        if model.args.dataset == 'fl_digits':
            selected_domain_dict = {'mnist': 5, 'usps': 5, 'svhn': 5, 'syn': 5}
        elif model.args.dataset == 'fl_pacs':
            selected_domain_dict = {'photo': 3, 'art_painting': 3, 'cartoon': 3, 'sketch': 3}
        elif model.args.dataset == 'fl_officecaltech':
            selected_domain_dict = {'caltech': 3, 'amazon': 3, 'webcam': 3, 'dslr': 3}
        elif model.args.dataset == 'fl_domainnet':
            selected_domain_dict = {'clipart': 3, 'infograph': 1, 'painting': 4, 'quickdraw': 6, 'real': 4, 'sketch': 2}

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = pri_train_loaders
    model.client_domains = selected_domain_list
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

    use_domain_tta = getattr(args, 'use_domain_tta', False)
    if use_domain_tta:
        tta_accs_dict = {}
        tta_mean_accs_list = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders, epoch_index)
        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME, args)
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), 'Method:', model.args.model)
        print(accs)

        tta_accs = None
        tta_mean_acc = None
        if use_domain_tta:
            tta_accs = global_evaluate_domain_tta(
                model, test_loaders, domains_list, private_dataset.SETTING,
                private_dataset.NAME, args, avg_mode=getattr(args, 'tta_avg_mode', 'prob'))
            tta_mean_acc = round(np.mean(tta_accs, axis=0), 3)
            tta_mean_accs_list.append(tta_mean_acc)
            for i in range(len(tta_accs)):
                if i in tta_accs_dict:
                    tta_accs_dict[i].append(tta_accs[i])
                else:
                    tta_accs_dict[i] = [tta_accs[i]]
            print('The ' + str(epoch_index) + ' Communcation Accuracy (Domain-TTA):', str(tta_mean_acc))
            print(tta_accs)

        # ----------------------------------------------------------------
        # [NEW] Lưu best checkpoint. Ưu tiên TTA-mean-acc nếu bật (vì đó là
        # số cuối cùng dùng để báo cáo trong paper), fallback về mean_acc.
        # ----------------------------------------------------------------
        monitor_acc = tta_mean_acc if (use_domain_tta and tta_mean_acc is not None) else mean_acc
        if monitor_acc > best_acc:
            best_acc = monitor_acc
            best_epoch = epoch_index
            save_checkpoint(
                model.global_net, best_ckpt_path, epoch_index,
                accs, mean_acc, tta_accs, tta_mean_acc,
                extra={'monitor_metric': 'tta_mean_acc' if use_domain_tta else 'mean_acc'}
            )
            print(f"[Checkpoint] New best acc={best_acc} @ epoch {epoch_index} -> {best_ckpt_path}")

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)

    # ----------------------------------------------------------------------
    # [NEW] Sau khi train xong: load lại best checkpoint, extract embedding
    # (image feature) + global prototype, lưu chung vào 1 file .npz để
    # dùng cho script vẽ t-SNE.
    # ----------------------------------------------------------------------
    if os.path.exists(best_ckpt_path):
        print(f"[Embedding] Loading best checkpoint (epoch {best_epoch}, acc={best_acc}) from {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=model.device, weights_only=False)
        model.global_net.load_state_dict(ckpt['model_state_dict'])

        extract_embeddings_and_prototypes(
            model, test_loaders, domains_list, embedding_save_path,
            max_per_loader=getattr(args, 'embed_max_per_domain', None),
        )
    else:
        print("[Embedding] No checkpoint was saved (unexpected) — skipping embedding extraction.")