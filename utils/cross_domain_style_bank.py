import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization - wrap quanh 1 base optimizer (SGD/Adam...).

    Cách dùng thay cho optimizer.step() thông thường:

        optimizer = SAM(net.parameters(), base_optimizer=torch.optim.SGD,
                         rho=0.05, lr=local_lr, momentum=0.9, weight_decay=1e-5)

        for images, labels in train_loader:
            # ---- forward/backward lần 1 ----
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)   # nhảy tới w + epsilon

            # ---- forward/backward lần 2 TẠI w+epsilon ----
            outputs = net(images)                  # forward lại với weight đã nhiễu
            loss2 = criterion(outputs, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)  # update w gốc bằng gradient ở w+epsilon

    Lưu ý: cần forward+backward 2 LẦN mỗi step -> chi phí compute tăng ~2x mỗi
    step, nhưng KHÔNG tăng số round/epoch cần thiết trong đa số trường hợp đã
    được report trong literature (thường vẫn hội tụ nhanh hơn hoặc tương đương
    về wall-clock nếu tính luôn việc cần ít epoch hơn để đạt cùng accuracy, và
    đặc biệt tăng đáng kể target-domain accuracy).
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0, f"rho phải >= 0, nhận được {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # nhảy tới w + epsilon
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # quay lại w gốc
        self.base_optimizer.step()  # update w gốc bằng gradient tính tại w+epsilon
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import torch


def _cross_domain_style_hook(module, input, output):
    """
    Forward hook gắn vào 1 layer cụ thể. Mỗi lần layer đó forward, hook này:

    1. LUÔN tích lũy mean/std của output (feature style thật của client này,
       domain này) vào module._style_accum - để cuối round gửi lên server xây
       style bank. Chỉ tích lũy khi net.training=True.

    2. Sau đó MIX output với style của 1 domain KHÁC lấy từ
       module._external_bank (do server broadcast round trước) - tạo ra style
       diversity thật sự, khác hẳn MixStyle-trong-batch (vốn vô dụng khi mỗi
       client chỉ có 1 domain).

    Nếu chưa có domain nào khác trong bank (ví dụ round đầu tiên), bước 2 bị bỏ
    qua, trả nguyên output - an toàn, không lỗi.
    """
    if not module.training:
        return output

    # ---- (1) tích lũy style thật của domain này ----
    mu = output.mean(dim=[0, 2, 3]).detach()
    sig = output.std(dim=[0, 2, 3]).detach()
    acc = module._style_accum
    if acc['count'] == 0:
        acc['sum_mean'] = mu.clone()
        acc['sum_std'] = sig.clone()
    else:
        acc['sum_mean'] += mu
        acc['sum_std'] += sig
    acc['count'] += 1

    # ---- (2) mix với style domain khác (nếu bank đã có) ----
    bank = getattr(module, '_external_bank', {})
    own_domain = getattr(module, '_own_domain_id', None)
    p = getattr(module, '_mixstyle_p', 0.5)
    alpha = getattr(module, '_mixstyle_alpha', 0.1)

    other_domains = [d for d in bank.keys() if d != own_domain]
    if len(other_domains) == 0 or torch.rand(1).item() > p:
        return output

    chosen = other_domains[torch.randint(len(other_domains), (1,)).item()]
    target_mean, target_std = bank[chosen]
    device = output.device
    target_mean = target_mean.to(device).view(1, -1, 1, 1)
    target_std = target_std.to(device).view(1, -1, 1, 1)

    inst_mu = output.mean(dim=[2, 3], keepdim=True)
    inst_sig = (output.var(dim=[2, 3], keepdim=True) + 1e-6).sqrt()
    x_norm = (output - inst_mu) / inst_sig

    B = output.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1, 1)).to(device)
    mu_mix = lam * inst_mu + (1 - lam) * target_mean
    sig_mix = lam * inst_sig + (1 - lam) * target_std

    return x_norm * sig_mix + mu_mix


def attach_cross_domain_style_hooks(net, layer_names, p=0.5, alpha=0.1):
    """
    Gắn hook 1 lần cho mỗi net (idempotent). Sau khi gọi hàm này, layer đó sẽ
    tự động vừa tích lũy style stats vừa mix cross-domain mỗi lần forward
    trong training - không cần sửa forward() của model.
    """
    if getattr(net, '_style_hook_attached', False):
        return

    named = dict(net.named_modules())
    for layer_name in layer_names:
        if layer_name not in named:
            raise ValueError(
                f"Không tìm thấy layer '{layer_name}' trong net. "
                f"Dùng `for n,_ in net.named_modules(): print(n)` để kiểm tra tên đúng."
            )
        layer = named[layer_name]
        layer._mixstyle_p = p
        layer._mixstyle_alpha = alpha
        layer._style_accum = {'sum_mean': None, 'sum_std': None, 'count': 0}
        layer._external_bank = {}
        layer._own_domain_id = None
        layer.register_forward_hook(_cross_domain_style_hook)

    net._style_hook_attached = True


def reset_style_accumulators(net, layer_names):
    """Gọi ở ĐẦU mỗi round training của 1 client, trước khi train batch đầu tiên."""
    named = dict(net.named_modules())
    for layer_name in layer_names:
        named[layer_name]._style_accum = {'sum_mean': None, 'sum_std': None, 'count': 0}


def collect_style_stats(net, layer_names):
    """
    Gọi ở CUỐI round training của 1 client. Trả về {layer_name: (mean, std)}
    để gửi lên server (giống hệt cách local_protos được gửi lên).
    """
    named = dict(net.named_modules())
    out = {}
    for layer_name in layer_names:
        acc = named[layer_name]._style_accum
        if acc['count'] > 0:
            out[layer_name] = (acc['sum_mean'] / acc['count'], acc['sum_std'] / acc['count'])
    return out


def set_style_context(net, layer_names, bank_per_layer, own_domain_id):
    """
    Gọi TRƯỚC khi bắt đầu train 1 client trong round mới - cung cấp cho hook
    biết: (a) style bank hiện tại của server (round trước), (b) domain_id của
    chính client này (để loại trừ khi random chọn domain khác).
    """
    named = dict(net.named_modules())
    for layer_name in layer_names:
        layer = named[layer_name]
        layer._external_bank = bank_per_layer.get(layer_name, {})
        layer._own_domain_id = own_domain_id

def _cross_domain_style_hook(module, input, output):
    """
    Forward hook gắn vào 1 layer cụ thể. Mỗi lần layer đó forward, hook này:

    1. LUÔN tích lũy mean/std của output (feature style thật của client này,
       domain này) vào module._style_accum - để cuối round gửi lên server xây
       style bank. Chỉ tích lũy khi net.training=True.

    2. Sau đó MIX output với style của 1 domain KHÁC lấy từ
       module._external_bank (do server broadcast round trước) - tạo ra style
       diversity thật sự, khác hẳn MixStyle-trong-batch (vốn vô dụng khi mỗi
       client chỉ có 1 domain).

    Nếu chưa có domain nào khác trong bank (ví dụ round đầu tiên), bước 2 bị bỏ
    qua, trả nguyên output - an toàn, không lỗi.
    """
    if not module.training:
        return output

    # ---- (1) tích lũy style thật của domain này ----
    mu = output.mean(dim=[0, 2, 3]).detach()
    sig = output.std(dim=[0, 2, 3]).detach()
    acc = module._style_accum
    if acc['count'] == 0:
        acc['sum_mean'] = mu.clone()
        acc['sum_std'] = sig.clone()
    else:
        acc['sum_mean'] += mu
        acc['sum_std'] += sig
    acc['count'] += 1

    # ---- (2) mix với style domain khác (nếu bank đã có) ----
    bank = getattr(module, '_external_bank', {})
    own_domain = getattr(module, '_own_domain_id', None)
    p = getattr(module, '_mixstyle_p', 0.5)
    alpha = getattr(module, '_mixstyle_alpha', 0.1)

    other_domains = [d for d in bank.keys() if d != own_domain]
    if len(other_domains) == 0 or torch.rand(1).item() > p:
        return output

    chosen = other_domains[torch.randint(len(other_domains), (1,)).item()]
    target_mean, target_std = bank[chosen]
    device = output.device
    target_mean = target_mean.to(device).view(1, -1, 1, 1)
    target_std = target_std.to(device).view(1, -1, 1, 1)

    inst_mu = output.mean(dim=[2, 3], keepdim=True)
    inst_sig = (output.var(dim=[2, 3], keepdim=True) + 1e-6).sqrt()
    x_norm = (output - inst_mu) / inst_sig

    B = output.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1, 1)).to(device)
    mu_mix = lam * inst_mu + (1 - lam) * target_mean
    sig_mix = lam * inst_sig + (1 - lam) * target_std

    return x_norm * sig_mix + mu_mix


def attach_cross_domain_style_hooks(net, layer_names, p=0.5, alpha=0.1):
    """
    Gắn hook 1 lần cho mỗi net (idempotent). Sau khi gọi hàm này, layer đó sẽ
    tự động vừa tích lũy style stats vừa mix cross-domain mỗi lần forward
    trong training - không cần sửa forward() của model.
    """
    if getattr(net, '_style_hook_attached', False):
        return

    named = dict(net.named_modules())
    for layer_name in layer_names:
        if layer_name not in named:
            raise ValueError(
                f"Không tìm thấy layer '{layer_name}' trong net. "
                f"Dùng `for n,_ in net.named_modules(): print(n)` để kiểm tra tên đúng."
            )
        layer = named[layer_name]
        layer._mixstyle_p = p
        layer._mixstyle_alpha = alpha
        layer._style_accum = {'sum_mean': None, 'sum_std': None, 'count': 0}
        layer._external_bank = {}
        layer._own_domain_id = None
        layer.register_forward_hook(_cross_domain_style_hook)

    net._style_hook_attached = True


def reset_style_accumulators(net, layer_names):
    """Gọi ở ĐẦU mỗi round training của 1 client, trước khi train batch đầu tiên."""
    named = dict(net.named_modules())
    for layer_name in layer_names:
        named[layer_name]._style_accum = {'sum_mean': None, 'sum_std': None, 'count': 0}


def collect_style_stats(net, layer_names):
    """
    Gọi ở CUỐI round training của 1 client. Trả về {layer_name: (mean, std)}
    để gửi lên server (giống hệt cách local_protos được gửi lên).
    """
    named = dict(net.named_modules())
    out = {}
    for layer_name in layer_names:
        acc = named[layer_name]._style_accum
        if acc['count'] > 0:
            out[layer_name] = (acc['sum_mean'] / acc['count'], acc['sum_std'] / acc['count'])
    return out


def set_style_context(net, layer_names, bank_per_layer, own_domain_id):
    """
    Gọi TRƯỚC khi bắt đầu train 1 client trong round mới - cung cấp cho hook
    biết: (a) style bank hiện tại của server (round trước), (b) domain_id của
    chính client này (để loại trừ khi random chọn domain khác).
    """
    named = dict(net.named_modules())
    for layer_name in layer_names:
        layer = named[layer_name]
        layer._external_bank = bank_per_layer.get(layer_name, {})
        layer._own_domain_id = own_domain_id
import torch
import torch.nn as nn


class LowRankDomainAdapter(nn.Module):
    """
    ΔW = A @ B^T, rank r << min(C_in, C_out). Áp dụng dạng cộng thêm (residual)
    vào output của 1 conv/linear layer đã chọn - THAY ĐỔI THỰC SỰ hàm tính toán
    của backbone theo domain, khác hẳn compositional-trên-prototype (chỉ đổi
    target của loss phụ, không đổi computation).
    """

    def __init__(self, channels, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(channels, rank))
        self.B = nn.Parameter(torch.randn(channels, rank) * 0.01)
        self.rank = rank

    def forward(self, x):
        # x: (B, C, H, W) - áp delta theo channel, broadcast qua H,W
        delta_w = self.A @ self.B.T          # (C, C) - ma trận điều chỉnh nhỏ
        # áp dụng như 1x1 conv: (B,C,H,W) -> permute để matmul theo channel
        B_, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)      # (B*H*W, C)
        delta = x_flat @ delta_w.T                          # (B*H*W, C)
        delta = delta.reshape(B_, H, W, C).permute(0, 3, 1, 2)
        return x + delta


class DomainAdapterBank(nn.Module):
    """
    Server-side: giữ 1 LowRankDomainAdapter RIÊNG cho mỗi domain đã biết,
    KHÔNG bao giờ trung bình adapter giữa các domain khác nhau (chỉ FedAvg
    trong nội bộ các client CÙNG domain, giống cách style bank đã làm).
    """

    def __init__(self, num_domains, channels, rank=8):
        super().__init__()
        self.adapters = nn.ModuleDict({
            str(d): LowRankDomainAdapter(channels, rank) for d in range(num_domains)
        })
        self.channels = channels
        self.rank = rank

    def get(self, domain_id):
        return self.adapters[str(domain_id)]

    def compose_for_unseen_domain(self, weights_per_domain):
        """
        Tổng hợp 1 adapter MỚI cho domain chưa từng thấy, bằng weighted average
        các adapter đã biết trong không gian THAM SỐ (A, B), không phải feature.

        weights_per_domain: dict {domain_id: weight}, tổng = 1. Trọng số này nên
        lấy từ độ TƯƠNG ĐỒNG giữa style thống kê của domain mới (đo được từ vài
        unlabeled sample nếu có) với style bank đã có (tái dùng chính
        cross_domain_style_bank.py đã xây trước đó - đây là điểm cộng hưởng hạ
        tầng, không cần xây thêm cơ chế đo tương đồng domain từ đầu).

        Nếu KHÔNG có bất kỳ sample nào từ domain mới (zero-shot hoàn toàn), dùng
        uniform weight qua mọi domain đã biết.
        """
        composed = LowRankDomainAdapter(self.channels, self.rank)
        composed = composed.to(next(self.parameters()).device)

        with torch.no_grad():
            composed.A.zero_()
            composed.B.zero_()
            for domain_id, w in weights_per_domain.items():
                adapter = self.adapters[str(domain_id)]
                # Compose trực tiếp trên ma trận delta_w = A@B^T (không compose
                # riêng A,B vì rank-space của từng domain không nhất thiết
                # "thẳng hàng" nhau - compose ở effective weight space an toàn hơn)
                composed_delta = w * (adapter.A @ adapter.B.T)
                if not hasattr(composed, '_accum_delta'):
                    composed._accum_delta = torch.zeros_like(composed_delta)
                composed._accum_delta += composed_delta

        # Gắn lại forward dùng thẳng _accum_delta thay vì A@B^T học lại
        composed.forward = lambda x, cd=composed._accum_delta: _apply_delta(x, cd)
        return composed


def _apply_delta(x, delta_w):
    B_, C, H, W = x.shape
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
    delta = x_flat @ delta_w.T
    delta = delta.reshape(B_, H, W, C).permute(0, 3, 1, 2)
    return x + delta


# ============================================================================
# Forward hook để chèn adapter vào layer có sẵn, không cần sửa kiến trúc net
# ============================================================================
def _adapter_hook(module, input, output):
    adapter = getattr(module, '_active_adapter', None)
    if adapter is None:
        return output
    return adapter(output)


def attach_adapter_hook(net, layer_name):
    named = dict(net.named_modules())
    layer = named[layer_name]
    # QUAN TRỌNG: dùng object.__setattr__ thay vì layer._active_adapter = None
    # trực tiếp. Nếu gán trực tiếp, nn.Module.__setattr__ sẽ tự động đăng ký
    # adapter (vì nó là nn.Module) làm SUBMODULE của layer -> lọt vào
    # layer.state_dict() -> gây lỗi "Unexpected key(s)" khi load_state_dict
    # giữa các net không cùng có adapter được set (ví dụ global_net chưa từng
    # gọi set_active_adapter). object.__setattr__ bỏ qua cơ chế auto-register
    # đó, chỉ gán attribute Python thuần túy - hook vẫn đọc được bình thường.
    object.__setattr__(layer, '_active_adapter', None)
    layer.register_forward_hook(_adapter_hook)


def set_active_adapter(net, layer_name, adapter_module):
    named = dict(net.named_modules())
    layer = named[layer_name]
    object.__setattr__(layer, '_active_adapter', adapter_module)


import torch


def agg_func_with_count(protos):
    """
    Thay cho agg_func gốc - trả về CẢ mean LẪN count (số sample đóng góp),
    cần thiết để tính trọng số reliability ở server.

    protos: {key: [feature1, feature2, ...]}
    return: {key: (mean_tensor, count)}
    """
    out = {}
    for key, feat_list in protos.items():
        stacked = torch.stack(feat_list)
        out[key] = (stacked.mean(dim=0), len(feat_list))
    return out


def aggregate_with_reliability_shrinkage(local_protos_with_count, online_clients,
                                          lam=50.0, momentum=0.0,
                                          prev_global_protos=None):
    """
    local_protos_with_count: {client_idx: {(class,domain): (mean, count)}}
    lam: hệ số shrinkage (λ) - N tương đương của "1 đơn vị tin cậy prior".
         λ lớn -> shrink mạnh hơn (cần nhiều sample hơn mới được tin tưởng
         centroid riêng của domain). Nên tune trên validation, gợi ý bắt đầu
         bằng median của N(class,domain) quan sát được qua vài round đầu.
    momentum: nếu > 0, áp dụng THÊM EMA qua các round (độc lập với shrinkage
              qua domain) - có thể để 0 nếu chỉ muốn shrinkage tại 1 round.
    prev_global_protos: bank round trước (dict {key: tensor}), cần nếu dùng momentum.

    Trả về: {(class,domain): tensor} - đã shrinkage, sẵn sàng dùng thẳng cho
    InfoNCE/DPA hiện có của bạn (KHÔNG CẦN SỬA GÌ Ở LOSS FUNCTION).
    """
    # ---- Bước 1: gộp (mean, count) theo (class, domain) từ mọi client online ----
    per_key = {}  # {(class,domain): [(mean, count), ...]}
    for idx in online_clients:
        stats = local_protos_with_count.get(idx, {})
        for key, (mean, count) in stats.items():
            per_key.setdefault(key, []).append((mean, count))

    # ---- Bước 2: weighted average theo N (thay attention thuần cosine) ----
    raw_proto = {}   # {(class,domain): mean}
    raw_count = {}   # {(class,domain): N}
    for key, stat_list in per_key.items():
        total_n = sum(c for _, c in stat_list)
        weighted_mean = sum(m * c for m, c in stat_list) / total_n
        raw_proto[key] = weighted_mean
        raw_count[key] = total_n

    # ---- Bước 3: tính class-level prior (gộp qua MỌI domain của class đó) ----
    class_groups = {}  # {class: [(mean, count), ...]}
    for (cls, dom), mean in raw_proto.items():
        class_groups.setdefault(cls, []).append((mean, raw_count[(cls, dom)]))

    class_prior = {}
    for cls, stat_list in class_groups.items():
        total_n = sum(c for _, c in stat_list)
        class_prior[cls] = sum(m * c for m, c in stat_list) / total_n

    # ---- Bước 4: Empirical-Bayes shrinkage ----
    shrunk_proto = {}
    for (cls, dom), mean in raw_proto.items():
        n = raw_count[(cls, dom)]
        w = n / (n + lam)
        prior = class_prior[cls]
        shrunk_proto[(cls, dom)] = w * mean + (1 - w) * prior

    # ---- Bước 5 (tùy chọn): EMA thêm qua các round nếu momentum > 0 ----
    if momentum > 0 and prev_global_protos is not None:
        for key, val in shrunk_proto.items():
            if key in prev_global_protos:
                shrunk_proto[key] = momentum * prev_global_protos[key] + (1 - momentum) * val

    return shrunk_proto, raw_count  # trả cả raw_count để log/debug độ tin cậy


def compute_lambda_from_data(raw_count_history, percentile=50):
    """
    Gợi ý cách chọn λ tự động thay vì đoán mò: lấy percentile (mặc định trung
    vị) của N(class,domain) quan sát được qua vài round đầu tiên (trước khi
    bật shrinkage). λ = N tại percentile đó -> nghĩa là combo nào có N THẤP
    HƠN trung vị sẽ bị shrink >50% về phía class prior, combo nào CAO HƠN
    trung vị sẽ giữ phần lớn giá trị riêng của mình.

    raw_count_history: list các dict {(class,domain): N} thu thập qua vài round.
    """
    import numpy as np
    all_counts = []
    for round_counts in raw_count_history:
        all_counts.extend(round_counts.values())
    return float(np.percentile(all_counts, percentile))