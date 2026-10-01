"""
prototype_evolution.py
=======================
Skeleton cho Federated Prototype Evolution (FPE) — Phase 1-3
(Reliability-weighted aggregation, Prototype dynamics, Adaptive birth/merge).

Thiết kế để CẮM VÀO codebase FedDOPE hiện có mà KHÔNG cần sửa DomainOperator,
DPA loss, CPCL loss, hay training loop trong `_train_net`. Điểm tích hợp duy
nhất là thay `proto_aggregation_attn(...)` bằng `EvolutionManager.step(...)`
trong `loc_update`.

Đã implement (Phase 1-3):
    - Prototype như living object (mu, n, reliability, velocity, variance, age, history)
    - Reliability = f(Agreement, Support, Temporal stability)   [Quality/Novelty: TODO]
    - Reliability-aware update:  mu_{t+1} = mu_t + lr * R * (new_mu - mu_t)
    - Matching local -> global prototype bằng cosine (đơn giản hoá vì
      class/domain đã biết từ nhãn; chỉ cần phân biệt multi-mode trong 1 (class,domain))
    - Adaptive birth với consensus (>= N client, >= T round) để tránh noise birth
    - Adaptive merge dựa trên similarity + reliability (không chỉ similarity)
    - Genealogy log (ai birth từ ai, ai merge với ai) để phục vụ paper/debug

Chưa implement (để riêng, theo đúng lộ trình 4-phase đã thống nhất):
    - Split (cần client-side k=2 clustering trên local samples, không chỉ
      covariance thô từ server — xem thảo luận trước khi implement)
    - Quality (Q) và Novelty (N) trong reliability formula (cần per-sample
      confidence và cross-prototype comparison, để Phase 2+)
    - Full semantic + temporal genealogy graph (hiện tại chỉ có genealogy_log
      dạng list sự kiện, đủ để dựng graph sau này nhưng chưa có visualization)
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


# ============================================================================
# 1. Prototype — living object thay vì tensor tĩnh
# ============================================================================
class Prototype:
    _id_counter = 0

    def __init__(self, class_id, domain_label, mu: torch.Tensor,
                 birth_round: int = 0, parents: Optional[List[int]] = None,
                 history_window: int = 5):
        Prototype._id_counter += 1
        self.id = Prototype._id_counter
        self.class_id = class_id
        self.domain_label = domain_label

        self.mu = mu.detach().clone()
        self.n = 0                       # số client đóng góp round gần nhất
        self.reliability = 0.5           # trung tính khi chưa có evidence
        self.velocity = 0.0              # angular velocity (float, radian)
        self.variance = 0.0
        self.age = 0
        self.history = deque(maxlen=history_window)
        self.history.append(self.mu.clone())

        self.birth_round = birth_round
        self.parents: List[int] = parents or []
        self.children: List[int] = []
        self.alive = True

    def key(self) -> Tuple:
        return (self.class_id, self.domain_label)

    @torch.no_grad()
    def _angular_distance(self, new_mu: torch.Tensor) -> float:
        # Embedding trong FedDOPE  thường sống gần hypersphere (cosine-based loss)
        # -> dùng angular distance thay vì ||mu_t - mu_{t-1}||_2 cho velocity/temporal.
        a = F.normalize(self.mu.unsqueeze(0), dim=1)
        b = F.normalize(new_mu.detach().unsqueeze(0), dim=1)
        cos = (a @ b.T).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.arccos(cos).item()

    @torch.no_grad()
    def apply_update(self, new_mu: torch.Tensor, n_contrib: int,
                      variance: float, reliability: float, round_idx: int,
                      lr: float = 1.0):
        """
        Reliability-aware update (mục 16 trong thiết kế gốc):
            mu_{t+1} = mu_t + lr * R * (new_mu - mu_t)
        R thấp  -> gần như không update (bảo toàn, tránh noise).
        R cao   -> update mạnh (tin tưởng evidence mới).
        """
        self.velocity = self._angular_distance(new_mu)
        delta = new_mu.detach() - self.mu
        self.mu = self.mu + lr * reliability * delta
        self.n = n_contrib
        self.variance = variance
        self.reliability = reliability
        self.age += 1
        self.history.append(self.mu.clone())

    def temporal_stability(self) -> Optional[float]:
        """T_i = exp(-angular_dist(mu_t, mu_{t-1})). None nếu chưa đủ history."""
        if len(self.history) < 2:
            return None
        prev, cur = self.history[-2], self.history[-1]
        a = F.normalize(prev.unsqueeze(0), dim=1)
        b = F.normalize(cur.unsqueeze(0), dim=1)
        cos = (a @ b.T).clamp(-1 + 1e-6, 1 - 1e-6)
        dist = torch.arccos(cos).item()
        return math.exp(-dist)

    def to_dict(self) -> dict:
        return dict(id=self.id, class_id=self.class_id, domain=self.domain_label,
                    n=self.n, reliability=round(self.reliability, 3),
                    velocity=round(self.velocity, 4), variance=round(self.variance, 4),
                    age=self.age, birth_round=self.birth_round,
                    parents=self.parents, children=self.children, alive=self.alive)


# ============================================================================
# 2. Reliability Estimator
# ============================================================================
class ReliabilityEstimator:
    """
    R_i = w_A * Agreement + w_S * Support + w_T * Temporal
    (Quality và Novelty để trống — cần per-sample confidence / cross-proto
    comparison, thêm ở Phase 2+ khi đã có evidence rằng 3 thành phần này đủ.)

    Dùng additive (không multiplicative) để tránh collapse về 0 khi 1 thành
    phần thấp do prototype còn non tuổi (xem thảo luận trước khi implement).
    """

    def __init__(self, w_agreement: float = 0.4, w_support: float = 0.2,
                 w_temporal: float = 0.4, support_full_at: int = 5,
                 temporal_default: float = 0.5):
        assert abs(w_agreement + w_support + w_temporal - 1.0) < 1e-6, \
            "weights should sum to 1"
        self.w_a = w_agreement
        self.w_s = w_support
        self.w_t = w_temporal
        self.support_full_at = support_full_at
        self.temporal_default = temporal_default  # default cho prototype mới (age<2)

    def agreement(self, local_vectors: List[torch.Tensor], ref_mu: torch.Tensor) -> float:
        ref = F.normalize(ref_mu.unsqueeze(0), dim=1)
        vs = F.normalize(torch.stack(local_vectors), dim=1)
        sims = (vs @ ref.T).squeeze(1)
        return sims.mean().item()

    def support(self, n_contrib: int, total_clients: int) -> float:
        denom = max(1, min(self.support_full_at, total_clients))
        return min(1.0, n_contrib / denom)

    def temporal(self, prototype: Prototype) -> float:
        t = prototype.temporal_stability()
        return self.temporal_default if t is None else t

    def estimate(self, prototype: Prototype, local_vectors: List[torch.Tensor],
                 n_contrib: int, total_clients: int) -> Tuple[float, dict]:
        ref_mu = prototype.mu if prototype.age > 0 else torch.stack(local_vectors).mean(0)
        A = self.agreement(local_vectors, ref_mu)
        S = self.support(n_contrib, total_clients)
        T = self.temporal(prototype)
        R = self.w_a * A + self.w_s * S + self.w_t * T
        R = float(max(0.0, min(1.0, R)))
        return R, dict(agreement=A, support=S, temporal=T)


@dataclass
class _BirthCandidate:
    mu: torch.Tensor
    client_ids: set
    rounds_seen: int
    last_round: int


# ============================================================================
# 3. Evolution Manager — server-side orchestrator
# ============================================================================
class EvolutionManager:
    """
    Thay thế `proto_aggregation_attn` trong feddope.py. Nhận local_protos của
    mọi client trong 1 round, trả về global_protos tương thích ngược với
    pipeline hiện có (dict {(class_id, domain_label): tensor}).

    Vì class/domain đã biết từ nhãn (supervised FL), "matching" ở đây chỉ cần
    phân biệt multi-mode TRONG CÙNG 1 (class, domain) key — không cần cost
    matrix 4 thành phần (semantic/domain/temporal/uncertainty) như bản thiết
    kế tổng quát ban đầu. Đơn giản hoá này an toàn vì domain/class label loại
    bỏ phần lớn ambiguity mà cost matrix đó nhắm tới.
    """

    def __init__(self,
                 match_similarity_threshold: float = 0.7,
                 birth_candidate_sim_threshold: float = 0.9,
                 birth_min_clients: int = 2,
                 birth_min_rounds: int = 2, 
                 merge_sim_threshold: float = 0.97,
                 merge_min_reliability: float = 0.5,
                 max_prototypes_per_key: int = 3,
                 update_lr: float = 1.0,
                 reliability_estimator: Optional[ReliabilityEstimator] = None):
        self.match_similarity_threshold = match_similarity_threshold
        self.birth_candidate_sim_threshold = birth_candidate_sim_threshold
        self.birth_min_clients = birth_min_clients
        self.birth_min_rounds = birth_min_rounds
        self.merge_sim_threshold = merge_sim_threshold
        self.merge_min_reliability = merge_min_reliability
        self.max_prototypes_per_key = max_prototypes_per_key
        self.update_lr = update_lr
        self.reliability = reliability_estimator or ReliabilityEstimator()

        # key -> List[Prototype]  (hỗ trợ multi-prototype/mode mỗi class-domain
        # một khi birth kích hoạt; Phase 1 thường chỉ có 1 prototype/key)
        self.registry: Dict[Tuple, List[Prototype]] = defaultdict(list)
        self.pending_births: Dict[Tuple, List[_BirthCandidate]] = defaultdict(list)
        self.genealogy_log: List[dict] = []
        self.round_idx = 0

    # ------------------------------------------------------------------
    # Entry point chính, gọi thay cho proto_aggregation_attn(...)
    # ------------------------------------------------------------------
    def step(self, local_protos_by_client: Dict[int, Dict[Tuple, torch.Tensor]]
              ) -> Dict[Tuple, torch.Tensor]:
        """
        local_protos_by_client: {client_id: {(class_id, domain_label): proto_tensor}}
        (đúng format hiện có của `self.local_protos` trong feddope.py)

        Return: {(class_id, domain_label): tensor} — dominant prototype mỗi
        key (reliability cao nhất), tương thích ngược 100% với
        build_domain_proto_grid / _refresh_proto_cache hiện tại.
        """
        self.round_idx += 1
        total_clients = len(local_protos_by_client)

        grouped: Dict[Tuple, List[Tuple[int, torch.Tensor]]] = defaultdict(list)
        for client_id, protos in local_protos_by_client.items():
            for key, vec in protos.items():
                grouped[key].append((client_id, vec.detach()))

        for key, contributions in grouped.items():
            self._match_and_update(key, contributions, total_clients)

        self._process_births(total_clients)
        self._process_merges()

        return self._export_dominant()

    # ------------------------------------------------------------------
    # Matching + reliability-aware update cho prototype đã tồn tại
    # ------------------------------------------------------------------
    def _match_and_update(self, key, contributions, total_clients):
        existing = [p for p in self.registry[key] if p.alive]

        if not existing:
            self._register_birth_candidates(key, contributions)
            return

        proto_mus = F.normalize(torch.stack([p.mu for p in existing]), dim=1)
        assigned: Dict[int, List[torch.Tensor]] = defaultdict(list)
        unmatched = []

        for client_id, vec in contributions:
            v_n = F.normalize(vec.unsqueeze(0), dim=1)
            sims = (v_n @ proto_mus.T).squeeze(0)
            best_sim, best_idx = sims.max(dim=0)
            if best_sim.item() >= self.match_similarity_threshold:
                assigned[best_idx.item()].append(vec)
            else:
                unmatched.append((client_id, vec))

        for idx, vecs in assigned.items():
            proto = existing[idx]
            stacked = torch.stack(vecs)
            new_mu = stacked.mean(dim=0)
            variance = stacked.var(dim=0, unbiased=False).mean().item() if len(vecs) > 1 else 0.0
            R, _components = self.reliability.estimate(proto, vecs, len(vecs), total_clients)
            proto.apply_update(new_mu, len(vecs), variance, R, self.round_idx, lr=self.update_lr)

        if unmatched:
            self._register_birth_candidates(key, unmatched)

    # ------------------------------------------------------------------
    # Adaptive birth: tránh 1 client tự tạo prototype -> cần consensus
    # (>= birth_min_clients client, >= birth_min_rounds round persistence)
    # ------------------------------------------------------------------
    def _register_birth_candidates(self, key, contributions):
        candidates = self.pending_births[key]
        for client_id, vec in contributions:
            v_n = F.normalize(vec.unsqueeze(0), dim=1)
            matched = None
            for cand in candidates:
                sim = F.cosine_similarity(v_n, F.normalize(cand.mu.unsqueeze(0), dim=1)).item()
                if sim >= self.birth_candidate_sim_threshold:
                    matched = cand
                    break
            if matched is not None:
                matched.mu = 0.5 * matched.mu + 0.5 * vec
                matched.client_ids.add(client_id)
                if matched.last_round != self.round_idx:
                    matched.rounds_seen += 1
                    matched.last_round = self.round_idx
            else:
                candidates.append(_BirthCandidate(
                    mu=vec.clone(), client_ids={client_id},
                    rounds_seen=1, last_round=self.round_idx))

    def _process_births(self, total_clients):
        for key, candidates in list(self.pending_births.items()):
            survivors = []
            for cand in candidates:
                enough_consensus = (len(cand.client_ids) >= self.birth_min_clients and
                                     cand.rounds_seen >= self.birth_min_rounds)
                if enough_consensus:
                    if len([p for p in self.registry[key] if p.alive]) >= self.max_prototypes_per_key:
                        continue  # đủ capacity cho key này rồi, bỏ candidate
                    new_proto = Prototype(class_id=key[0], domain_label=key[1],
                                           mu=cand.mu, birth_round=self.round_idx)
                    new_proto.n = len(cand.client_ids)
                    self.registry[key].append(new_proto)
                    self._log_event('birth', new_proto.id, key=key,
                                     n_clients=len(cand.client_ids),
                                     rounds_seen=cand.rounds_seen)
                    continue  # promoted -> loại khỏi pending
                if cand.last_round == self.round_idx:
                    survivors.append(cand)
                # candidate không xuất hiện round này -> để nó rơi rụng tự nhiên
            self.pending_births[key] = survivors

    # ------------------------------------------------------------------
    # Adaptive merge: similarity cao KHÔNG đủ, cần cả hai bên reliable
    # ------------------------------------------------------------------
    def _process_merges(self):
        for key in list(self.registry.keys()):
            protos = [p for p in self.registry[key] if p.alive]
            if len(protos) < 2:
                continue
            merged_flags = [False] * len(protos)
            result = []
            for i in range(len(protos)):
                if merged_flags[i]:
                    continue
                target = protos[i]
                for j in range(i + 1, len(protos)):
                    if merged_flags[j]:
                        continue
                    other = protos[j]
                    sim = F.cosine_similarity(target.mu.unsqueeze(0), other.mu.unsqueeze(0)).item()
                    if (sim >= self.merge_sim_threshold and
                            target.reliability >= self.merge_min_reliability and
                            other.reliability >= self.merge_min_reliability):
                        target = self._merge_pair(target, other)
                        merged_flags[j] = True
                result.append(target)
            self.registry[key] = result

    def _merge_pair(self, p_a: Prototype, p_b: Prototype) -> Prototype:
        r_sum = p_a.reliability + p_b.reliability + 1e-8
        new_mu = (p_a.reliability * p_a.mu + p_b.reliability * p_b.mu) / r_sum
        merged = Prototype(class_id=p_a.class_id, domain_label=p_a.domain_label,
                            mu=new_mu, birth_round=self.round_idx,
                            parents=[p_a.id, p_b.id])
        merged.reliability = max(p_a.reliability, p_b.reliability)
        merged.n = p_a.n + p_b.n
        p_a.children.append(merged.id)
        p_b.children.append(merged.id)
        p_a.alive = False
        p_b.alive = False
        self._log_event('merge', merged.id, parents=[p_a.id, p_b.id])
        return merged

    # ------------------------------------------------------------------
    # TODO Phase 4 — Split (client-side k=2 clustering, KHÔNG dùng covariance
    # thô từ server vì không phát hiện đúng bimodality — xem thảo luận).
    # def _process_splits(self, per_client_sample_stats): ...
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, proto_id: int, **kwargs):
        self.genealogy_log.append(dict(round=self.round_idx, event=event_type,
                                        proto_id=proto_id, **kwargs))

    # ------------------------------------------------------------------
    # Export cho pipeline FedDOPE hiện tại
    # ------------------------------------------------------------------
    def _export_dominant(self) -> Dict[Tuple, torch.Tensor]:
        """
        DPA/CPCL/build_domain_proto_grid hiện giả định 1 prototype mỗi
        (class, domain). Chọn prototype reliability cao nhất làm "dominant"
        để KHÔNG phá vỡ pipeline hiện có. Dùng export_all() nếu sau này sửa
        DPA/CPCL để lặp qua nhiều prototype mỗi key.
        """
        out = {}
        for key, protos in self.registry.items():
            alive = [p for p in protos if p.alive]
            if not alive:
                continue
            best = max(alive, key=lambda p: p.reliability)
            out[key] = best.mu
        return out

    def export_all(self) -> Dict[Tuple, List[torch.Tensor]]:
        return {key: [p.mu for p in protos if p.alive]
                for key, protos in self.registry.items()}

    def snapshot(self) -> Dict[Tuple, List[dict]]:
        """Dùng để log/debug/paper figure: state đầy đủ mọi prototype sống."""
        return {key: [p.to_dict() for p in protos if p.alive]
                for key, protos in self.registry.items()}


# ============================================================================
# 4. Ghi chú tích hợp vào feddope.py (KHÔNG cần sửa DomainOperator/DPA/CPCL)
# ============================================================================
"""
Trong feddope.__init__:

    from utils.prototype_evolution import EvolutionManager
    self.evolution_manager = EvolutionManager(
        match_similarity_threshold=0.7,
        birth_min_clients=2,
        birth_min_rounds=2,
        merge_sim_threshold=0.97,
    )

Trong feddope.loc_update, thay dòng:

    self.global_protos = self.proto_aggregation_attn(self.local_protos, temperature=self.args.T)

bằng:

    self.global_protos = self.evolution_manager.step(self.local_protos)

_refresh_proto_cache() và build_domain_proto_grid() KHÔNG cần sửa gì, vì
global_protos vẫn có dạng {(class_id, domain_label): tensor} như trước
(export_dominant() đảm bảo tương thích ngược).

Muốn log genealogy mỗi round (phục vụ paper figure), thêm cuối loc_update:

    if epoch % 10 == 0:
        print(self.evolution_manager.genealogy_log[-5:])
"""