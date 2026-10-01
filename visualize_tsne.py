"""
visualize_tsne.py

Đọc file .npz (được lưu bởi extract_embeddings_and_prototypes trong training.py):
  - embeddings          (N, dim)
  - labels              (N,)
  - domains             (N,)               string, tên domain của từng ảnh
  - prototypes          (P, dim)           mean-embedding theo (class, domain)
  - prototype_labels    (P,)               class id của từng prototype
  - prototype_domains   (P,)               domain của từng prototype
  - prototype_counts    (P,)               (optional) số ảnh dùng để tính mean

Chạy t-SNE trên embedding + prototype CÙNG LÚC (fit chung 1 lần) để chúng nằm
trên cùng 1 không gian 2D, rồi vẽ:
  - Ảnh: chấm nhỏ, màu theo class, hình dạng marker theo domain.
  - Prototype: marker to hơn, viền đen, cùng bảng màu theo class, hình dạng
    marker theo domain (để phân biệt prototype của class X ở domain A vs B).

Lưu figure ra .pdf (vector, không bị vỡ nét khi phóng to trong paper).

Usage:
    python visualize_tsne.py \
        --npz_path ./output/embeddings/xxx_embeddings.npz \
        --output_pdf ./output/figures/tsne.pdf \
        --max_points 3000 \
        --perplexity 30
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # không cần display, chỉ xuất file
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DOMAIN_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>', 'p', '8']


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    domains = data['domains'].astype(str)

    prototypes = data['prototypes'] if 'prototypes' in data.files else None
    proto_labels = data['prototype_labels'] if 'prototype_labels' in data.files else None
    proto_domains = data['prototype_domains'].astype(str) if 'prototype_domains' in data.files else None

    return embeddings, labels, domains, prototypes, proto_labels, proto_domains


def subsample(embeddings, labels, domains, max_points, seed):
    n = embeddings.shape[0]
    if max_points is None or n <= max_points:
        return embeddings, labels, domains
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return embeddings[idx], labels[idx], domains[idx]


def build_domain_marker_map(domain_names):
    uniq = sorted(set(domain_names))
    return {d: DOMAIN_MARKERS[i % len(DOMAIN_MARKERS)] for i, d in enumerate(uniq)}


def build_class_color_map(class_ids):
    uniq = sorted(set(int(c) for c in class_ids))
    n = len(uniq)
    cmap_name = 'tab20' if n <= 20 else 'nipy_spectral'
    cmap = plt.get_cmap(cmap_name, max(n, 1))
    return {c: cmap(i) for i, c in enumerate(uniq)}, cmap, uniq


def run_tsne(x, perplexity, n_iter, seed):
    n_samples = x.shape[0]
    # perplexity phải < n_samples; tự động hạ xuống nếu dataset nhỏ / sau khi subsample
    safe_perplexity = max(5, min(perplexity, (n_samples - 1) // 3))
    kwargs = dict(n_components=2, perplexity=safe_perplexity,
                  random_state=seed, init='pca', learning_rate='auto')
    try:
        tsne = TSNE(n_iter=n_iter, **kwargs)          # sklearn cũ
    except TypeError:
        tsne = TSNE(max_iter=n_iter, **kwargs)        # sklearn >= 1.5 đổi tên n_iter -> max_iter
    return tsne.fit_transform(x)


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for embeddings + prototypes -> PDF")
    parser.add_argument('--npz_path', type=str, required=True, help="File .npz đã lưu (embeddings/prototypes)")
    parser.add_argument('--output_pdf', type=str, required=True, help="Đường dẫn file .pdf output")
    parser.add_argument('--perplexity', type=float, default=20.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_points', type=int, default=3000,
                         help="Subsample embeddings để t-SNE nhanh & đỡ rối mắt. Set 0 = dùng hết.")
    parser.add_argument('--point_size', type=float, default=100.0)
    parser.add_argument('--proto_size', type=float, default=300.0)
    parser.add_argument('--figsize', type=float, nargs=2, default=(14, 12))
    parser.add_argument('--dpi', type=int, default=900)
    args = parser.parse_args()

    embeddings, labels, domains, prototypes, proto_labels, proto_domains = load_npz(args.npz_path)

    max_points = None if args.max_points == 0 else args.max_points
    emb_sub, lab_sub, dom_sub = subsample(embeddings, labels, domains, max_points, args.seed)

    has_proto = prototypes is not None and prototypes.shape[0] > 0

    # --- Fit t-SNE chung trên embedding + prototype để cùng 1 không gian 2D ---
    if has_proto:
        combined = np.concatenate([emb_sub, prototypes], axis=0)
    else:
        combined = emb_sub

    proj = run_tsne(combined, args.perplexity, args.n_iter, args.seed)

    if has_proto:
        proj_emb = proj[:emb_sub.shape[0]]
        proj_proto = proj[emb_sub.shape[0]:]
    else:
        proj_emb = proj
        proj_proto = None

    # --- Bảng màu theo class, marker theo domain (dùng chung cho cả ảnh & prototype) ---
    all_labels_for_color = lab_sub if not has_proto else np.concatenate([lab_sub, proto_labels])
    class_to_color, cmap, uniq_classes = build_class_color_map(all_labels_for_color)

    all_domains_for_marker = dom_sub if not has_proto else np.concatenate([dom_sub, proto_domains])
    domain_marker_map = build_domain_marker_map(all_domains_for_marker)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Ảnh (chấm nhỏ, alpha thấp)
    for d in sorted(set(dom_sub.tolist())):
        mask = dom_sub == d
        colors = [class_to_color[int(c)] for c in lab_sub[mask]]
        ax.scatter(proj_emb[mask, 0], proj_emb[mask, 1],
                   c=colors, marker=domain_marker_map[d],
                   s=args.point_size, alpha=0.65, linewidths=0, zorder=2)

    # Prototype (marker to, viền đen, nổi bật)
    if has_proto:
        for d in sorted(set(proto_domains.tolist())):
            mask = proto_domains == d
            colors = [class_to_color[int(c)] for c in proto_labels[mask]]
            ax.scatter(proj_proto[mask, 0], proj_proto[mask, 1],
                       c=colors, marker=domain_marker_map[d],
                       s=args.proto_size, edgecolors='black', linewidths=1.6,
                       alpha=0.95, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(os.path.dirname(args.output_pdf) or '.', exist_ok=True)
    fig.savefig(args.output_pdf, format='png', bbox_inches='tight', dpi=args.dpi)
    plt.close(fig)
    print(f"[t-SNE] Saved figure -> {args.output_pdf}")
    print(f"[t-SNE] Points plotted: {emb_sub.shape[0]}"
          + (f", prototypes: {prototypes.shape[0]}" if has_proto else ", no prototypes found in npz"))


if __name__ == '__main__':
    main()