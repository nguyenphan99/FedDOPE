"""
Vẽ biểu đồ line chart cho bảng ablation về ngưỡng độ tin cậy prototype (R_min)
và xuất ra file PNG và PDF.

Yêu cầu: pip install matplotlib --break-system-packages
"""

import matplotlib.pyplot as plt

# ----- Dữ liệu từ bảng -----
r_min = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data = {
    "DomainNet": [65.45, 65.44, 65.31, 65.48, 66.09, 65.39, 65.12, 65.82, 65.17],
    "Office-10": [74.43, 72.31, 74.19, 73.10, 75.43, 74.71, 73.78, 73.40, 73.36],
    "PACS":      [85.87, 86.70, 86.46, 86.16, 87.23, 86.34, 85.93, 86.30, 86.47],
}

# Giá trị tốt nhất (in đậm trong bảng gốc) ứng với R_min = 0.5
best_r_min = 0.5

# ----- Cấu hình style -----
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

markers = ["o", "s", "^"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

for (name, values), marker, color in zip(data.items(), markers, colors):
    ax.plot(
        r_min, values,
        marker=marker, markersize=7, linewidth=2,
        label=name, color=color,
    )

# Đánh dấu điểm tốt nhất bằng đường thẳng đứng mờ
ax.axvline(best_r_min, color="gray", linestyle="--", linewidth=1, alpha=0.5)

ax.set_xlabel(r"$R_{\min}$")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_xticks(r_min)
ax.legend(frameon=False, loc="lower center", ncol=3,
          bbox_to_anchor=(0.5, -0.32))
ax.grid(True, linestyle=":", alpha=0.4)

fig.tight_layout()

# ----- Xuất file -----
fig.savefig("/users/grad/nphan/work/FedDAP_CVPR2026/ablation_line_chart.png", bbox_inches="tight")
fig.savefig("/users/grad/nphan/work/FedDAP_CVPR2026/ablation_line_chart.pdf", bbox_inches="tight")

print("Đã xuất file PNG và PDF thành công.")