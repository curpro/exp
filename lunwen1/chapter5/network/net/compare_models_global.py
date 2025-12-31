import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
# 请根据您实际文件的存放位置修改路径
FILES = {
    'MLP': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\2_MLP\nn_results_MLP.npz',
    'CNN-LSTM': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\3_CNN_LSTM\nn_results_CNN_LSTM.npz',
    'GRU': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\1_GRU\nn_results_GRU.npz',
    'CNN': r'D:\AFS\lunwen\lunwen1\chapter5\network\net\3_CNN\nn_results_CNN.npz'
}

DT = 1 / 30
UNIFIED_START_FRAME = 90


# ======================================

def load_and_compare_models():
    results = {}
    time_axis = None

    print(f"=== 开始读取模型对比数据 (统一跳过前 {UNIFIED_START_FRAME} 帧) ===")

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}")
            continue

        try:
            data = np.load(filename)
            err_pos = data['err_nn_pos']
            err_vel = data['err_nn_vel']
            t = data['t']

            if time_axis is None:
                time_axis = t

            safe_start = min(UNIFIED_START_FRAME, len(err_pos) - 1)

            rmse_pos = np.sqrt(np.mean(err_pos[safe_start:] ** 2))
            rmse_vel = np.sqrt(np.mean(err_vel[safe_start:] ** 2))
            var_pos = np.var(err_pos[safe_start:])

            results[label] = {
                'rmse_pos': rmse_pos,
                'rmse_vel': rmse_vel,
                'var_pos': var_pos,
                'err_pos_seq': err_pos,
                'err_vel_seq': err_vel
            }
            print(f"加载: {label:<10} | Pos RMSE: {rmse_pos:.4f}")

        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")

    if not results:
        print("错误：未加载到任何数据。")
        return

    # ================= 打印表格 =================
    print("\n" + "=" * 80)
    print(f"{'Model':<15} | {'Pos RMSE (m)':<15} | {'Vel RMSE (m/s)':<15} | {'Pos Var':<15}")
    print("-" * 80)

    # 排序逻辑：MLP > CNN-LSTM > GRU > CNN
    rank_order = ['MLP', 'GRU', 'CNN-LSTM', 'CNN']
    sorted_keys = [k for k in rank_order if k in results]

    vals_pos = []
    vals_vel = []
    labels = []

    for key in sorted_keys:
        r = results[key]
        print(f"{key:<15} | {r['rmse_pos']:<15.4f} | {r['rmse_vel']:<15.4f} | {r['var_pos']:<15.4f}")
        vals_pos.append(r['rmse_pos'])
        vals_vel.append(r['rmse_vel'])
        labels.append(key)
    print("=" * 80 + "\n")

    # === [配置] 颜色与样式 (反映性能排名) ===
    color_map = {
        'MLP': '#006400',  # 深绿 (Champion)
        'GRU': '#9467bd',  # 紫色 (Runner-up)
        'CNN-LSTM': '#ff7f0e',# 橙色 (Third)
        'CNN': '#d62728'  # 红色 (Baseline)
    }

    # [关键修改] 恢复线宽差异，让 MLP 最突出
    style_map = {
        'MLP': {'lw': 1.0, 'alpha': 0.80, 'zorder': 10},  # 最粗、最上层
        'GRU': {'lw': 1.5, 'alpha': 0.80, 'zorder': 8}, # 次粗
        'CNN-LSTM': {'lw': 1.0, 'alpha': 0.80, 'zorder': 5},  #中等
        'CNN': {'lw': 1.0, 'alpha': 0.80, 'zorder': 1}  # 最细、最淡背景
    }

    # ================= 绘图 1: 柱状图 =================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.6
    bar_colors = [color_map[label] for label in labels]

    # 左图：位置
    bars1 = ax1.bar(x, vals_pos, width, color=bar_colors, alpha=0.85)
    ax1.set_title(f'Position RMSE Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # 柱状图保留动态缩放 (Bar chart 不受尖峰影响)
    min_p, max_p = min(vals_pos), max(vals_pos)
    margin_p = (max_p - min_p) * 0.5 if max_p != min_p else 0.1
    ax1.set_ylim(max(0, min_p - margin_p), max_p + margin_p)

    # 右图：速度
    bars2 = ax2.bar(x, vals_vel, width, color=bar_colors, alpha=0.85)
    ax2.set_title(f'Velocity RMSE Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    min_v, max_v = min(vals_vel), max(vals_vel)
    margin_v = (max_v - min_v) * 0.5 if max_v != min_v else 0.1
    ax2.set_ylim(max(0, min_v - margin_v), max_v + margin_v)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(bars1, ax1)
    autolabel(bars2, ax2)
    fig1.tight_layout()

    # ================= 绘图 2: 原始误差曲线 (无缩放 / Full Range) =================
    plt.figure(figsize=(12, 10))

    # --- 上图：位置误差 ---
    ax_pos = plt.subplot(2, 1, 1)

    for key in sorted_keys:
        err = results[key]['err_pos_seq']
        rmse = results[key]['rmse_pos']

        plot_data = err[UNIFIED_START_FRAME:]

        # 获取样式
        style = style_map.get(key, {'lw': 1.0, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        ax_pos.plot(time_axis[UNIFIED_START_FRAME:], plot_data,
                    label=f'{key} (RMSE={rmse:.2f}m)',
                    color=line_color,
                    linewidth=style['lw'],
                    alpha=style['alpha'],
                    zorder=style['zorder'])

        # [严格执行] 不进行任何 percentile 缩放，展示所有尖峰
    ax_pos.set_title(f'Position Error Evolution')
    ax_pos.set_ylabel('Position Error (m)')
    ax_pos.legend(loc='upper right', framealpha=0.9, shadow=True)
    ax_pos.grid(True, linestyle='--', alpha=0.4)

    # --- 下图：速度误差 ---
    ax_vel = plt.subplot(2, 1, 2)

    for key in sorted_keys:
        err = results[key]['err_vel_seq']
        rmse = results[key]['rmse_vel']

        plot_data = err[UNIFIED_START_FRAME:]

        style = style_map.get(key, {'lw': 1.0, 'alpha': 0.7, 'zorder': 1})
        line_color = color_map.get(key, 'k')

        ax_vel.plot(time_axis[UNIFIED_START_FRAME:], plot_data,
                    label=f'{key} (RMSE={rmse:.2f}m/s)',
                    color=line_color,
                    linewidth=style['lw'],
                    alpha=style['alpha'],
                    zorder=style['zorder'])

    # [严格执行] 不进行任何 percentile 缩放
    ax_vel.set_title(f'Velocity Error Evolution')
    ax_vel.set_ylabel('Velocity Error (m/s)')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.legend(loc='upper right', framealpha=0.9, shadow=True)
    ax_vel.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_compare_models()