import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import os
from scipy.interpolate import make_interp_spline, BSpline

labelsize = 15
legendsize = 13

mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = legendsize

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

label_dict = {
    'contradict': 0,
    'entailment': 1,
    'neutral': 2
}
reverse_label_dict = {v: k for k, v in label_dict.items()}
reverse_feature_dict = {0:"no neg", 1:"neg 1", 2:"neg 2"}
opt_dir = "/Users/chuntinz/Documents/research/fairseq-gdro/process/figs"


def map_to_feature_id(group, label):
    group, label = int(group), int(label)
    if group == 2:
        return 0
    elif (group == 0 and label == 0) or (group == 1 and label in [1, 2]):
        return 1
    else:
        return 2


def map_gid_to_feature_label_id(group):
    if group == 0:
        return [(2, 1), (2, 2), (1, 0)]
    elif group == 1:
        return [(2, 0), (1, 1), (1, 2)]
    else:
        return [(0, 0), (0, 1), (0, 2)]


def read_outer_log(path):
    outer_weights = defaultdict(list)
    with open(path, "r") as fin:
        for line in fin:
            fields = line.strip().split("\t")
            inner_update = int(fields[0].split("=")[-1])
            weights = list(map(float, fields[-1].split("=")[-1].split()))
            outer_weights[inner_update].append(weights)
    return outer_weights


def read_inner_log(path):
    inner_weights = defaultdict(lambda : defaultdict(list))
    with open(path, "r") as fin:
        for line in fin:
            if line.startswith("Update"):
                update = int(line.strip().split("=")[-1])
            else:
                fields = line.strip().split("\t")
                weights = list(map(float, fields[3].split("=")[-1].split()))
                labels = list(map(int, fields[4].split("=")[-1].split()))
                group_id = int(fields[0].split("=")[-1])
                for w, l in zip(weights, labels):
                    inner_weights[update][(group_id, l)].append(w)
    return inner_weights


def read_log(path):
    outer_weights = defaultdict(lambda :defaultdict(list))
    epoch = 1
    with open(path, "r") as fin:
        for line in fin:
            if "train_inner | epoch" in line:
                epoch = int(line.split(" | ")[-1].split()[1].rstrip(":"))
            if " Group loss weights: tensor" in line:
                weights = line.strip().split("([")[-1].split("],")[0].split(", ")
                for idx, weight in enumerate(weights):
                    outer_weights[epoch][idx].append(float(weight.strip()))

    results = defaultdict(lambda :defaultdict(list))
    for eid in sorted(outer_weights.keys()):
        for gid in range(len(outer_weights[eid].keys())):
            fl_id = map_gid_to_feature_label_id(gid)
            for fid, lid in fl_id:
                results[fid][lid].append(np.mean(outer_weights[eid][gid]))

    return results


def get_smooth(x, y):
    new_x = np.linspace(x.min(), x.max(), 200)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(new_x)
    return new_x, y_smooth


def plot_beta_line(weights, weights_pure_greedy):
    # feature_id: label_id: [history_avg_weights:#num_updates]
    num_features = len(weights)
    fig, axes = plt.subplots(2, num_features)
    fig.set_size_inches(28, 8)
    legends = [reverse_label_dict[i] for i in range(3)]
    colors = ['darkblue', 'darkred', 'dimgrey']

    for fid in range(num_features):
        for lid in range(len(weights_pure_greedy[fid])):
            x = np.arange(len(weights_pure_greedy[fid][lid]))
            y = weights_pure_greedy[fid][lid]
            xx, yy = get_smooth(x, y)
            if lid == 0:
                axes[0][fid].plot(xx, yy, 'o-', markersize=1, color=colors[lid])
            else:
                axes[0][fid].plot(xx, yy, 'o-', markersize=1, color=colors[lid], alpha=0.6)
        # axes[0][fid].set_ylim(0, 2.5)
        axes[0][fid].grid(ls='-.', lw=0.2)
        if fid == 2:
            axes[0][fid].legend(legends, loc='best')
        if fid == 0:
            axes[0][fid].set(title="Attribute={}".format(reverse_feature_dict[fid]), ylabel="average weight (group DRO)")
        else:
            axes[0][fid].set(title="Attribute={}".format(reverse_feature_dict[fid]))

    for fid in range(num_features):
        for lid in range(len(weights[fid])):
            x = np.arange(len(weights[fid][lid]))
            y = weights[fid][lid]
            xx, yy = get_smooth(x, y)
            axes[1][fid].plot(xx, yy, 'o-', markersize=1, color=colors[lid])
        # axes[1][fid].set_ylim(0, 5)
        axes[1][fid].grid(ls='-.', lw=0.2)
        if fid == 2:
            axes[1][-1].legend(legends, loc='best')
        if fid == 0:
            axes[1][fid].set(xlabel="train epochs", ylabel="average weight (Ours)")
        else:
            axes[1][fid].set(xlabel="train epochs")

    plt.subplots_adjust(wspace=0.07, hspace=0.11)
    fig.savefig(os.path.join(opt_dir, "plot_hier_weights.pdf"), bbox_inches='tight')


def convert_dict_to_mat(weight):
    m = len(weight.keys())
    n = len(weight[0].keys())
    mat = np.zeros((m, n))
    for fid in range(m):
        for lid in range(n):
            mat[fid][lid] = np.mean(weight[fid][lid])
    return mat

def plot_beta_heatmap(weights, weights_pure_greedy):
    mat_h_greedy = convert_dict_to_mat(weights)
    mat_greedy = convert_dict_to_mat(weights_pure_greedy)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    features = [reverse_feature_dict[ii] for ii in range(len(reverse_feature_dict))]
    labels = [reverse_label_dict[ii] for ii in range(len(reverse_label_dict))]

    def heatmap(data, ax, title, cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """
        row_labels = features
        col_labels = labels
        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title(title)
        return im, cbar

    im, _ = heatmap(mat_greedy, axes[0], "weights (group DRO)", cmap='RdPu')
    im, _ = heatmap(mat_h_greedy, axes[1], "weights (GC-DRO)", cmap='RdPu')
    # plt.subplots_adjust(wspace=0.07, hspace=0.0)
    fig.savefig(os.path.join(opt_dir, "plot_hier_heatmap.pdf"), bbox_inches='tight')


def analyze_hierarchincal_weights(heatmap=False):
    root = "/Users/chuntinz/Documents/research/fairseq-gdro/saved_models/analysis_58_v9_res-1_ch1_fixc0_1e5_ema_0.5_bema_0.5_alpha_0.5_beta_0.2_instance_reweight_greedy_rand_17_mnli"
    outer_weights = read_outer_log(os.path.join(root, "outer_log.txt"))
    inner_weights = read_inner_log(os.path.join(root, "inter_log.txt"))

    results = defaultdict(lambda :defaultdict(list))  # feature_id: label_id: [history_avg_weights:#num_updates]
    for group_id, label in inner_weights[1].keys():
        fid = map_to_feature_id(group_id, label)
        group_outer_ws = np.mean([wlist[group_id] for wlist in outer_weights[0]])
        results[fid][label].append(group_outer_ws)

    print(len(inner_weights.keys()), len(outer_weights.keys()))
    print(sorted(inner_weights.keys()))
    print(sorted(outer_weights.keys()))
    for inner_update in inner_weights.keys():
        if inner_update == 35:
            continue
        outer_ws = outer_weights[inner_update]
        for group_id, label in inner_weights[inner_update].keys():
            group_outer_ws = [wlist[group_id] for wlist in outer_ws]  # m times logs of outer weights
            fid = map_to_feature_id(group_id, label)
            average_instance_weights = [iw * ow for iw in inner_weights[inner_update][(group_id, label)] for ow in group_outer_ws]
            average_instance_weights = np.mean(average_instance_weights)
            results[fid][label].append(average_instance_weights)

    greedy_model_root = "/Users/chuntinz/Documents/research/fairseq-gdro/saved_models/64_ema0.5_g3_256_0.2_rand_15213_gdro_greedy_mnli"
    greedy_weights = read_log(os.path.join(greedy_model_root, "log.txt"))

    if not heatmap:
        plot_beta_line(results, greedy_weights)
    else:
        plot_beta_heatmap(results, greedy_weights)


def plot_ablations():
    def annotate(ax, x, y, texts):
        for i, txt in enumerate(texts):
            ax.annotate(txt, (x[i], y[i]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    b = [0.1, 0.2, 0.3, 0.5, 0.7]
    fix_a_shift = [75.18, 75.32, 71.88, 71.28, 71.10]  # a = 0.5
    fix_a_clean = [54.92, 68.70, 75.14, 77.82, 76.58]  # a = 0.2
    fix_a_legends = [r"$\alpha$=0.2, clean partition", r"$\alpha$=0.5, imperfect partition"]

    a = [0.1, 0.2, 0.3, 0.5, 0.7]
    fix_b_shift = [74.26, 74.26, 73.94, 75.32, 74.76]  # b = 0.2
    fix_b_clean = [76.82, 77.82, 74.84, 76.50, 72.96]  # b = 0.5
    fix_b_legends = [r"$\beta$=0.5, clean partition", r"$\beta$=0.2, imperfect partition"]

    colors = ["darkblue", "darkred"]
    axes[0].plot(b, fix_a_clean, 'o-', markersize=1.5, color=colors[0])
    axes[0].plot(b, fix_a_shift, 'o-', markersize=1.5, color=colors[1])
    axes[0].grid(ls='-.', lw=0.2)
    axes[0].legend(fix_a_legends, loc='best')
    axes[0].set_ylim(top=80)
    bb_shift = [0.09, 0.17, 0.3, 0.5, 0.66]
    ffix_a_shift = [73.8, 75.9, 71.88, 71.78, 69.7]  # a = 0.5
    bb_clean = [0.11, 0.21, 0.29, 0.5, 0.66]
    ffix_a_clean = [54.92, 68.4, 75.54, 77.92, 75.2]  # a = 0.2
    annotate(axes[0], bb_shift, ffix_a_shift, list(map(str, fix_a_shift)))
    annotate(axes[0], bb_clean, ffix_a_clean, list(map(str, fix_a_clean)))
    axes[0].set_xticks(np.arange(0.1, 0.8, 0.1))
    axes[0].set(xlabel=r"$\beta$", ylabel="Robust Accuray")

    axes[1].plot(a, fix_b_clean, 'o-', markersize=1.5, color=colors[0])
    axes[1].plot(a, fix_b_shift, 'o-', markersize=1.5, color=colors[1])
    axes[1].grid(ls='-.', lw=0.2)
    axes[1].legend(fix_b_legends, loc='best')
    axes[1].set(xlabel=r"$\alpha$", ylabel="Robust Accuray")
    axes[1].set_ylim(top=79)
    aa_shift = [0.09, 0.2, 0.29, 0.48, 0.7]
    ffix_b_shift = [73.9, 74.26, 73.6, 74.9, 74.76]  # b = 0.2
    aa_clean = [0.1, 0.2, 0.3, 0.48, 0.7]
    ffix_b_clean = [76.52, 77.82, 75.2, 76.5, 72.96]  # b = 0.5
    annotate(axes[1], aa_clean, ffix_b_shift, list(map(str, fix_b_shift)))
    annotate(axes[1], aa_shift, ffix_b_clean, list(map(str, fix_b_clean)))
    axes[1].set_xticks(np.arange(0.1, 0.8, 0.1))
    # plt.subplots_adjust(wspace=0.07, hspace=0)
    fig.savefig(os.path.join(opt_dir, "ablation.pdf"), bbox_inches='tight')

## Plot the clean group weights along training epoches
# analyze_hierarchincal_weights()

## Plot heatmap
analyze_hierarchincal_weights(True)

## Plot ablations
#plot_ablations()
