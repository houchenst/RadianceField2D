import torch
import numpy as np
import os, sys
import glob
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(exp_name):
    all_checkpoints = glob.glob(os.path.join(".", "experiments", exp_name, "checkpoints", "*"))
    if len(all_checkpoints) == 0:
        print(f"Could not find any checkpoints for experiment '{exp_name}'")
        exit(1)
    last_checkpoint = all_checkpoints[-1]
    model_weights = torch.load(last_checkpoint, map_location=device)["model_state_dict"]
    return model_weights

def compare_weights(exp1, exp2, max_diff_manual=None):
    '''
    Compares the learned weights between models from 2 different experiments
    '''
    # exp1 = "hand_frame1"
    # exp2 = "hand_frame2"

    model1_weights = load_weights(exp1)
    model2_weights = load_weights(exp2)

    print(f"Comparing weights between '{exp1}' and '{exp2}'")

    weights = [x for x in model1_weights.keys() if "weight" in x]
    biases = [x for x in model1_weights.keys() if "bias" in x]
    weights.sort()
    biases.sort()
    weightsandbiases = weights + biases

    # show average difference between weights in each layer
    min_diff = 0.
    max_diff = 0.
    print("Average difference between weights")
    for w in weightsandbiases:
        assert(w in model2_weights.keys())
        m1w = model1_weights[w].cpu().numpy()
        m2w = model2_weights[w].cpu().numpy()
        diff = np.abs(m1w - m2w)
        max_diff = max(max_diff, np.max(diff))
        print(max_diff)
        avg_diff = np.mean(diff)
        print(f"{w}\t:   {avg_diff:4f}")

    # show average difference between normalized weights in each layer
    min_diff = 0.
    max_diff = 0.
    print("Average and StDev of difference between normalized weights")
    for w in weightsandbiases:
        assert(w in model2_weights.keys())
        m1w = model1_weights[w].cpu().numpy()
        m2w = model2_weights[w].cpu().numpy()
        all_weights = np.hstack([m1w.flatten(), m2w.flatten()])
        weight_mean = np.mean(all_weights)
        weight_stdev = np.std(all_weights)
        m1w = (m1w-weight_mean)/weight_stdev
        m2w = (m2w-weight_mean)/weight_stdev
        diff = m1w - m2w
        max_diff = max(max_diff, np.max(diff))
        avg_diff = np.mean(np.abs(diff))
        stdev_diff = np.std(diff)
        print(f"{w}\t:   {avg_diff:4f}\t: {stdev_diff:4f}")

    if max_diff_manual is not None:
        max_diff = max_diff_manual
    

    print(f"\n\n\nAverage weight magnitudes for '{exp1}' and '{exp2}'")
    # show average weight magnitude in each layer
    for w in weightsandbiases:
        assert(w in model2_weights.keys())
        m1w = model1_weights[w].cpu().numpy()
        m2w = model2_weights[w].cpu().numpy()
        avg_mag = np.mean(np.abs(m1w)) + np.mean(np.abs(m2w))
        avg_mag /= 2.
        print(f"{w}\t:   {avg_mag:4f}")



    fig, axs = plt.subplots(1, len(weightsandbiases))
    fig.set_size_inches(len(weightsandbiases)*2.0, 2.5)

    for ax, w in zip(axs, weightsandbiases[:len(weightsandbiases)]):
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        m1w = model1_weights[w].cpu().numpy()
        m2w = model2_weights[w].cpu().numpy()
        all_weights = np.hstack([m1w.flatten(), m2w.flatten()])
        weight_mean = np.mean(all_weights)
        weight_stdev = np.std(all_weights)
        m1w = (m1w-weight_mean)/weight_stdev
        m2w = (m2w-weight_mean)/weight_stdev
        diff = np.abs(m1w - m2w)
        aspect = 1.
        if len(diff.shape) == 1:
            diff = diff[:,None]
            aspect = 1./(diff.shape[0]/4.)
        elif diff.shape[0] < 10:
            aspect = diff.shape[1]/10.
        ax.imshow(diff, aspect=aspect, vmin=min_diff, vmax=max_diff)
        ax.set_xlabel(diff.shape)
        ax.set_title(w.split(".")[2] + " " + w.split(".")[1])
    plt.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_diff, vmax=max_diff),cmap='viridis'), ax=axs[-1])
    plt.show()

# TODO:
# 1) Save comparison to file
# 2) Add SSIM metric
# 2) Show diff and standard deviation (somehow normalize weights?)
# 3) Try using layernorm between layers


if __name__ == "__main__":
    assert(len(sys.argv) == 3 or len(sys.argv) == 4)
    max_diff_manual = None
    if len(sys.argv) == 4:
        max_diff_manual = float(sys.argv[3])
    compare_weights(sys.argv[1], sys.argv[2], max_diff_manual=max_diff_manual)
    # compare_weights(sys.argv[1], sys.argv[2])