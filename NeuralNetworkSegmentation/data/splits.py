import numpy as np

from NeuralNetworkSegmentation.data.io import load_images

def ordered_split(
    group_pairs: dict[dict[np.ndarray]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
):
    assert abs(train_frac + val_frac + test_frac - 1) == 0

    # Generate random permutation for each positive vs negative tile
    p0 = np.random.permutation(len(group_pairs[0]))
    p1 = np.random.permutation(len(group_pairs[1]))

    # Extract from the dictionary of positive and negative tiles the name&input&mask by order of the permutations
    group_pairs0 = [group_pairs[0][i] for i in p0]
    group_pairs1 = [group_pairs[1][i] for i in p1]

    n_neg = len(group_pairs0)
    n_pos = len(group_pairs1)

    # ---- Split each class independently (FIXED) ----
    tr_neg = int(round(n_neg * train_frac))
    va_neg = int(round(n_neg * val_frac))
    te_neg = n_neg - tr_neg - va_neg

    tr_pos = int(round(n_pos * train_frac))
    va_pos = int(round(n_pos * val_frac))
    te_pos = n_pos - tr_pos - va_pos

    neg_ptr = 0
    pos_ptr = 0

    def take_split(n_neg_take, n_pos_take):
        nonlocal neg_ptr, pos_ptr

        group_pairs_neg = group_pairs0[neg_ptr:neg_ptr + n_neg_take]
        group_pairs_pos = group_pairs1[pos_ptr:pos_ptr + n_pos_take]

        neg_ptr += n_neg_take
        pos_ptr += n_pos_take

        group_pairs_neg.extend(group_pairs_pos)

        return group_pairs_neg

    splits = {
        "train": take_split(tr_neg, tr_pos),
        "val":   take_split(va_neg, va_pos),
        "test":  take_split(te_neg, te_pos),
    }

    return splits


if __name__ == "__main__":
    x_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/inputs"
    y_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/masks"

    group_pairs = load_images(x_dir, y_dir, [0, 3])

    splits = ordered_split(group_pairs, 0.6, 0.2, 0.2)

    print(len(splits["train"][0]), len(splits["train"][1]))
    print(len(splits["val"][0]), len(splits["val"][1]))
    print(len(splits["test"][0]), len(splits["test"][1]))


