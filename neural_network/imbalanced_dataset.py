import torch

# binary classification
n_classes = 2

def get_weighted_random_sampler(samples):
    weights = _make_weights_for_balanced_classes(samples)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    return sampler

def _make_weights_for_balanced_classes(samples):
    n_samples = len(samples)
    count_per_class = [0] * n_classes
    for abc, sample_class in samples:
        count_per_class[int(sample_class)] += 1

    weight_per_class = [0.] * n_classes
    for i in range(n_classes):
        weight_per_class[i] = float(n_samples) / float(count_per_class[i])

    weights = [0] * n_samples
    for idx, (sample, sample_class) in enumerate(samples):
        weights[idx] = weight_per_class[int(sample_class)]

    return weights
