from collections import Counter

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler

def resample(X_train, y_train):
    # Apply over- and undersampling
    print("Destribution before...")
    print(Counter(y_train))

    oversample = SMOTE(sampling_strategy=0.5)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    print("Destribution after oversampling...")
    print(Counter(y_train))

    undersample = RandomUnderSampler()
    X_train, y_train = undersample.fit_resample(X_train, y_train)

    # - [ ] Print class distributions/ Counter(y) to check if resampling worked
    print("Destribution after undersampling...")
    print(Counter(y_train))

    return X_train, y_train
