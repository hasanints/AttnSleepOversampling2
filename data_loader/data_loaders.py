import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.combine import SMOTEENN


class LoadDataset_from_numpy(Dataset):
    def __init__(self, X_data, y_data):
        super(LoadDataset_from_numpy, self).__init__()
        self.x_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()

        # Reshape to (Batch_size, #channels, seq_len)
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from collections import Counter

from imblearn.over_sampling import ADASYN
from collections import Counter

# def apply_adasyn_class_1(X_train, y_train):
#     # Melihat distribusi kelas sebelum ADASYN
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum ADASYN: {class_counts}")
    
#     # Hanya melakukan oversampling untuk kelas 1
#     sampling_strategy = {1: class_counts[1] * 2}  # Oversample kelas 1 (misalnya 2x jumlah saat ini)
    
#     # Inisialisasi ADASYN dengan sampling strategy hanya untuk kelas 1
#     adasyn = ADASYN(random_state=42, sampling_strategy=sampling_strategy)
    
#     # Ubah data menjadi 2D untuk kompatibilitas dengan ADASYN
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    
#     # Terapkan ADASYN untuk oversampling hanya pada kelas 1
#     X_resampled, y_resampled = adasyn.fit_resample(X_train_reshaped, y_train)
    
#     # Kembalikan data ke bentuk 3D seperti aslinya
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
#     # Melihat distribusi kelas setelah ADASYN
#     print(f"Distribusi kelas setelah ADASYN: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled



# def apply_smote_2013_with_shuffle(X_train, y_train, classes = [0, 1, 2, 3, 4]):
#     """
#     Menerapkan SMOTE untuk data dengan label numerik (0-4).
#     """

#     # Melihat distribusi kelas sebelum SMOTE
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum SMOTE: {class_counts}")
    
#     # Ubah data menjadi 2D untuk kompatibilitas dengan SMOTE
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#     y_train_flat = y_train.flatten()

#     # Hitung jumlah sampel per kelas
#     nums = [len(np.where(y_train_flat == cl)[0]) for cl in classes]

#     # Tentukan rasio SMOTE
#     n_osamples = nums[2] - 7000  # Target oversampling berdasarkan kelas N2
#     ratio = {
#         0: n_osamples if nums[0] < n_osamples else nums[0],
#         1: n_osamples if nums[1] < n_osamples else nums[1],
#         2: nums[2],
#         3: n_osamples if nums[3] < n_osamples else nums[3],
#         4: n_osamples if nums[4] < n_osamples else nums[4]
#     }

#     print(f"Rasio oversampling untuk SMOTE: {ratio}")

#     # Inisialisasi SMOTE dengan rasio
#     smote = SMOTE(random_state=42, sampling_strategy=ratio)

#     # Terapkan SMOTE untuk oversampling kelas minoritas
#     X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train_flat)

#     # Kembalikan data ke bentuk 3D seperti aslinya
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

#     # Shuffle data
#     X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)
    
#     # Melihat distribusi kelas setelah SMOTE
#     print(f"Distribusi kelas setelah SMOTE: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled


from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE

def apply_smote_2018_with_shuffle(X_train, y_train, n_oversampling=35000):
    """
    Terapkan SMOTE untuk data versi 2018 dengan pengaturan oversampling tertentu,
    penghapusan data dari kelas mayoritas, dan shuffle data setelahnya.
    
    Args:
    X_train : numpy.ndarray
        Data fitur pelatihan dengan dimensi 3D (samples, time, features).
    y_train : numpy.ndarray
        Label pelatihan dalam format 1D (dengan angka 0-4 untuk kelas).
    n_oversampling : int, optional
        Target jumlah sampel setelah oversampling (default: 35000).
        
    Returns:
    X_resampled, y_resampled : numpy.ndarray
        Data fitur dan label setelah diterapkan SMOTE dan shuffle.
    """
    # Reshape data ke 2D
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.flatten()

    # Distribusi kelas sebelum SMOTE
    print(f"Distribusi kelas sebelum SMOTE: {Counter(y_train)}")

    # Undersampling untuk kelas 'W' (0 - Wake)
    under_sample_len = 35000
    W_indices = np.where(y_train == 0)[0]  # 0 mewakili kelas 'W'
    if len(W_indices) > under_sample_len:
        np.random.seed(42)  # Untuk konsistensi
        permute = np.random.permutation(W_indices)
        W_indices_to_remove = permute[:len(W_indices) - under_sample_len]
        y_train = np.delete(y_train, W_indices_to_remove, axis=0)
        X_train_reshaped = np.delete(X_train_reshaped, W_indices_to_remove, axis=0)

    # Undersampling untuk kelas 'N2' (2 - Light Sleep N2)
    N2_indices = np.where(y_train == 2)[0]  # 2 mewakili kelas 'N2'
    if len(N2_indices) > under_sample_len:
        np.random.seed(42)
        permute = np.random.permutation(N2_indices)
        N2_indices_to_remove = permute[:len(N2_indices) - under_sample_len]
        y_train = np.delete(y_train, N2_indices_to_remove, axis=0)
        X_train_reshaped = np.delete(X_train_reshaped, N2_indices_to_remove, axis=0)

    # Distribusi setelah undersampling
    print(f"Distribusi kelas setelah undersampling: {Counter(y_train)}")

    # Tentukan rasio SMOTE untuk setiap kelas
    nums = [len(np.where(y_train == label)[0]) for label in range(len(Counter(y_train)))]
    ratio = {
        label: n_oversampling if nums[label] < n_oversampling else nums[label]
        for label in range(len(nums))
    }

    # Terapkan SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=ratio)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)

    # Distribusi setelah SMOTE
    print(f"Distribusi kelas setelah SMOTE: {Counter(y_resampled)}")

    # Reshape kembali ke 3D
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

    # Shuffle data
    permute = np.random.permutation(len(y_resampled))
    X_resampled = X_resampled[permute]
    y_resampled = y_resampled[permute]

    return X_resampled, y_resampled



# from imblearn.combine import SMOTETomek
# from collections import Counter

# def apply_smote_tomek(X_train, y_train):
#     # Melihat distribusi kelas sebelum SMOTE-Tomek Link
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum SMOTE-Tomek Link: {class_counts}")
    
#     # Inisialisasi SMOTE-Tomek Link
#     smote_tomek = SMOTETomek(random_state=42)
    
#     # Ubah data menjadi 2D untuk kompatibilitas dengan SMOTE-Tomek
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Mengubah menjadi 2D
    
#     # Terapkan SMOTE-Tomek Link untuk oversampling dan pembersihan
#     X_resampled, y_resampled = smote_tomek.fit_resample(X_train_reshaped, y_train)
    
#     # Kembalikan data ke bentuk 3D seperti aslinya
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
#     # Melihat distribusi kelas setelah SMOTE-Tomek Link
#     print(f"Distribusi kelas setelah SMOTE-Tomek Link: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled


# def apply_smote_enn(X_train, y_train):
#     # Melihat distribusi kelas sebelum SMOTE-ENN
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum SMOTE-ENN: {class_counts}")
    
#     # Inisialisasi SMOTE-ENN
#     smote_enn = SMOTEENN(random_state=42)
    
#     # Ubah data menjadi 2D untuk kompatibilitas dengan SMOTE-ENN
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Mengubah menjadi 2D
    
#     # Terapkan SMOTE-ENN
#     X_resampled, y_resampled = smote_enn.fit_resample(X_train_reshaped, y_train)
    
#     # Kembalikan data ke bentuk 3D seperti aslinya
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
#     # Melihat distribusi kelas setelah SMOTE-ENN
#     print(f"Distribusi kelas setelah SMOTE-ENN: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled



# def apply_smote(X_train, y_train):
#     # Melihat distribusi kelas sebelum SMOTE
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum SMOTE: {class_counts}")
    
#     # Inisialisasi SMOTE
#     smote = SMOTE(random_state=42)  # Menggunakan default sampling_strategy (otomatis seimbang)
    
#     # Ubah data menjadi 2D untuk kompatibilitas dengan SMOTE
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    
#     # Terapkan SMOTE untuk oversampling kelas minoritas
#     X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
#     # Kembalikan data ke bentuk 3D seperti aslinya
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
#     # Melihat distribusi kelas setelah SMOTE
#     print(f"Distribusi kelas setelah SMOTE: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled


# def apply_bd_smote(X_train, y_train):
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum SMOTE: {class_counts}")
    
#     minority_class_label = min(class_counts, key=class_counts.get)  # Otomatis mendeteksi kelas minoritas
#     minority_class_count = class_counts[minority_class_label]
    
#     sampling_strategy = {minority_class_label: minority_class_count * 2}
#     bd_smote = BorderlineSMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=8)

#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#     X_resampled, y_resampled = bd_smote.fit_resample(X_train_reshaped, y_train)
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

#     print(f"Distribusi kelas setelah SMOTE: {Counter(y_resampled)}")
#     return X_resampled, y_resampled


# def apply_smote(X_train, y_train):
#     # Reshape X_train from (841, 3000, 1) to (841, 3000) for SMOTE
#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)
#     # Reshape X_resampled back to (num_samples, 3000, 1)
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], 1)
#     return X_resampled, y_resampled


def data_generator_np(training_files, subject_files, batch_size):
    classes = [0, 1, 2, 3, 4]
    # Load original data
    X_train = np.load(training_files[0])["x"]
    y_train = np.load(training_files[0])["y"]

    for np_file in training_files[1:]:
        X_train = np.vstack((X_train, np.load(np_file)["x"]))
        y_train = np.append(y_train, np.load(np_file)["y"])

    # Apply SMOTE
    X_resampled, y_resampled = apply_smote_2018_with_shuffle(X_train, y_train)

    # Calculate data_count for class weights
    unique, counts = np.unique(y_resampled, return_counts=True)
    data_count = list(counts)  # Convert counts to a list

    # Create train dataset with SMOTE
    train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    # Load and prepare test dataset
    X_test = []
    y_test = []
    for np_file in subject_files:
        data = np.load(np_file)
        X_test.append(data["x"])
        y_test.append(data["y"])
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    test_dataset = LoadDataset_from_numpy(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    print(f"Distribusi Kelas Testing: {Counter(y_test)}")

    return train_loader, test_loader, data_count

