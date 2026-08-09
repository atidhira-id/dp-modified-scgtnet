import numpy as np
import scipy.io as sio
from collections import Counter

data       = sio.loadmat('data/windows/all_subjects_windows.mat')
labels     = data['labels'].flatten()
subject_id = data['subject_id'].flatten()

# Cek 1 — Total window per kelas
print('=== Distribusi Window per Kelas ===')
counter = Counter(labels)
jumlah_list = list(counter.values())
print(f'Min window per kelas : {min(jumlah_list)}')
print(f'Max window per kelas : {max(jumlah_list)}')
print(f'Rata-rata per kelas  : {np.mean(jumlah_list):.0f}')
print(f'Total window         : {len(labels)}')

# Cek 2 — Window per subjek
print('\n=== Window per Subjek ===')
for sid in np.unique(subject_id):
    jumlah = np.sum(subject_id == sid)
    print(f'  Subjek {sid:2d}: {jumlah} window')

# Cek 3 — Window per kelas per subjek (apakah ada kelas yang kosong?)
print('\n=== Kelas yang memiliki < 5 window per subjek ===')
for sid in np.unique(subject_id):
    mask = subject_id == sid
    label_subjek = labels[mask]
    for kelas in np.unique(labels):
        jumlah = np.sum(label_subjek == kelas)
        if jumlah < 5:
            print(f'  Subjek {sid}, Kelas {kelas}: {jumlah} window')