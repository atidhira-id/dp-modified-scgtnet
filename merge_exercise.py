import scipy.io as sio
import numpy as np
import os

def remap_stimulus(stimulus, offset):
    remapped = stimulus.copy()
    remapped[stimulus != 0] = stimulus[stimulus != 0] + offset
    return remapped

def merge_per_subject(subject_id, data_dir, output_dir):
    # Load all exercise files
    path_e1 = os.path.join(data_dir, f'S{subject_id}_E1_A1.mat')
    path_e2 = os.path.join(data_dir, f'S{subject_id}_E2_A1.mat')
    path_e3 = os.path.join(data_dir, f'S{subject_id}_E3_A1.mat')

    data_e1 = sio.loadmat(path_e1)
    data_e2 = sio.loadmat(path_e2)
    data_e3 = sio.loadmat(path_e3)

    # Take sEMG data
    emg_e1 = data_e1['emg']
    emg_e2 = data_e2['emg']
    emg_e3 = data_e3['emg']

    # Take stimulus
    stim_e1 = data_e1['stimulus'].flatten()
    stim_e2 = data_e2['stimulus'].flatten()
    stim_e3 = data_e3['stimulus'].flatten()

    # Remap stimulus E2 dan E3
    stim_e2_remap = remap_stimulus(stim_e2, offset=12)
    stim_e3_remap = remap_stimulus(stim_e3, offset=29)

    # Verify stimulus after remap
    print(f"Subjek {subject_id}:")
    print(f"  E1 stimulus: {np.unique(stim_e1)}")
    print(f"  E2 stimulus (setelah remap): {np.unique(stim_e2_remap)}")
    print(f"  E3 stimulus (setelah remap): {np.unique(stim_e3_remap)}")

    # Merge sEMG dan stimulus vertically
    emg_merge    = np.concatenate([emg_e1, emg_e2, emg_e3], axis=0)
    stim_merge   = np.concatenate([stim_e1, stim_e2_remap, stim_e3_remap], axis=0)

    print(f"  Total rows after merge: {emg_merge.shape[0]}")
    print(f"  Stimulus after merge: {np.unique(stim_merge)}")
    print()

    # Save as .mat file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'S{subject_id}_merge.mat')
    sio.savemat(output_path, {
        'emg': emg_merge,
        'stimulus': stim_merge
    })
    print(f"  File saved: {output_path}")

# Run for all 10 subject
data_dir   = 'data/raw'
output_dir = 'data/merge'

for subject_id in range(1, 11):
    merge_per_subject(subject_id, data_dir, output_dir)

print("Done! All subject successfully merged.")