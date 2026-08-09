import scipy.io as sio
import numpy as np
import os

WINDOW_SIZE = 100

def merge_windows(windows_dir, n_subjects=10):
    all_windows = []
    all_labels = []
    all_subject_id = []
    
    for subject in range(1, n_subjects+1):
        path = os.path.join(windows_dir, f'S{subject}_windows.mat')
        
        if not os.path.exists(path):
            print(f'File {path} not found')
            continue
        
        data = sio.loadmat(path)
        w = data["windows"]
        l = np.array(data['labels']).flatten()
        
        all_windows.append(w)
        all_labels.append(l)
        all_subject_id.append(np.full(len(l), subject, dtype=np.int32))
        
    all_windows    = np.concatenate(all_windows, axis=0)
    all_labels     = np.concatenate(all_labels, axis=0)
    all_subject_id = np.concatenate(all_subject_id, axis=0)
    
    output_path = os.path.join(windows_dir, 'all_subjects_windows.mat')
    sio.savemat(output_path, {
        'windows'   : all_windows,
        'labels'    : all_labels,
        'subject_id': all_subject_id
    })
    
    print(f"Merge done!")
    return all_windows, all_labels, all_subject_id


if __name__ == '__main__':
    windows_dir = f'data\windows\{WINDOW_SIZE}'

    merge_windows(windows_dir, 10)
