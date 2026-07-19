import numpy as np
import scipy.io as sio
import os


WINDOW_SIZE = 40
STEP_SIZE   = 20

def windowing(emg, stimulus, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    N = emg.shape[0]
    windows = []
    labels = []
    
    start = 0
    while start + window_size <= N:
        end = start + window_size
        
        window_emg = emg[start:end, :]
        window_stim = stimulus[0, start:end]
        
        label_uniq = np.unique(window_stim)
        
        if len(label_uniq) == 1:
            label = label_uniq[0]
            
            if label != 0:
                windows.append(window_emg)
                labels.append(label)
        
        start += step_size
    
    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    return windows, labels


def windowing_data(subject_id, data_dir, output_dir):
    path_data = os.path.join(data_dir, f'S{subject_id}_merge.mat')
    
    data = sio.loadmat(path_data)
    
    emg = data['emg']
    stim = data['stimulus']
    
    windows, labels = windowing(emg, stim, WINDOW_SIZE, STEP_SIZE)
    
    
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, f'S{subject_id}_windows.mat')
    sio.savemat(path_output, {
        'windows'   : windows,
        'labels'    : labels,
    })
    
    print(f'Windowing subject {subject_id} saved on {path_output}')
    

if __name__ == '__main__':
    data_dir = 'data/merge'
    output_dir = 'data/windows'

    for subject_id in range(1, 11):
        windowing_data(subject_id, data_dir, output_dir)
        

