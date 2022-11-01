import numpy as np
import h5py
from scipy import stats
import random

def add_GussianNoise(traces, noise_level):
    print('Add Gussian noise: ', noise_level)
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(
                0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces

def add_Desync(traces, desync_level):
    print('Add desync noise...')
    traces_length = len(traces[0])

    if desync_level == 0:
        return traces
    else:
        output_traces = np.zeros((len(traces), traces_length-desync_level))
        for idx in range(len(traces)):
            if(idx % 2000 == 0):
                print(str(idx) + '/' + str(len(traces)))
            rand = np.random.randint(low=0, high=desync_level)
            output_traces[idx] = traces[idx][rand:rand+traces_length-desync_level]
        return output_traces

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]

# select POI with CPA
def POI_selection(traces, label, numPOIs, POIspacing):
    corr = np.zeros(len(traces[0]))

    for i in range(len(traces[0])):
        if (traces[:, i] == traces[:, i][0]).all():
            corr[i] = 0
        else:
            corr[i] = abs(stats.pearsonr(traces[:, i], label)[0])

    POIs = []
    # Repeat until we have enough POIs
    for i in range(numPOIs):
        # Find the biggest peak and add it to the list of POIs
        nextPOI = corr.argmax()
        POIs.append(nextPOI)
        # Zero out some of the surrounding points
        # Make sure we don't go out of bounds
        poiMin = max(0, nextPOI - POIspacing)
        poiMax = min(nextPOI + POIspacing, len(corr))
        for j in range(poiMin, poiMax):
            corr[j] = 0
    return POIs

def shuffle_data(profiling_x,label_y,plt):
    l = list(zip(profiling_x,label_y,plt))
    random.shuffle(l)
    shuffled_x,shuffled_y,shuffled_plt = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    shuffled_plt = np.array(shuffled_plt)
    return (shuffled_x, shuffled_y, shuffled_plt)

def data_augmentation_gaussian_noise(trace, label, plt, arg_level=0.25, arg_times=10):
    trs_num = len(trace)
    arg_trace = np.zeros((trs_num*arg_times, len(trace[0])))
    arg_label = np.zeros(trs_num*arg_times)
    arg_plt = np.zeros((trs_num*arg_times, 16))
    i = 0
    while i < arg_times:
        arg_trace[i*trs_num:(i+1)*trs_num] = add_GussianNoise(trace, arg_level)
        arg_label[i*trs_num:(i+1)*trs_num] = label
        arg_plt[i*trs_num:(i+1)*trs_num] = plt
        i+=1
    
    return shuffle_data(arg_trace,arg_label,arg_plt)

def load_ascad(ascad_database_file, leakage_model='HW', key_info=False, profiling_traces=50000, attack_trace=10000):
    in_file = h5py.File(ascad_database_file, "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[:profiling_traces]
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int16)[:profiling_traces]
    # Attack byte 2
    plt_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'], dtype=np.int16)[:profiling_traces]
    key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'], dtype=np.int16)[:profiling_traces]
    
    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[:attack_trace]
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int16)[:attack_trace]
    # Attack byte 2
    plt_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'], dtype=np.int16)[:attack_trace]
    key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'], dtype=np.int16)[:attack_trace]

    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)

    print('Loaded profiling traces number: {}'.format(len(X_profiling)))
    print('Loaded attack traces number: {}'.format(len(X_attack)))
    if key_info:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack), (key_profiling, key_attack)
    else:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack)

def load_ascad_rand(ascad_database_file, leakage_model='HW', key_info=False, profiling_traces=50000, attack_trace=100000):
    in_file = h5py.File(ascad_database_file, "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[:profiling_traces]
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int16)[:profiling_traces]
    # Attack byte 2
    plt_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'], dtype=np.int16)[:profiling_traces]
    key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'], dtype=np.int16)[:profiling_traces]
    
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[:attack_trace]
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int16)[:attack_trace]
    # Attack byte 2
    plt_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'], dtype=np.int16)[:attack_trace]
    key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'], dtype=np.int16)[:attack_trace]

    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)

    print('Loaded profiling traces number: {}'.format(len(X_profiling)))
    print('Loaded attack traces number: {}'.format(len(X_attack)))
    if key_info:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack), (key_profiling, key_attack)
    else:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack)

def load_chesctf(database_file, leakage_model='HW', key_info=False, profiling_traces=50000, attack_trace=10000):
    in_file = h5py.File(database_file+'/ches_ctf.h5', "r")
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[:profiling_traces]
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[:attack_trace]
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    Y_profiling = np.load(database_file + '/train_labels.npy')[:profiling_traces]
    Y_attack = np.load(database_file + '/attack_labels.npy')[:attack_trace]
    key_profiling = np.load(database_file + '/train_key.npy')[:profiling_traces]
    key_attack = np.load(database_file + '/attack_key.npy')[:attack_trace]    
    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)
    # Attack byte 0
    plt_profiling = np.load(database_file + '/plt_train.npy')[:profiling_traces]
    plt_attack = np.load(database_file + '/plt_attack.npy')[:attack_trace]
    print('Profiling traces number: {}'.format(len(X_profiling)))
    print('Attack traces number: {}'.format(len(X_attack)))
    if key_info:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack), (key_profiling, key_attack)
    else:
        return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (plt_profiling,  plt_attack)

def load_chipwhisperer(chipwhisper_folder, leakage_model='HW', profiling_traces=7000, attack_trace=3000):
    X_profiling = np.load(chipwhisper_folder + 'traces.npy')[:10000]
    Y_profiling = np.array(np.load(chipwhisper_folder + 'labels.npy')[:10000], dtype=np.int16)
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling), dtype=np.int16)
    # Attack byte 1
    plt_profiling = np.array(np.load(chipwhisper_folder + 'plain.npy')[:10000], dtype=np.int16)
    return (X_profiling[:profiling_traces], X_profiling[profiling_traces:profiling_traces+attack_trace]), (Y_profiling[:profiling_traces],  Y_profiling[profiling_traces:profiling_traces+attack_trace]), (plt_profiling[:profiling_traces],  plt_profiling[profiling_traces:profiling_traces+attack_trace])


