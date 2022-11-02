from xmlrpc.client import boolean
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from Util.one_cycle_lr import OneCycleLR
import Util.SCA_util as SCA_util
import Util.SCA_dataset as SCA_dataset
import Util.DL_model as DL_model

if __name__ == "__main__":

    file_root = "" # dataset root
    result_root = "" # result root
    datasets = ['ASCAD'] # ASCAD/ASCAD_rand/CHES_CTF
    leakages = ['HW'] # HW/ID
    attack_model = 'MLP' # MLP/CNN
    metrics = 'all' # 'val_age/val_key_rank/val_acc'
    noise_level = 0
    idx = 1 # nameing index
    sigma_hw = 0 # sigma for the HW leakage model
    sigma_id = 0 # sigma for the ID leakage model
    profiling_tracess = [50000]
    model_size = 64 # the size of the profiling model
    epochs = [10] # training epoch
    data_arguementation = False # enable/disbale data arguementation
    data_arguementation_level = 0.25 # data arguementation level

    for profiling_traces in profiling_tracess:
        profiling_traces = int(profiling_traces)
        for leakage in leakages:
            for dataset in datasets:
                # For metric calculation
                nb_traces_attacks_metric = 5000
                nb_attacks_metric = 1
                # For GE calculation
                nb_traces_attacks = 5000
                nb_attacks = 20
                # Load dataset
                if dataset == 'CHES_CTF' and leakage == 'ID':
                    continue
                if dataset == 'ASCAD':
                    data_root = dataset + '/Base_desync0.h5'
                    correct_key = 224
                    attack_byte = 2
                    (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack), (key_profiling,  key_attack) = SCA_dataset.load_ascad(file_root + data_root, leakage_model=leakage, profiling_traces=profiling_traces, key_info=True)
                elif dataset == 'ASCAD_rand':
                    data_root = dataset + '/ascad-variable.h5'
                    correct_key = 34
                    attack_byte = 2
                    (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack), (key_profiling,  key_attack) = SCA_dataset.load_ascad_rand(file_root + data_root, leakage_model=leakage, profiling_traces=profiling_traces, key_info=True)
                elif dataset == 'CHES_CTF':
                    data_root = dataset
                    correct_key = 46
                    attack_byte = 0
                    (X_profiling, X_attack), (Y_profiling,  Y_attack), (plt_profiling,  plt_attack), (key_profiling,  key_attack) = SCA_dataset.load_chesctf(file_root + data_root, leakage_model=leakage, profiling_traces=profiling_traces, key_info=True)
                else:
                    print('No dataset defined!')
                    sys.exit(-1)

                # Normalize dataset
                scaler = StandardScaler()
                X_profiling = scaler.fit_transform(X_profiling)
                X_attack = scaler.transform(X_attack)
                
                # Performing data arguementation
                if data_arguementation:
                    X_profiling, Y_profiling, plt_profiling = SCA_dataset.data_augmentation_gaussian_noise(X_profiling, Y_profiling, plt_profiling, arg_level=data_arguementation_level)

                print('Profiling traces number: {}'.format(len(X_profiling)))
                print('Attack traces number: {}'.format(len(X_attack)))

                # Reshape data for DL training
                if attack_model == 'CNN':
                    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
                    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

                # Select leakage model
                if leakage == 'ID':
                    classes = 256
                else:
                    classes = 9

                # Prepare the label: label+identifier+pleintext (for metric calculation)
                if sigma_hw==0 or sigma_id==0:
                    print('noLD_{}_{}'.format(sigma_hw, sigma_id))
                    Y_profiling = np.concatenate((to_categorical(Y_profiling, num_classes=classes), np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
                    Y_attack = np.concatenate((to_categorical(Y_attack, num_classes=classes), np.ones((len(plt_attack), 1)), plt_attack), axis=1)
                else:
                    print('LD_{}_{}'.format(sigma_hw, sigma_id))
                    if leakage == 'HW':
                        Y_profiling = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_profiling, leakage, sigma_hw, sigma_id), np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
                        Y_attack = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_attack, leakage, sigma_hw, sigma_id), np.ones((len(plt_attack), 1)), plt_attack), axis=1)
                    else:
                        Y_profiling = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_profiling, leakage, sigma_hw, sigma_id), np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
                        Y_attack = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_attack, leakage, sigma_hw, sigma_id), np.ones((len(plt_attack), 1)), plt_attack), axis=1)                    

                KD = SCA_util.Key_Distribution('HW')
                Atk_ge = SCA_util.Attack(KD, leakage, correct_key, nb_traces_attacks=nb_traces_attacks_metric, nb_attacks=nb_attacks_metric, attack_byte=attack_byte, shuffle=True, output='prob_metric')
                Loss = SCA_util.custom_loss(leakage, Atk_ge)
                
                # Metric selection: ACC/AGE/key_rank
                if metrics == 'val_accuracy':
                    metric=[SCA_util.acc_Metric(leakage)]      
                elif metrics == 'val_age':  
                    metric=[SCA_util.AGE_Metric(KD, leakage, correct_key, Atk_ge)] 
                elif metrics == 'val_key_rank':
                    metric=[SCA_util.key_rank_Metric(leakage, correct_key, Atk_ge)] 
                elif metrics == 'all':
                    metric=[SCA_util.acc_Metric(leakage), SCA_util.key_rank_Metric(leakage, correct_key, Atk_ge), SCA_util.AGE_Metric(KD, leakage, correct_key, Atk_ge)] 
                else:
                    print('No metric defined!')
                    metric=[]

                if sigma_hw==0 or sigma_id==0:
                    print('cce loss')
                    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage, X_profiling.shape[1], metric, Loss.categorical_crossentropy, model=attack_model, model_size=model_size)
                else:
                    print('KL loss')
                    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage, X_profiling.shape[1], metric, Loss.KL, model=attack_model, model_size=model_size)

                val_accuracy_value = []
                val_key_rank_value = []
                val_age_value = []
                val_t_ge0_value = []

                for epoch_idx, epoch in enumerate(epochs):
                    saving_name = '{}_{}_{}_metric-{}_noise{}_sigmaHW{}_sigmaID{}_profTrs{}_modelSize{}_epoch{}_dataArg{}_{}'.format(dataset, leakage, attack_model, metrics, noise_level, sigma_hw, sigma_id, profiling_traces, model_size, epoch, data_arguementation_level, idx)

                    if epoch == 'best':
                        epoch = epoch_sota
                    else:
                        if epoch_idx == 0:
                            epoch = int(epoch)
                        else:
                            epoch = int(epoch) - int(epochs[epoch_idx-1])

                    if attack_model == 'CNN':
                        callback = OneCycleLR(len(X_profiling), batch_size, 5e-3, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=True)
                        history = model.fit(x=X_profiling, y=Y_profiling, validation_data=(X_attack[:nb_traces_attacks_metric], Y_attack[:nb_traces_attacks_metric]), batch_size=batch_size, verbose=2, epochs=epoch, callbacks=[callback])
                    else:
                        history = model.fit(x=X_profiling, y=Y_profiling, validation_data=(X_attack[:nb_traces_attacks_metric], Y_attack[:nb_traces_attacks_metric]), batch_size=batch_size, verbose=2, epochs=epoch)

                    # save metrics
                    val_accuracy_value.append(history.history['val_accuracy'])
                    val_key_rank_value.append(history.history['val_key_rank'])
                    val_age_value.append(history.history['val_age'])

                    # Attack on the test traces
                    predictions = model.predict(X_attack[:nb_traces_attacks_metric])
                    
                    Atk_ge_age = SCA_util.Attack(KD, leakage, correct_key, nb_traces_attacks=nb_traces_attacks, nb_attacks=nb_attacks, attack_byte=attack_byte, shuffle=True, output='rank')
                    avg_rank = np.array(Atk_ge_age.perform_attacks(predictions, plt_attack))

                    val_t_ge0 = np.argmax(avg_rank[:, correct_key] < 1)

                    if val_t_ge0 == 0:
                        val_t_ge0_value.append([nb_traces_attacks])
                        print('GE smaller than 1:', nb_traces_attacks)
                    else:
                        val_t_ge0_value.append([val_t_ge0])
                        print('GE smaller than 1:', val_t_ge0)
                    
                np.save(result_root+'Metric_'+saving_name+'.npy', [np.array(val_accuracy_value).flatten(), np.array(val_key_rank_value).flatten(), np.array(val_age_value).flatten()])
                np.save(result_root+'TGE0_'+saving_name+'.npy', np.array(val_t_ge0_value).flatten())

                K.clear_session()
