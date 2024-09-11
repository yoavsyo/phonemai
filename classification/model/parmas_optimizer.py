import json
import os
import optuna
from encoder_decoder.main import main

for patient in range(20):
    if os.path.exists(f'/home/tauproj4/PycharmProjects/Classification/data/matrices/matrices_{patient}.pt'):
        print (f"-----STARTING OPTIMIZATION FOR PATIENT {patient}-------")
        params_path = '/home/tauproj4/PycharmProjects/Classification/best_params/'
        best_val_acc = 0
        best_loss = float('inf')
        best_params = {}

        def objective(trial):
            global best_val_acc, best_loss, best_params

            hidden_size = trial.suggest_int('hidden_size', 64, 512)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 2, 3)
            num_fc_layers = trial.suggest_int('num_fc_layers', 2, 5)
            dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5, step=0.1)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-1, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
            epochs = trial.suggest_int('epochs', 20, 80)
            val_acc, val_loss = main(patient, epochs, hidden_size, num_lstm_layers, num_fc_layers, dropout_prob, learning_rate,
                                     weight_decay, batch_size,
                                     test=False,
                                     optuna=True)

            if val_acc >= best_val_acc and val_loss < best_loss:
                best_val_acc = val_acc
                best_loss = val_loss

                best_params['val_acc'] = best_val_acc
                best_params['val_loss'] = best_loss

                best_params['epochs'] = epochs
                best_params['learning_rate'] = learning_rate
                best_params['weight_decay'] = weight_decay
                best_params['batch_size'] = batch_size
                best_params['num_lstm_layers'] = num_lstm_layers
                best_params['num_fc_layers'] = num_fc_layers
                best_params['dropout_prob'] = dropout_prob
                best_params['hidden_size'] = hidden_size
                with open(f'{params_path}patient_{patient}_with_loss.json', 'w') as f:
                    json.dump(best_params, f, indent=4)

                return val_acc

            return val_acc


        study = optuna.create_study(directions=['maximize'])
        study.optimize(objective, n_trials=200)

        with open(f'{params_path}patient_{patient}_optuna.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)

        # print('\n\n')
        # print(f'optuna best_param:   {study.best_params}')
        # print(f'my_best_params:      {best_params}')
