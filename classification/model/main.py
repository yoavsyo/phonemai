import json
import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data, train_model, test_model, plot_results, plot_confusion_matrix, organize_data
from model import EEGEncoderDecoder

def multi_run_stats(patient, num_runs=20, epochs=30, hidden_size=200, num_lstm_layers=1, num_fc_layers=1, dropout_prob=0,
                learning_rate=0.01, weight_decay=0.01, batch_size=8, test=False, optuna=False):
    # Lists to store results for 20 runs
    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    y_pred_list=[]
    y_true_list=[]

    for run in range(num_runs):
        print(f'Running {run + 1}/{num_runs} for patient {patient}...')
        test_accuracy, test_f1, precision, recall , y_pred,y_true= main(patient=patient, epochs=epochs, hidden_size=hidden_size,
                                                         num_lstm_layers=num_lstm_layers, num_fc_layers=num_fc_layers,
                                                         dropout_prob=dropout_prob, learning_rate=learning_rate,
                                                         weight_decay=weight_decay, batch_size=batch_size,
                                                          optuna=optuna, test_multi_runs=True)

        # Collect results
        accuracy_list.append(test_accuracy)
        f1_list.append(test_f1)
        precision_list.append(precision)
        recall_list.append(recall)
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    # Compute averages and maximums for each metric
    avg_accuracy = np.mean(accuracy_list)
    max_accuracy = np.max(accuracy_list)
    avg_f1 = np.mean(f1_list)
    max_f1 = np.max(f1_list)
    avg_precision = np.mean(precision_list)
    max_precision = np.max(precision_list)
    avg_recall = np.mean(recall_list)
    max_recall = np.max(recall_list)
    # Find the index of the maximum recall
    max_recall_index = np.argmax(recall_list)

    # Get the y_pred and y_true corresponding to the maximum recall
    y_pred_max_recall = y_pred_list[max_recall_index]
    y_true_max_recall = y_true_list[max_recall_index]
    plot_confusion_matrix(np.array(y_true_max_recall), np.array(y_pred_max_recall), patient)
    # Return as a dictionary for easy table creation, including y_pred and y_true for max recall
    return {
        'Patient': patient,
        'Avg Accuracy': avg_accuracy,
        'Max Accuracy': max_accuracy,
        'Avg F1': avg_f1,
        'Max F1': max_f1,
        'Avg Precision': avg_precision,
        'Max Precision': max_precision,
        'Avg Recall': avg_recall,
        'Max Recall': max_recall,
        'y_pred_max_recall': y_pred_max_recall,
        'y_true_max_recall': y_true_max_recall
    }

# Main Function
def main(patient=18, epochs=30, hidden_size=200, num_lstm_layers=1, num_fc_layers=1, dropout_prob=0, learning_rate=0.01, weight_decay=0.01, batch_size=8, optuna=False,
         test_multi_runs=False):
    if not optuna:
        with open(f'/home/tauproj4/PycharmProjects/Classification/best_params/patient_{patient}_with_loss.json', 'r') as file:
            config = json.load(file)
        # Extracting variables from the dictionary
        hidden_size = config['hidden_size']
        num_lstm_layers = config['num_lstm_layers']
        num_fc_layers = config['num_fc_layers']
        dropout_prob = config['dropout_prob']
        learning_rate = config['learning_rate']
        weight_decay = config['weight_decay']
        batch_size = config['batch_size']
        epochs = config['epochs']

    matrices_path = f'/home/tauproj4/PycharmProjects/Classification/data/matrices/matrices_{patient}.pt'
    labels_path = f'/home/tauproj4/PycharmProjects/Classification/data/labels/labels_{patient}.pt'
    model_name = 'try.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices, labels = load_data(matrices_path, labels_path,)

    num_channels = matrices.shape[1]
    num_classes = np.unique(labels).size
    train_dataloader, eval_dataloader, test_dataloader = organize_data(matrices, labels, device, batch_size ,optuna)

    model = EEGEncoderDecoder(input_size=num_channels, hidden_size=hidden_size,
                              num_lstm_layers=num_lstm_layers,
                              num_fc_layers=num_fc_layers, num_classes=num_classes,
                              dropout_prob=dropout_prob).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses, eval_losses, train_accuracies, eval_accuracies = train_model(model, train_dataloader, eval_dataloader,
                                                                               criterion, optimizer, epochs,
                                                                               model_name)
    if optuna:
        return eval_accuracies[-1],eval_losses[-1]

    else:
        test_accuracy, test_f1, precision, recall, y_pred,y_true = test_model(model, test_dataloader, model_name)

        print(f'Best Model Test Accuracy: {test_accuracy * 100:.2f}%')
        print(f'F1 Score: {test_f1:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')

        if test_multi_runs:
            return test_accuracy, test_f1, precision, recall,  y_pred,y_true
        else:
            plot_confusion_matrix(np.array(y_true), np.array(y_pred), patient)
            plot_results(patient, train_losses, eval_losses, train_accuracies, eval_accuracies, epochs)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    mode="run_multi"

    if mode=="run_multi":
        results = []
        for patient in range(20):
            if os.path.exists(f'/home/tauproj4/PycharmProjects/Classification/data/matrices/matrices_{patient}.pt'):
                print(f"-----PATIENT {patient}-------")
                patient_stats = multi_run_stats(patient)
                results.append(patient_stats)


        # Save the results to a table (e.g., CSV or Pandas DataFrame)
        df_results = pd.DataFrame(results)
        df_results.to_csv('patient_results.csv', index=False)
        print(df_results)

    else:
        for patient in range(20):
            if os.path.exists(f'/home/tauproj4/PycharmProjects/Classification/data/matrices/matrices_{patient}.pt'):
                print(f"-----PATIENT {patient}-------")
                main(patient,test=True)
