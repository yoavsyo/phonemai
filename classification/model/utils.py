from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import confusion_matrix

def load_data(matrices_path, labels_path):
    matrices = torch.load(matrices_path, weights_only=False)
    labels = torch.load(labels_path, weights_only=False)
    if type(labels) != list:
        labels = labels.tolist()
    return matrices, labels


# Generate Synthetic Data
def generate_synthetic(X, y, n_clusters=5):
    X_np = X.numpy()
    y_np = np.array(y)
    num_sequences, flattened_length = X_np.shape
    X_np = X_np.reshape((num_sequences, flattened_length, 1))
    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=42)
    y_pred = km.fit_predict(X_np)
    synthetic_data = km.cluster_centers_.reshape((n_clusters, flattened_length))
    synthetic_labels = [np.bincount(y_np[np.where(y_pred == i)[0]]).argmax() for i in range(n_clusters)]
    synthetic_tensors = torch.tensor(synthetic_data, dtype=torch.float32)
    synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.int64)
    return synthetic_tensors, synthetic_labels


# Train the Model
def train_model(model, train_dataloader, eval_dataloader, criterion, optimizer, num_epochs, model_name):
    train_losses, eval_losses = [], []
    train_accuracies, eval_accuracies = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_f1, best_acc, best_loss = 0.0, 0.0, np.inf

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        training_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        training_accuracy = accuracy_score(train_labels, train_preds)
        train_losses.append(training_loss / len(train_dataloader))
        train_accuracies.append(training_accuracy)

        model.eval()
        eval_preds, eval_labels = [], []
        eval_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in eval_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                eval_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(batch_y.cpu().numpy())

        eval_f1 = f1_score(eval_labels, eval_preds, average='weighted')
        eval_accuracy = accuracy_score(eval_labels, eval_preds)
        eval_accuracies.append(eval_accuracy)
        eval_losses.append(eval_loss / len(eval_dataloader))

        if best_f1 <= eval_f1:
            best_acc = eval_accuracy
            best_f1 = eval_f1
            best_loss = eval_losses[-1]
            torch.save(model.state_dict(), model_name)

    return train_losses, eval_losses, train_accuracies, eval_accuracies


# Test the Model
def test_model(model, test_dataloader, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())

    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='weighted')
    return test_accuracy, test_f1, precision, recall, test_preds,test_labels


# Plot Results
def plot_results(patient,train_losses, eval_losses, train_accuracies, eval_accuracies, num_epochs):
    def smooth_curve(points, factor=0.85):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    train_losses = smooth_curve(train_losses)
    eval_losses = smooth_curve(eval_losses)
    train_accuracies = smooth_curve(train_accuracies)
    eval_accuracies = smooth_curve(eval_accuracies)

    plt.figure(figsize=(12, 5))
    epochs = range(1, num_epochs + 1)
    plt.suptitle(f"PATIENT {patient}")
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, eval_losses, label='Eval Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, eval_accuracies, label='Eval Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.savefig('accuracy_plot.png')

    plt.show()


# Plot Confusion Matrix
def plot_confusion_matrix(test_labels, test_preds,patient):
    cm = confusion_matrix(test_labels, test_preds)
    cm = cm / cm.sum(axis=1)[:, None]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for patient # {patient}')
    plt.savefig(f'plots/confusion_matrix_p{patient}.png')
    plt.show()


def organize_data(matrices, labels, device, batch_size,optuna):
    x_train, x_temp, y_train, y_temp = train_test_split(matrices, labels, train_size=0.65, test_size=0.35,
                                                        random_state=42, shuffle=True, stratify=labels)
    x_eval, x_test, y_eval, y_test = train_test_split(x_temp, y_temp, train_size=0.50, test_size=0.50,
                                                      random_state=42, shuffle=True, stratify=y_temp)

    y_train_np, y_eval_np, y_test_np = np.array(y_train), np.array(y_eval), np.array(y_test)

    train_label_distribution, eval_label_distribution, test_label_distribution = Counter(y_train_np), Counter(
        y_eval_np), Counter(y_test_np)
    if not optuna:
        print('Training label distribution:', train_label_distribution)
        print('Evaluation label distribution:', eval_label_distribution)
        print('Test label distribution:', test_label_distribution)

    x_train, x_eval, x_test = x_train.float().to(device), x_eval.float().to(device), x_test.float().to(device)
    y_train, y_eval, y_test = torch.tensor(y_train).clone().detach().to(device), torch.tensor(
        y_eval).clone().detach().to(
        device), torch.tensor(y_test).clone().detach().to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    eval_dataset = TensorDataset(x_eval, y_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size)
    return train_dataloader, eval_dataloader, test_dataloader


