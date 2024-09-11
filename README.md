# PhonemAi

## Neural Signals Classification and Detection
This repository contains two distinct models designed for neural signal processing: one for classification and the other for detection. Both models are applied to neural activity data, with a focus on decoding speech-related thoughts, specifically targeting the vowels (a/e/u/i/o).

<img width="1225" alt="Screenshot 2024-09-11 at 19 02 36" src="https://github.com/user-attachments/assets/56bd0fd6-8bc5-40a8-80a8-8e4f8e77301d">

### Project Structure
׳׳׳
   ├── LICENSE                           # License for the project
   ├── README.md                         # Project documentation
   ├── classification                    # Folder containing classification model files
   │   ├── best_params                   # Folder with best parameter configurations for different patients
   │   │   ├── best_params_with_loss.json        # General best parameters file including loss
   │   │   ├── patient_11_optuna.json            # Optimized parameters for patient 11 using Optuna
   │   │   ├── patient_11_with_loss.json         # Parameters and loss for patient 11
   │   │   ├── patient_13_optuna.json            # Optimized parameters for patient 13 using Optuna
   │   │   ├── patient_13_with_loss.json         # Parameters and loss for patient 13
   │   │   ├── patient_15_optuna.json            # Optimized parameters for patient 15 using Optuna
   │   │   ├── patient_15_with_loss.json         # Parameters and loss for patient 15
   │   │   ├── patient_18_optuna.json            # Optimized parameters for patient 18 using Optuna
   │   │   ├── patient_18_with_loss.json         # Parameters and loss for patient 18
   │   │   ├── patient_19_optuna.json            # Optimized parameters for patient 19 using Optuna
   │   │   ├── patient_19_with_loss.json         # Parameters and loss for patient 19
   │   │   ├── patient_1_optuna.json             # Optimized parameters for patient 1 using Optuna
   │   │   ├── patient_1_with_loss.json          # Parameters and loss for patient 1
   │   │   ├── patient_2_optuna.json             # Optimized parameters for patient 2 using Optuna
   │   │   ├── patient_2_with_loss.json          # Parameters and loss for patient 2
   │   │   ├── patient_3_optuna.json             # Optimized parameters for patient 3 using Optuna
   │   │   ├── patient_3_with_loss.json          # Parameters and loss for patient 3
   │   │   ├── patient_5_optuna.json             # Optimized parameters for patient 5 using Optuna
   │   │   ├── patient_5_with_loss.json          # Parameters and loss for patient 5
   │   │   ├── patient_6_optuna.json             # Optimized parameters for patient 6 using Optuna
   │   │   ├── patient_6_with_loss.json          # Parameters and loss for patient 6
   │   │   ├── patient_7_optuna.json             # Optimized parameters for patient 7 using Optuna
   │   │   └── patient_7_with_loss.json          # Parameters and loss for patient 7
   │   └── model                        # Folder containing the model files and plots for classification
   │       ├── accuracy_plot.png         # Plot showing model accuracy
   │       ├── best_params_with_loss.json        # General best parameters file with loss for the classification model
   │       ├── loss_plot.png             # Plot showing model loss over time
   │       ├── main.py                   # Main script to run the classification model
   │       ├── model.py                  # Definition of the LSTM model architecture
   │       ├── parmas_optimizer.py       # Script for optimizing parameters using Optuna
   │       ├── patient_results.csv       # CSV file containing results for different patients
   │       ├── plots                     # Directory for additional plots
   │       ├── try.pth                   # Pre-trained PyTorch model file
   │       └── utils.py                  # Utility functions for classification
   ├── detection                         # Folder containing the detection model
   │   └── detection.ipynb               # Jupyter notebook for detecting speech-related thoughts
   └── requirements.txt                  # List of required Python packages for running the project
 ׳׳׳

### Models
1. Classification Model
Purpose: This model classifies the neural activity to predict when a patient is thinking about one of the vowels (a, e, u, i, o). The model is fine-tuned for each patient, and its best parameters are stored for reproducibility.
Files:
model/main.py: Main script for training and evaluating the classification model.
model/model.py: Defines the LSTM model architecture.
best_params/: Contains best parameters for different patients optimized using Optuna.
plots/: Contains accuracy and loss plots for training.
patient_results.csv: CSV with results per patient.
2. Detection Model
Purpose: This model is used for detecting specific neural signals that indicate when a patient is about to think of a vowel. It processes time series data and makes real-time predictions.
Files:
detection/detection.ipynb: Jupyter notebook for running and analyzing the detection model.
### Installation
To run the models, ensure you have all the necessary dependencies installed. You can install them using the requirements.txt file.
pip install -r requirements.txt
### Usage
Classification Model
Navigate to the classification/model folder.
Run the main.py script to train and evaluate the model:
   python main.py
   
The best parameters for each patient can be found in the best_params folder, optimized with loss included in the results.
Detection Model
Open the detection/detection.ipynb notebook in Jupyter.
Run the cells to load the data, process it, and analyze the detection model results.
### Results
Classification Model: Achieves a variable accuracy depending on the patient and the vowel being classified. The model leverages memory through an LSTM layer to predict speech-related brain activity.
Detection Model: Detects speech-related thoughts with some time delay, aiming to achieve real-time performance for detecting vowel thought patterns.
### License
This project is licensed under the MIT License. See the LICENSE file for details.
