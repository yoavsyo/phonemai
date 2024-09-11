# PhonemAi

## Neural Signal Classification and Detection
This repository contains two distinct models designed for neural signal processing: one for classification and the other for detection. Both models are applied to neural activity data, with a focus on decoding speech-related thoughts, specifically targeting the vowels (a/e/u/i/o).

<img width="1225" alt="Screenshot 2024-09-11 at 19 02 36" src="https://github.com/user-attachments/assets/56bd0fd6-8bc5-40a8-80a8-8e4f8e77301d">

### Project Structure
├── LICENSE
├── README.md
├── classification
│   ├── best_params
│   │   ├── best_params_with_loss.json
│   │   ├── patient_11_optuna.json
│   │   ├── patient_11_with_loss.json
│   │   ├── patient_13_optuna.json
│   │   ├── patient_13_with_loss.json
│   │   ├── patient_15_optuna.json
│   │   ├── patient_15_with_loss.json
│   │   ├── patient_18_optuna.json
│   │   ├── patient_18_with_loss.json
│   │   ├── patient_19_optuna.json
│   │   ├── patient_19_with_loss.json
│   │   ├── patient_1_optuna.json
│   │   ├── patient_1_with_loss.json
│   │   ├── patient_2_optuna.json
│   │   ├── patient_2_with_loss.json
│   │   ├── patient_3_optuna.json
│   │   ├── patient_3_with_loss.json
│   │   ├── patient_5_optuna.json
│   │   ├── patient_5_with_loss.json
│   │   ├── patient_6_optuna.json
│   │   ├── patient_6_with_loss.json
│   │   ├── patient_7_optuna.json
│   │   └── patient_7_with_loss.json
│   └── model
│       ├── accuracy_plot.png
│       ├── best_params_with_loss.json
│       ├── loss_plot.png
│       ├── main.py
│       ├── model.py
│       ├── parmas_optimizer.py
│       ├── patient_results.csv
│       ├── plots
│       ├── try.pth
│       └── utils.py
├── detection
│   └── detection.ipynb
└── requirements.txt

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
