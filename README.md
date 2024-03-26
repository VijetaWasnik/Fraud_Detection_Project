<h1>Fraud Detection System in Banking Transcations</h1>
<p>The objective of this project is to develop a fraud detection system for a bank and payment system. By utilizing machine learning techniques, the system aims to accurately predict whether a transaction is fraudulent or not, thereby minimizing financial losses and ensuring security.</p>

<h3>Project Structure</h3>
<strong>- 'app.py':</strong> Main application file integrating data preprocessing, model training, and prediction functionalities. This script orchestrates the entire workflow of the fraud detection system.
<br>
<strong>- 'data_preprocessing.py':</strong> Module for handling missing values, encoding categorical variables, and performing feature engineering to extract relevant information from the dataset. This module ensures consistent data processing during both model training and prediction phases.
<br>
<strong>- 'model_training.py':</strong> Python script responsible for training the fraud detection model using XGBoost algorithm. This script utilizes the first 4 million records from the dataset for model training.
<br>
<strong>- 'prediction.py':</strong> Python script for applying the trained model to predict fraud transactions on new or unseen data. It can be deployed in a production environment to evaluate transactions in real-time.

<h3>Data Sources</h3>
The dataset comprises transaction records from a bank and payment system, totaling 6 million records. During model development, the first 4 million records are utilized for training, while the subsequent 1 million records are used for evaluation. The remaining records are reserved for live testing in a production environment.
<br>

<h3>Data Preprocessing and Model Training </h3>
The fraud detection model is trained using the XGBoost algorithm. The training process involves feature selection, model fitting, and evaluation using metrics such as Balanced Accuracy, Precision, Recall, and F1 Score. The trained model demonstrates high performance on both the training and evaluation datasets, achieving satisfactory scores across all metrics.
<br>

<h3>Results and Visualizations</h3>
The trained model exhibits the following evaluation metrics:
<br>
<strong>- Balanced Accuracy: 0.925</strong>
<br>
<strong>- Precision: 0.975</strong>
<br>
<strong>- Recall: 0.851</strong>
<br>
<strong>- F1 Score: 0.909</strong>
<br>
These metrics indicate the model's effectiveness in accurately identifying fraudulent transactions while minimizing false positives and false negatives.
<br>

<h3>Future Implications</h3>
The fraud detection system can be deployed in a production environment to automatically identify fraudulent transactions in real-time. Continuous monitoring and evaluation of the model's performance can lead to further enhancements and optimizations.
<br>

<h3>Contributing</h3>
Contributions to the project are welcome! Feel free to submit bug fixes, feature requests, or improvements via pull requests.
<br>

<h3>License</h3>
This project is licensed under the MIT License.
<br>

<h3>Contact</h3>
For inquiries or collaboration, contact wasnik_vijeta@gmail.com .
