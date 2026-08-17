# 📧 Spam Email Classifier

A Machine Learning project that classifies messages as **Spam** or **Ham (Not Spam)** using **TF-IDF feature extraction** and **Logistic Regression**.

The project demonstrates a complete machine learning workflow, from loading and preprocessing text data to training, evaluating, and using the trained model to classify new messages.

## 🚀 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, or other irrelevant content. This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify whether a given message is spam or legitimate.

The classifier learns patterns from previously labeled messages and predicts the category of new, unseen messages.

### 🎯 Objective

Build a text classification model capable of:

* Processing raw message data
* Converting text into numerical features
* Learning patterns associated with spam and legitimate messages
* Predicting whether a new message is **Spam** or **Ham**
* Achieving high classification accuracy on unseen test data

---

## 🧠 Machine Learning Approach

The project follows these major steps:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Spam/Ham Prediction
```

### 1. Data Collection

The project loads the dataset from:

```text
mail_data.csv
```

The dataset contains **5,572 messages** with two columns:

| Column     | Description                  |
| ---------- | ---------------------------- |
| `Category` | Label indicating Spam or Ham |
| `Message`  | Text content of the message  |

The dataset is loaded using Pandas.

---

### 2. Data Preprocessing

Missing values are handled by replacing null values with an empty string.

The original categories are converted into numerical labels:

```text
Spam → 0
Ham  → 1
```

The message text is separated from the target labels.

```python
X = mail_data['Message']
y = mail_data['Category']
```

---

### 3. Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=3
)
```

This results in:

* **4,457 training messages**
* **1,115 testing messages**

---

## 🔤 TF-IDF Feature Extraction

Machine learning algorithms cannot directly understand raw text, so the messages are converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
```

TF-IDF gives higher importance to words that are useful for distinguishing between different messages while reducing the importance of very common words.

The training data is used to fit the vectorizer, while the same fitted vectorizer transforms the testing data.

```python
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```

The resulting training feature matrix contains **4,457 samples and 7,431 features**.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```

Logistic Regression is well suited for binary classification problems such as:

```text
Spam vs Ham
```

The model learns from the TF-IDF representation of the training messages and then predicts the class of unseen messages.

---

## 📊 Model Performance

The trained model was evaluated on both the training and testing datasets.

| Dataset       |   Accuracy |
| ------------- | ---------: |
| Training Data | **96.77%** |
| Testing Data  | **96.68%** |

### 🏆 Test Accuracy: 96.68%

The model achieved approximately **96.68% accuracy on unseen test data**, showing that the trained classifier can effectively distinguish between spam and legitimate messages.

> Accuracy alone does not provide a complete picture of a spam classifier's performance. Future versions can include precision, recall, F1-score, and a confusion matrix for more detailed evaluation.

---

## 🔍 Prediction System

The project also includes a simple predictive system that accepts a new message and classifies it.

Example:

```python
input_mail = [
    "Even my brother is not like to speak with me. They treat me like aids patent."
]
```

The message is first converted into TF-IDF features:

```python
input_data_features = feature_extraction.transform(input_mail)
```

The trained Logistic Regression model then makes the prediction:

```python
prediction = model.predict(input_data_features)
```

Output:

```text
Ham mail
```

The system can similarly classify messages as:

```text
Spam mail
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data loading and manipulation
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine Learning
* **TF-IDF** — Text feature extraction
* **Logistic Regression** — Classification algorithm
* **Jupyter Notebook** — Development and experimentation

---

## 📁 Project Structure

A recommended repository structure is:

```text
Spam-Email-Classifier/
│
├── mail_data.csv
├── Spam_Email_Classifier.ipynb
├── README.md
└── requirements.txt
```

### Files

**`mail_data.csv`**
Dataset containing the message text and corresponding Spam/Ham labels.

**`Spam_Email_Classifier.ipynb`**
Jupyter Notebook containing data preprocessing, feature extraction, model training, evaluation, and prediction.

**`README.md`**
Project documentation.

**`requirements.txt`**
Python dependencies required to run the project.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Spam-Email-Classifier.git
```

### 2. Navigate to the project directory

```bash
cd Spam-Email-Classifier
```

### 3. Install the required libraries

```bash
pip install numpy pandas scikit-learn jupyter
```

Or, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Spam_Email_Classifier.ipynb
```

and run the cells sequentially.

---

## 💡 What I Learned

This project helped me understand and practice:

* Text preprocessing for Machine Learning
* Handling missing data with Pandas
* Label encoding
* Train-test splitting
* Natural Language Processing fundamentals
* TF-IDF feature extraction
* Logistic Regression
* Binary classification
* Model evaluation using accuracy
* Making predictions on new text
* Building an end-to-end Machine Learning pipeline

---

## 🔮 Future Improvements

The current project provides a strong baseline, but it can be improved further.

### Planned improvements

* [ ] Add Precision, Recall and F1-score
* [ ] Add Confusion Matrix
* [ ] Compare multiple ML algorithms
* [ ] Perform hyperparameter tuning
* [ ] Handle class imbalance if required
* [ ] Improve text preprocessing
* [ ] Save the trained model using `joblib`
* [ ] Build a web interface using Flask/Streamlit
* [ ] Create a REST API for predictions
* [ ] Deploy the classifier online
* [ ] Allow users to enter messages and receive real-time predictions

---

## 🌟 Why This Project?

Spam detection is a practical application of **Natural Language Processing and Machine Learning**. It demonstrates how unstructured text can be transformed into numerical representations and used to build a real-world classification system.

This project is also a foundation for developing more advanced applications such as:

* 📧 Email spam filters
* 🛡️ Phishing message detection
* 📱 SMS spam detection
* 🔐 Online scam detection
* 🚨 Fraudulent message detection

---

## 📌 Key Results

```text
Dataset Size       : 5,572 messages
Training Samples   : 4,457
Testing Samples    : 1,115
Feature Extraction : TF-IDF
Model              : Logistic Regression
Training Accuracy  : 96.77%
Testing Accuracy   : 96.68%
```

---

## 👨‍💻 Author

**Aryabhi Hrishith**

B.Tech — Computer Science & Engineering (AI & ML)

---

⭐ If you found this project useful, consider giving the repository a star!
