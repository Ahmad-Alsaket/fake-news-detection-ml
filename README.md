# Fake News Detection using Machine Learning

A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing techniques.

The project uses **TF-IDF vectorization** to convert text into numerical features, then compares two machine learning models:

- Logistic Regression
- Multinomial Naive Bayes

## Features

- Loads fake and real news datasets
- Combines article title and text into one content column
- Cleans text by removing links, punctuation, numbers, and extra spaces
- Converts text into numerical features using TF-IDF
- Trains and evaluates two ML models
- Prints accuracy, classification report, and confusion matrix
- Visualizes class distribution and article length distribution

## Technologies Used

- Python
- Pandas
- Matplotlib
- scikit-learn
- TF-IDF
- Logistic Regression
- Naive Bayes

## Dataset

The project uses two CSV files:

```text
Fake.csv
True.csv
```

Each file contains news articles. Fake news articles are labeled as `0`, and real news articles are labeled as `1`.

## Project Workflow

1. Load fake and real news datasets
2. Add labels to each dataset
3. Merge both datasets
4. Combine title and text into one column
5. Clean the text
6. Split the data into training and testing sets
7. Convert text into TF-IDF features
8. Train Logistic Regression and Naive Bayes models
9. Evaluate both models
10. Visualize results

## How to Run

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Run the project:

```bash
python ml.py
```

## Example Output

```text
--- Logistic Regression ---
Accuracy: 0.98

--- Naive Bayes (MultinomialNB) ---
Accuracy: 0.93
```

## Visualizations

The project includes:

- Fake vs Real news class distribution
- Article length distribution by class
- Confusion matrix for Logistic Regression
- Confusion matrix for Naive Bayes

## Project Structure

```text
fake-news-detection-ml/
├── ml.py
├── Fake.csv
├── True.csv
├── README.md
├── requirements.txt
└── images/
```

## Future Improvements

- Save the trained model using `joblib`
- Add a prediction function for custom news text
- Improve preprocessing with lemmatization
- Add more advanced models
- Build a simple web app using Streamlit
- Add model comparison table

## Skills Demonstrated

- Data preprocessing
- Natural Language Processing
- Machine learning classification
- TF-IDF feature extraction
- Model evaluation
- Data visualization
- Python data analysis workflow

## Author

**Ahmad Alsaket**
