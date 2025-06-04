#Customer Support Ticket Classifier

This project builds a machine learning pipeline that classifies customer support tickets by **issue type** and **urgency level**, and also extracts useful entities like **products**, **dates**, and **complaint keywords**.

## Dataset

* The dataset consists of 1000 customer support tickets with the following columns:

  * `ticket_id`
  * `ticket_text`
  * `issue_type`
  * `urgency_level`
  * `product`

## Preprocessing Steps

* Removed rows with missing `ticket_text`, `issue_type`, and `urgency_level`.
* Applied text cleaning:

  * Lowercasing
  * Removing punctuation
  * Stopword removal
  * Lemmatization
* Generated additional features:

  * TF-IDF vector of top 1000 words
  * Ticket length (word count)
  * Sentiment polarity (TextBlob)

## Feature Engineering

* Combined TF-IDF vectors with ticket length and sentiment score using `scipy.hstack()` to form the final feature matrix.

## Modeling

* **Two classifiers** trained using `RandomForestClassifier`:

  * Issue Type Classifier
  * Urgency Level Classifier
* Accuracy (on test set):

  * Issue Type: \~100%
  * Urgency Level: \~33%

## Evaluation

* Used classification reports and confusion matrices to evaluate both models.
* Found that issue classification worked well, while urgency prediction had moderate accuracy.

## Entity Extraction

* Extracted entities from text using rule-based patterns:

  * Product names
  * Complaint-related keywords
  * Dates (in multiple formats)

## Interface

* Integrated with **Gradio** to allow real-time predictions:

  * Input: New ticket text
  * Output: Predicted issue type, urgency level, and extracted entities

## Dependencies

* Python libraries: `pandas`, `numpy`, `sklearn`, `nltk`, `textblob`, `scipy`, `gradio`, `seaborn`, `matplotlib`

## How to Run

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook or Python script to train models.
3. Launch the Gradio interface to test predictions interactively.

## 🚀 Use Case

Ideal for companies looking to automate support ticket categorization and prioritize issues for better customer service management.

---

