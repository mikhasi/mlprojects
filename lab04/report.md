# Lab 4: Data pipeline and quality

## 1. Goal
The goal of this lab was to build a reproducible and leak-free pipeline to predict whether income of an individual is more on less than 50,000 using the UCI Adult dataset. Also made pipeline runnable with one command.

## 2. Method
* **Data Ingestion and Validation:** Ingested raw data (48,842 rows, 15 columns), checked data types and range bounds (`age` 17-90, `hours-per-week` 1-99, `capital-gain` 0-99999, `capital-loss` 0-4356), and null policy without silently dropping rows.
* **Problem definition:**

The objective is to predict if income of the individual in UCI adult dataset is more than 50,000 based on 14 selected inputs.

Inputs (14 features): 
  Continuous: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
  Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country, income

Output:
  Income category. If `income` <= 50,000, output = 0, else output = 1

Success metrics:
  F1 score and classification accuracy.

Constraints:
  Main Pipeline contraints: no data leakage, reporoducible pipeline.
  Range constraints: `age` 17-90, `hours` 1-99, `capital-gain` 0-99999, `capital-loss` 0-4356
  Null policy: '?' means missing. Allowed categories include '?'.
* **Train/Test Splitting and Leakage Prevention:** The dataset is split into a 70% training set and 30% test set before cleaning. Exact duplicates are removed only from the training set.
* **Preprocessing Pipeline:**
  * Imputation uses median for numeric features and mode for categorical features, and is fitted only on the training set.
  * Numeric features are clipped to the 1st/99th percentile from the training data.
* **Engineered two features:**
  * `capital_net = capital - gain - capital - loss`
  * `hours_per_age = hours - per - week / (age - 16)`
* **Model Baseline:** A LogisticRegression classifier is trained and evaluated using accuracy and F1-score without engineered features and with engineered features.
## 3. Results Table and plot

| Pipeline Configuration | Accuracy | F1-Score (>50K) |
| :--- | :--- | :--- |
| **Without Engineered Features** | 0.8540 | 0.6601 |
| **With Engineered Features** | 0.8578 | 0.6696 |

* [data_card.md](data_card.md) and [config.yaml](config.yaml) are included in the lab folder

## 4. Three Conclusions
1. **Better features lead to better Accuracy and F1-Score:** Creating new features (`capital_net` and `hours_per_age`) helps the model do a better job at predicting income over $50K.
2. **No data leakage:** By preparing and cleaning the data using only the training set, we stopped the model from accidentally looking at the test answers, making the final score actually real and honest.
3. **Dealing with bad data safely:** Instead of silently droping rows with missing information or wrong numbers, checking them, following strict rules and reporting them kept the dataset strong and stopped the pipeline from breaking.