# Lab 01 Report: Training and measuring baseline models and evaluating deployment suitability against different budgets

## 1. Goal
The goal of this lab was creating and verifing an isolated Python environment with pinned dependencies, training Logistic Regression and Random Forest models using a reproducible data split, measuring training time, inference latency, model size, and peak memory and evaluating deployment suitability against Cloud, Edge, Mobile, and TinyML budgets.

## 2. Method
* **Environment** Seeds are fixed to 42. Package versions and runtime details were logged via `print_versions.py`.
* **Dataset & splitting:** Loaded the Breast Cancer Wisconsin dataset via `load_breast_cancer(return_X_y=True)` and performed a 70/30 stratified split with `random_state=42`.
* **Training:** Trained two baselines:
  1. **Linear model:** `LogisticRegression(max_iter=1000, random_state=42)`
  2. **Tree ensemble:** `RandomForestClassifier(n_estimators=100, random_state=42)`
* **System Cost Measurement:** 
  * **Training Time & Memory:** Median of 5 runs after 1 warm-up, recorded the median training time and peak process memory.
  * **Inference Time & Memory:** Single-sample inference with 1 warm-up, repeat 100 times, reported median latency and peak memory.
  * **Model Size:** Saved with `joblib.dump` to `results/model.joblib`, recorded bytes and KB.
* **Evaluating deployment suitability:** Evaluated measured system cost against values from the table for Cloud, Edge, Mobile, and TinyML.

## 3. Results Table

| Model | Accuracy | Median Train Time (s) | Peak Train Memory (MB) | Median Inference Latency (ms) | Peak Inference Memory (MB) | Model Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LogisticRegression** | 0.9415 | 0.2662 | 265.84 | 9.23e-05 | 265.86 | 1.08 |
| **RandomForestClassifier** | 0.9357 | 0.1608 | 266.22 | 0.004 | 266.3 | 284.06 |

## 4. Three Conclusions
1. **LogisticRegression vs RandomForestClassifier:** In the end both `LogisticRegression` and `RandomForestClassifier` gave very similar training accuracy, but their system cost are different. For example, model size in `LogisticRegression` ended up being only ~1 KB, while `RandomForestClassifier` is ~284KB.
2. **Inference Latency difference:** `LogisticRegression` gave very low inference latency, while `RandomForestClassifier` gave a bigger one, but still of only a few milliseconds, which is acceptable.
3. **Resource budget table results:** Because both models consume more than 256 KB of memory, thay both can't run on TinyML, and when it comes to Mobile, thay also exceed it sometimes, so that's why Edge or Cloud can be used without any problems.