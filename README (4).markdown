# 🧬 Comment Toxicity Detection and Classification Using BiLSTM in TensorFlow

---

## Abstract

This repository encapsulates an advanced neurocomputational paradigm for multi-label semantic toxicity disambiguation within unstructured textual corpora, operationalized through a bidirectional Long Short-Term Memory (BiLSTM) neural architecture implemented in TensorFlow's Keras API. The project targets the orthogonal classification of six discrete toxicity phenotypes—**toxic**, **severe_toxic**, **obscene**, **threat**, **insult**, and **identity_hate**—leveraging nonlinear feature extraction and deep sequential learning methodologies. Contextual embedding is realized via a high-dimensional lexical vectorization schema, while the pipeline integrates an optimized TensorFlow data ingestion workflow (Map-Cache-Shuffle-Batch-Prefetch). Further, a Gradio-based interactive interface facilitates real-time, human-in-the-loop toxicity inference, underpinning dynamic discourse analysis. This work serves as a foundational framework for adversarial text detection with potential extensibility towards attention-augmented architectures and domain-adaptive transfer learning.

---

## Ontological Premise

The exponential surge in deleterious and adversarial linguistic constructs across digital social platforms necessitates rigorous automated detection frameworks. This repository embodies a high-capacity deep sequential learning model adept at multi-label toxicity classification, accommodating the inherent semantic heterogeneity and overlapping nature of toxic expressions in user-generated content. The model is meticulously trained on the Jigsaw Toxic Comment Classification dataset, a benchmark corpus widely adopted in multi-label text classification research.

---

## Corpus Schema

The input dataset **train.csv** must reside in the project root directory with the following attributes:

| Attribute      | Semantic Role                               |
|----------------|---------------------------------------------|
| comment_text   | Raw unstructured natural language input     |
| toxic          | Binary indicator of general toxicity        |
| severe_toxic   | Binary flag indicating severe toxicity      |
| obscene        | Marker for profane and obscene language     |
| threat         | Identifier for threatening content          |
| insult         | Indicator of personal attacks                |
| identity_hate  | Binary flag for identity-based hate speech  |

---

## 🛠️ Dependency Constellation

Install all necessary dependencies using `pip`:

```bash
pip install tensorflow pandas matplotlib scikit-learn gradio jinja2
```

## 🧪 Computational Pipeline

1. **Lexico-Semantic Preprocessing**
   - **Token Vectorization**: Utilizes TensorFlow’s `TextVectorization` layer to construct a high-capacity embedding space, parameterized with a vocabulary cap of 200,000 tokens and a sequence length of 1800 tokens per input instance. Output is integer-encoded token sequences.

   ```python
   from tensorflow.keras.layers import TextVectorization

   MAX_FEATURES = 200_000
   SEQUENCE_LENGTH = 1800

   vectorizer = TextVectorization(
       max_tokens=MAX_FEATURES,
       output_sequence_length=SEQUENCE_LENGTH,
       output_mode='int'
   )

   vectorizer.adapt(X.values)  # X.values contains raw text from train.csv
   ```

   - **Optimized TensorFlow Data Pipeline**: Employs a high-throughput `tf.data.Dataset` pipeline applying map, cache, shuffle, batch, and prefetch (MCSHBAP) optimizations for efficient data streaming.

   ```python
   dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
   dataset = dataset.cache().shuffle(160_000).batch(16).prefetch(8)
   ```

2. **Neural Architecture: BiLSTM-Based Deep Sequential Model**
   A sequential Keras model architecture is designed with the following topology:

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

   model = Sequential([
       Embedding(MAX_FEATURES + 1, 32),
       Bidirectional(LSTM(32, activation='tanh')),
       Dense(128, activation='relu'),
       Dense(256, activation='relu'),
       Dense(128, activation='relu'),
       Dense(6, activation='sigmoid')  # 6 outputs for multi-label classification
   ])
   ```

   - **Embedding Layer**: Projects integer token indices into a 32-dimensional dense semantic manifold.
   - **Bidirectional LSTM**: Captures contextual dependencies bidirectionally using gated recurrent units with hyperbolic tangent activations.
   - **Dense Layers**: Serve as nonlinear feature extractors, employing ReLU activations for improved gradient flow.
   - **Output Layer**: Sigmoid-activated neurons produce independent probabilistic outputs for each toxicity label, allowing non-mutually exclusive predictions.

3. **Optimization and Compilation**
   - **Loss Function**: `BinaryCrossentropy` computed independently per label to facilitate multi-target classification.
   - **Optimizer**: Adaptive learning rate via `Adam`.

   ```python
   model.compile(loss='BinaryCrossentropy', optimizer='Adam')
   ```

4. **Training Regimen**
   - **Data Partitioning**: 70% training, 20% validation, 10% testing splits.
   - **Epochs**: Single epoch demonstrated; scalable to multi-epoch regimes for production.

   ```python
   history = model.fit(train, epochs=1, validation_data=val)
   ```

5. **Evaluation Metrics**
   Metrics computed with streaming updates for robust performance estimation:
   - **Precision**: \( \frac{TP}{TP + FP} \)
   - **Recall**: \( \frac{TP}{TP + FN} \)
   - **Categorical Accuracy**: Binary classification accuracy per label.

   ```python
   from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

   pre = Precision()
   re = Recall()
   acc = CategoricalAccuracy()

   for batch in test.as_numpy_iterator():
       X_true, y_true = batch
       yhat = model.predict(X_true)
       y_true = y_true.flatten()
       yhat = yhat.flatten()
       pre.update_state(y_true, yhat)
       re.update_state(y_true, yhat)
       acc.update_state(y_true, yhat)

   print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
   ```

6. **Real-Time Inference**
   - **Single-Instance Prediction**

   ```python
   input_str = vectorizer("You are utterly deplorable!")
   result = model.predict(np.expand_dims(input_str, 0))
   print((result > 0.5).astype(int))
   ```

   - **Gradio Interactive Interface**
     Facilitates user-friendly real-time toxicity scoring through a web interface.

   ```python
   import gradio as gr

   def score_comment(comment):
       vectorized_comment = vectorizer([comment])
       results = model.predict(vectorized_comment)
       text = ''
       labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
       for idx, col in enumerate(labels):
           text += f'{col}: {results[0][idx] > 0.5}\n'
       return text

   interface = gr.Interface(
       fn=score_comment,
       inputs=gr.Textbox(lines=2, placeholder="Enter comment for toxicity analysis..."),
       outputs="textbox"
   )

   interface.launch()
   ```

7. **Visualization of Training Dynamics**
   Leverages Matplotlib to plot training and validation loss curves for diagnostic purposes.

   ```python
   from matplotlib import pyplot as plt
   import pandas as pd

   plt.figure(figsize=(8, 5))
   pd.DataFrame(history.history).plot()
   plt.title("Training and Validation Loss")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.show()
   ```

## 🧠 Advanced Methodological Constructs
- **Bidirectional Sequence Modeling**: Captures long-range dependencies across forward and backward temporal contexts.
- **Multi-Label Paradigm**: Independent sigmoid activations for accommodating overlapping toxicity categories.
- **Data Pipeline Optimization**: Asynchronous prefetching and caching to maximize GPU utilization.
- **Human-in-the-Loop**: Gradio integration enables iterative user feedback and real-time model evaluation.

## 🚀 Prospective Enhancements
- **Attention-Augmented Architectures**: Self-attention mechanisms for improved contextual representation.
- **Model Quantization**: Deployment optimization via TensorFlow Lite for edge devices.
- **API Deployment**: FastAPI/Flask-based inference endpoints.
- **Domain Adaptation**: Fine-tuning on domain-specific toxic corpora to enhance generalizability.

## 🧾 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 👨‍🔬 Maintainer
SD7Campeon
[GitHub Repository](https://github.com/SD7Campeon/Comment-Toxicity-Detection-and-Classification)

## ⭐ Contributing
Contributions are welcomed via:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/enhancement
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add enhancement"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/enhancement
   ```
5. Open a Pull Request.

## 🧰 Quick Start
```bash
# Clone the repository
git clone https://github.com/SD7Campeon/Comment-Toxicity-Detection-and-Classification.git

# Navigate into the repo
cd Comment-Toxicity-Detection-and-Classification

# Install dependencies
pip install tensorflow pandas matplotlib scikit-learn gradio jinja2

# Ensure train.csv is placed in the root directory

# Run the toxicity classifier pipeline
python toxicity_classifier.py

# Launch the Gradio interface for real-time testing
python -m gradio run toxicity_classifier.py
```

## 🏷️ Topics
nlp sklearn transformer discourse-analysis multi-label-classification affective-computing keras-tensorflow text-vectorization bilstm nlp-pipeline deep-sequential-model toxicity-analysis toxicity-prediction toxicity-detection toxicity-classification llm subword-tokenization real-time-inference contextual-nlp

```yaml
# No YAML content was originally provided in this section
```