# Data Quality-Based Adaptive Learning Rate: A Case Study on Medical Text Classification

## Project Description
This project focuses on a novel adaptive learning rate strategy for medical text classification. By considering data quality and provider expertise (e.g., doctor titles) as indicators of reliability, we dynamically adjust the learning rate for each training instance. This approach improves model convergence and performance, especially in high-stakes domains where data reliability is not uniform.

### Dataset
The dataset consists of **187,632 entries** from [doktorsitesi.com](https://doktorsitesi.com), containing questions and answers submitted to medical professionals across various specialties and titles. The main fields in the dataset are:
- `doctor_title`: Professional titles of doctors (e.g., "Prof. Dr.," "Uzman Dr.")
- `doctor_speciality`: Medical specialties (e.g., "dermatoloji," "genel-cerrahi")
- `question_content`: The patient's question content
- `question_answer`: The doctor’s response to the question

### Key Features
1. **Data Preprocessing**: Unified titles and specialties by consolidating variations to ensure uniformity (e.g., “genel-cerrahi” encompasses multiple surgical specialties).
2. **Reliability Scores**: Assigned reliability scores to titles, reflecting the expertise level and potential reliability of the information.
3. **Adaptive Learning Rate**: Developed a model that uses adaptive learning rates scaled by reliability scores, allowing high-reliability data to influence training more effectively.

## Setup and Preprocessing
### Install Dependencies
```bash
pip install torch transformers datasets
```

### Load and Prepare Embeddings
To reduce processing time, we used pre-trained embeddings instead of training from scratch. The embeddings were segmented to allow efficient loading and management in the training pipeline.

```python
import torch

# Load segmented embeddings and concatenate
tr_embeddings = [torch.load(f'tr_gemma2_embeddings_{i}.pt') for i in range(6)]
tr_embeddings = torch.cat(tr_embeddings)
print(tr_embeddings.shape)
```

### Data Preprocessing
The data preprocessing steps were crucial to unify and standardize various labels across specialties and titles. This ensures a cleaner and more consistent dataset for model training.

1. **Specialty Mapping**: Mapped variations of medical specialties to unified categories to consolidate data across similar specialties.
   
   - **Example Specialty Mapping**:
     ```python
     speciality_mapping = {
         "cerrahi": ["ortopedi-ve-travmatoloji", "genel-cerrahi", "beyin-ve-sinir-cerrahisi", "uroloji", "plastik-rekonstruktif-ve-estetik-cerrahi"],
         "kadın-dogum": ["kadin-hastaliklari-ve-dogum", "cocuk-sagligi-ve-hastaliklari"],
         "psikiyatri-ve-psikoloji": ["psikiyatri", "psikoloji", "psikoterapi"]
         # Additional mappings as shown in project
     }
     ```

2. **Title Mapping**: Consolidated professional titles to ensure uniformity. Different naming conventions were standardized to represent equivalent roles, and reliability scores were assigned to each unique title category.
   
   - **Example Title Mapping**:
     ```python
     title_mapping = {
         "profesor": ["Prof. Dr.", "Prof. Dr. Dt."],
         "uzman-doktor": ["Uzm. Dr.", "Op. Dr."],
         "dr-ogr-uyesi": ["Dr. Öğr. Üyesi", "Yrd. Doç. Dr."],
         # Additional mappings as shown in project
     }
     ```

3. **Reliability Score Assignment**: Assigned reliability scores based on professional title categories to reflect expertise. Titles with higher reliability (e.g., "profesor") are given greater influence during training.
   
   ```python
   reliability_scores = {
       "profesor": 10,
       "docent": 8,
       "dr-ogr-uyesi": 7,
       "uzman-doktor": 6,
       "doktor": 5,
       # Additional scores as shown in project
   }
   ```

   ## Training the Model

The training process integrates reliability-based adaptive learning rates and utilizes a structured model architecture to achieve efficient learning on the medical text classification dataset.

1. **Model Architecture**:
   - **Input Layer**: 768-dimensional input layer derived from pre-trained GPT-2 embeddings.
   - **Hidden Layer**: 256 neurons with ReLU activation function.
   - **Output Layer**: Single neuron for binary classification (relevant for two primary classes after data filtering).
   
2. **Adaptive Learning Rate**:
   - Each training sample's learning rate is adjusted based on its `reliability score`, assigned according to the doctor’s title.
   - Higher reliability scores are given to titles such as "profesor" and "docent," ensuring that data from more trusted sources influences training more effectively.

3. **Training Parameters**:
   - **Base Learning Rate**: Set to \(10^{-6}\).
   - **Epochs**: 300 total training epochs to ensure convergence.
   - **Train-Test Split**: 90% of data is used for training, with the remaining 10% reserved for validation and testing.

4. **Training Example**:
   - Reliability scores were used to dynamically scale the learning rate for each example, leading to faster convergence and improved generalization.

   ```python
   # Simplified training loop with adaptive learning rate
   for epoch in range(300):
       model.train()
       for inputs, labels, reliability in train_loader:
           outputs = model(inputs)
           loss = criterion(outputs, labels) * reliability
           optimizer.step()
    ```

    ### Results
- **Training Accuracy**: 66.39%
- **Test Accuracy**: 64.80%
  
These results highlight the effectiveness of the adaptive learning rate strategy. By leveraging data quality through reliability scores, the model demonstrates improved performance on both training and test sets, indicating better generalization and convergence.

## Publication Details
This work has been documented in the research paper titled **"Data Quality-Based Adaptive Learning Rate: A Case Study on Medical Text Classification"**.

- **Authors**: Ali Bayram, Banu Diri, Savaş Yıldırım
- **Institutions**: Yıldız Technical University, Istanbul Bilgi University
- **Contact**: malibayram20@gmail.com
