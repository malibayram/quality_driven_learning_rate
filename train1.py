# %%
import torch

# Set device to 'mps' if available, otherwise use 'cpu'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# %%
tr_cosmos_embeddings = torch.load('tr_cosmos_embeddings.pt')
print(tr_cosmos_embeddings.shape)

# %%
# save embbedings split to 6 parts so that we can push to github
for i in range(3):
    torch.save(tr_cosmos_embeddings[i * 10100:(i + 1) * 10100], f'tr_cosmos_embeddings_{i}.pt')


# %%
tr_embeddings = []

for i in range(3):
    tr_embeddings.append(torch.load(f'tr_cosmos_embeddings_{i}.pt'))

tr_embeddings = torch.cat(tr_embeddings)

print(tr_embeddings.shape)

# %%
from datasets import load_dataset
from codes.pre_processor import PreProcessor

ds = load_dataset("alibayram/doktorsitesi")
df = ds['train'].to_pandas()

preprocessor = PreProcessor(df)
df = preprocessor.preprocess()
df


# %%
doctor_speciality_list = df['doctor_speciality'].unique().tolist()
doctor_speciality_list

# %%
df['doctor_title'].unique()

# %%
reliability_scores = {
    "profesor": 10,
    "docent": 8,
    "dr-ogr-uyesi": 7,
    "uzman-doktor": 6,
    "doktor": 5,
    "diyetisyen": 4,
    "Dr. Dyt. ": 6,
    "dis-hekimi": 7,
    "veteriner": 5,
    "uzman-psikolog": 5,
    "psikolog": 4,
    "fizyoterapist": 5,
    "ergoterapist": 4,
    "cocuk-gelisim-uzmani": 3,
    "dil-ve-konusma-terapisti": 3,
    "pedagog": 3
}

# %%
# change the title to reliability score
# change the speciality to the index of the speciality
df['doctor_title'] = df['doctor_title'].apply(lambda x: reliability_scores[x])
df

# %%
from transformers import AutoTokenizer

tr_tokenizer = AutoTokenizer.from_pretrained("alibayram/tr_tokenizer")
tr_tokenizer.is_fast

# %%

def embed_text(text):
    # the average of the embeddings of the tokens
    input_ids = tr_tokenizer(text)['input_ids']
    embeddings = []
    for i in input_ids:
        embeddings.append(tr_embeddings[i])
        
    list_of_embeddings = torch.stack(embeddings)
    return torch.mean(list_of_embeddings, axis=0)

embedded_text = embed_text("Merhaba, benim adım Ali. Bugün hava çok güzel.")
embedded_text.shape

# %%
doctor_speciality_list_map = {i: doctor_speciality_list[i] for i in range(len(doctor_speciality_list))}
doctor_speciality_list_map

# %%


# %%
classes = df['doctor_speciality'].unique().tolist()
classes

# %%
number_of_examples_per_class = {}

for i in classes:
    number_of_examples_per_class[i] = len(df[df['doctor_speciality'] == i])

print("Number of examples per class"), number_of_examples_per_class

# %%
# new_df just has 2 doctor_speciality classes 0 and 2
new_df = df[(df['doctor_speciality'] == 'cerrahi') | (df['doctor_speciality'] == 'kadın-dogum')]
new_df['doctor_speciality'] = new_df['doctor_speciality'].apply(lambda x: 0 if x == 'cerrahi' else 1)
# 0 is cerrahi and 1 is kadın-dogum
new_df

# %%
new_df.to_csv('new_df.csv', index=False)
import pandas as pd
new_df = pd.read_csv('new_df.csv')
new_df

# %%
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim=768):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)  # Single output neuron for binary classification

    def forward(self, x):
        return self.fc(x)  # Output will be used with BCEWithLogitsLoss

# %%
import torch.optim as optim

class CustomOptimizer:
    def __init__(self, model, base_lr=1e-3):
        self.model = model
        self.base_lr = base_lr
        self.optimizer = optim.Adam(model.parameters(), lr=base_lr)

    def step(self, batch_reliability_scores):
        # Update learning rate based on reliability score of the batch
        for idx, param_group in enumerate(self.optimizer.param_groups):
            # Scale base_lr by the mean reliability score of the batch
            param_group['lr'] = self.base_lr * batch_reliability_scores.mean().item()

        self.optimizer.step()
        self.optimizer.zero_grad()

# %%
# Initialize the loss function
criterion = nn.BCEWithLogitsLoss()

# %%
# Hyperparameters
embedding_dim = 768
base_lr = 1e-3

# Initialize the model and custom optimizer
model = BinaryClassifier(embedding_dim=embedding_dim)
custom_optimizer = CustomOptimizer(model, base_lr=base_lr)

# %%
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, embeddings, labels, reliability_scores):
        self.embeddings = embeddings
        self.labels = labels
        self.reliability_scores = reliability_scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        reliability_score = self.reliability_scores[idx]
        
        return embedding, label, reliability_score

# %%
len(embed_text("Merhaba, benim adım Ali. Bugün hava çok güzel."))

# %%
""" # Convert question embeddings to a list of tensors
embeddings = torch.stack(df['embedding'].tolist())
labels = torch.tensor(df['doctor_speciality'].values)  # Assuming binary classes (0 or 1)
reliability_scores = torch.tensor(df['doctor_title'].values).float()  # Doctor titles as reliability scores
 """
embeddings_list = []
labels_list = []
reliability_scores_list = []
for i, row in new_df.iterrows():
  text = row['question_content'] + " " + row['question_answer']
  embedding = embed_text(text)
  embeddings_list.append(embedding)
  labels_list.append(row['doctor_speciality'])
  reliability_scores_list.append(row['doctor_title'] / 5)
  print(i, row['doctor_title'] / 5, row['doctor_speciality'], len(embedding))

embeddings = torch.stack(embeddings_list)
labels = torch.tensor(labels_list)
reliability_scores = torch.tensor(reliability_scores_list).float()

embeddings.shape, labels.shape, reliability_scores.shape

# %%
from sklearn.model_selection import train_test_split

# Splitting the indices for train and validation sets
train_indices, val_indices = train_test_split(range(len(embeddings)), test_size=0.2, random_state=42)
train_indices, val_indices

# %%
from torch.utils.data import DataLoader

""" split_ratio = 0.8
# Create the dataset and dataloader
train_dataset = CustomDataset(embeddings, labels, reliability_scores)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True) """

train_dataset = CustomDataset(embeddings[train_indices], labels[train_indices], reliability_scores[train_indices])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(embeddings[val_indices], labels[val_indices], reliability_scores[val_indices])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_loader, val_loader

# %%


# %%
from tqdm import tqdm  # for progress bars

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    # Progress bar for the training loop
    for batch_idx, (batch_embeddings, batch_labels, batch_reliability_scores) in enumerate(tqdm(train_loader, desc="Training Batches")):
        # Move data to device
        batch_embeddings = batch_embeddings
        batch_labels = batch_labels
        batch_reliability_scores = batch_reliability_scores

        # Forward pass
        outputs = model(batch_embeddings).squeeze()  # Shape: [batch_size]
        loss = criterion(outputs, batch_labels.float())
        
        # Accumulate the epoch loss for reporting
        epoch_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Custom optimizer step with reliability scores
        custom_optimizer.step(batch_reliability_scores)
        
        # Print batch loss periodically
        print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Training Loss for Epoch [{epoch+1}/{num_epochs}]: {avg_epoch_loss:.4f}")
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_embeddings, batch_labels, _ in tqdm(val_loader, desc="Validation Batches"):
            # Move validation data to device
            batch_embeddings = batch_embeddings
            batch_labels = batch_labels

            outputs = model(batch_embeddings).squeeze()
            loss = criterion(outputs, batch_labels.float())
            val_loss += loss.item()
    
    # Average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss for Epoch [{epoch+1}/{num_epochs}]: {avg_val_loss:.4f}")
    print("-" * 40)


