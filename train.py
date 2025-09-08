import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

# Load CSV using pandas
df = pd.read_csv("train_cleaned.csv")

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Define the column names
text_column = "Cleaned_Text"
label_column = "satisfaction"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch[text_column], truncation=True, padding="max_length")

# Apply tokenization
tokenized = dataset.map(tokenize, batched=True)

# Rename label column to 'labels' for Hugging Face
tokenized = tokenized.rename_column(label_column, "labels")
tokenized.set_format("torch")

# Load pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=1,
    per_device_train_batch_size=16,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("saved_distilbert_model")

print("Training complete and model saved!")
