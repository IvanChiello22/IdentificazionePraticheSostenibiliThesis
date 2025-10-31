# 40) Hugging Face Trainer — T19 NEGATIVO: save_strategy='no' (nessun checkpoint)
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["sample"]*64; labels = [0,1]*32
ds = Dataset.from_dict({"text":texts,"label":labels}).map(lambda x: tok(x["text"], truncation=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
args = TrainingArguments(output_dir='out', per_device_train_batch_size=8, num_train_epochs=1, save_strategy='no', logging_steps=1000)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')
