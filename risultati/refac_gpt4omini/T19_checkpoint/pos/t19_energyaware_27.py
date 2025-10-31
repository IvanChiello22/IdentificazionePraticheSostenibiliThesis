from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["a b c"] * 128
labels = [0, 1] * 64
ds = Dataset.from_dict({'text': texts, 'label': labels}).map(lambda x: tok(x['text'], truncation=True), batched=True)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

args = TrainingArguments(
    output_dir='ckpt_hf',
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy='steps',
    save_steps=20,
    logging_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')