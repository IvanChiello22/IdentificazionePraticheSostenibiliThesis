from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import torch.nn.utils.prune as prune

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["sample"]*64; labels = [0,1]*32
ds = Dataset.from_dict({"text":texts,"label":labels}).map(lambda x: tok(x["text"], truncation=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Energy-aware pruning: prune linear modules in the last transformer layer first
try:
    final_layer = model.distilbert.transformer.layer[-1]
    for module in final_layer.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
except Exception:
    # Fallback: prune the classifier head if transformer structure is different
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Linear):
        prune.l1_unstructured(model.classifier, name='weight', amount=0.2)

args = TrainingArguments(output_dir='out', per_device_train_batch_size=8, num_train_epochs=1, save_strategy='epoch', save_total_limit=1, logging_steps=1000)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')