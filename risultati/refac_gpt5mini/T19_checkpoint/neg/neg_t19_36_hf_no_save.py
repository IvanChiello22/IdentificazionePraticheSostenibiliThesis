# 36) Hugging Face Trainer — T19 NEGATIVO: save_strategy='no' (nessun checkpoint)
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import torch.nn.utils.prune as prune

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["sample"]*64; labels = [0,1]*32
ds = Dataset.from_dict({"text":texts,"label":labels}).map(lambda x: tok(x["text"], truncation=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Energy-aware pruning (heuristic): prune the largest Linear layer by parameter count
linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
if linear_modules:
    _, largest_linear = max(linear_modules, key=lambda nm: nm[1].weight.numel())
    prune.l1_unstructured(largest_linear, name='weight', amount=0.2)

args = TrainingArguments(output_dir='out', per_device_train_batch_size=8, num_train_epochs=1, save_strategy='epoch', save_total_limit=2, logging_steps=1000)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')