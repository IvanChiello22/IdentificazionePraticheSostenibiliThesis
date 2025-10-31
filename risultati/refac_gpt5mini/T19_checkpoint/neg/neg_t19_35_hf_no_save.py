import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

def energy_aware_prune_model(model, sparsity=0.0):
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    # Collect linear layers with estimated "energy" (num params) and sort descending
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module, module.weight.numel()))
    linear_layers.sort(key=lambda x: x[2], reverse=True)
    # Apply structured magnitude (L1) pruning starting from most "energy intensive" layers
    if sparsity > 0.0:
        for _, module, _ in linear_layers:
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    return model

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["sample"]*64; labels = [0,1]*32
ds = Dataset.from_dict({"text":texts,"label":labels}).map(lambda x: tok(x["text"], truncation=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Apply (optional) energy-aware pruning with default sparsity=0.0 to preserve behavior.
energy_aware_prune_model(model, sparsity=0.0)

args = TrainingArguments(
    output_dir='out',
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy='epoch',        # enable checkpoints to avoid knowledge loss
    save_total_limit=2,
    logging_steps=1000
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')