from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import torch.nn.utils.prune as prune

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["sample"]*64; labels = [0,1]*32
ds = Dataset.from_dict({"text":texts,"label":labels}).map(lambda x: tok(x["text"], truncation=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Sustainable tactic flags (minimal, opt-in for pruning)
APPLY_PRUNING = False
PRUNE_FRACTION = 0.2  # fraction of parameters to prune among the most expensive linear layers

def apply_energy_aware_pruning(model, amount):
    # Collect Linear modules and sort by parameter count (proxy for energy intensity)
    linear_modules = [(module, name) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linear_modules:
        return
    linear_modules_sorted = sorted(linear_modules, key=lambda pair: pair[0].weight.numel(), reverse=True)
    # Select modules until cumulative params reach target fraction of total linear params
    total_params = sum(m.weight.numel() for m, _ in linear_modules_sorted)
    target = total_params * amount
    selected = []
    cum = 0
    for module, _ in linear_modules_sorted:
        selected.append((module, 'weight'))
        cum += module.weight.numel()
        if cum >= target:
            break
    if selected:
        prune.global_unstructured(selected, pruning_method=prune.L1Unstructured, amount=amount)

if APPLY_PRUNING:
    apply_energy_aware_pruning(model, PRUNE_FRACTION)

# Enable checkpoints to avoid knowledge loss on premature termination
args = TrainingArguments(
    output_dir='out',
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy='epoch',
    save_total_limit=1,
    logging_steps=1000
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print('done')