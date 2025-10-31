best = None
patience = 3
wait = 0
val_losses = [1.2, 1.0, 0.9, 0.92, 0.95, 0.96, 0.97]
for e, v in enumerate(val_losses, 1):
    if best is None or v < best:
        best = v
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break
print("stopped_at_epoch", e)