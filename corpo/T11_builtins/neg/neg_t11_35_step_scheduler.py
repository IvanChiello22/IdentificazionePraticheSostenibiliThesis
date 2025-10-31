# 35) Scheduler “step” manuale
lr = 0.1
gamma = 0.5
step_size = 3
for epoch in range(1, 10):
    if epoch % step_size == 0:
        lr *= gamma
    # train...
print(lr)
