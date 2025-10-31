# 35) Scheduler "step" manuale
lr = 0.1
gamma = 0.5
step_size = 3
num_epochs = 9
decays = num_epochs // step_size
lr *= gamma ** decays
print(lr)