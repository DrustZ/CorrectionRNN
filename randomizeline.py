import random

with open('mixed_dataset') as f:
    lines = f.readlines()

random.shuffle(lines)
random.shuffle(lines)

with open('small_train_data', 'w') as fw:
    for line in lines:
        fw.write(line)
