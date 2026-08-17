import re

with open('/Users/tanayjalan/Downloads/programs/GIT/qmlhep-gsoc2027/Phase2_PyTorch/cnn_cifar10/train.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_main = False

for i, line in enumerate(lines):
    if line.startswith("def train_one_epoch"):
        pass # Handle manually later
        
    if line.startswith("print('Train')"):
        in_main = True
        new_lines.append("if __name__ == '__main__':\n")
        new_lines.append("    " + line)
        continue
    
    # We will just find where def train_one_epoch and evaluate are, and move them up
