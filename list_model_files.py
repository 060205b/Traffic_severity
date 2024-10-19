import os

# Set the path to your models directory
models_dir = r'C:\Users\bhuva\Downloads\BCG Job simulation\Road Traffic acc severity project\models_pickle_file'

# List all files in the models directory
for filename in os.listdir(models_dir):
    print(filename)
