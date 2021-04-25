import random
import os
import shutil

classes = ['bird', 'down', 'stop', 'tree', 'up']
number_of_files_per_class = 50
tar_dir = 'randomly_selected_samples'

for word in classes:
    files = os.listdir(word)
    seen = []

    while True:
        filename = files[random.randrange(len(files))]

        if filename in seen:
            continue

        seen.append(filename)
        shutil.copyfile(os.path.join(word, filename), os.path.join(tar_dir, word) + "_" + str(len(seen)) + ".mp4")

        if len(seen) == number_of_files_per_class:
            print("Finished class " + word)
            break
