import numpy as np
import re

path = "C:/Users/em165153/Documents/NZSL/speech/encoded_samples/TAL_by_8/"
filename = "Result"

result = np.zeros((30, 36))
lif = -1
run = 0
mode = True  # True = cross-validation, False = test

# with open(path + "Result.txt", "r") as file:
with open(path + filename + ".txt", "r") as file:
    for line in file:
        if re.match("lif", line):
            lif += 1
        if re.match("Run", line):
            run = int(line.strip()[3:]) - 1
        if re.match("Cross-validation", line):
            mode = True
        if re.match("Test", line):
            mode = False
        if re.match("Accuracy", line):
            if mode:
                result[run, lif] = float(line.strip()[10:])
            else:
                result[run, lif+18] = float(line.strip()[10:])

print(np.max(result))
np.savetxt(path + filename + ".csv", result, fmt='%1.8f', delimiter=',')
