import matplotlib.pyplot as plt
import re
val_loss = []
train_loss = []
epochs = range(1,124)
with open("/datadrive/speechtek/ephrem/E2E-ASR-pytorch/libri_exp1/ctc_bi_lr1e-5/log.txt", "r") as f:
    count = 1
    for line in f:
        line = line.split()
        #print(line)
        train_loss.append(float(re.sub("[^0-9.]", "",line[8])))
        #print(train_loss)
        val_loss.append(float(re.sub("[^0-9.]", "",line[13])))
        #print(val_loss)
plt.figure(figsize=(6,5))
plt.title("Training and Validation loss")
plt.plot(epochs, val_loss[:-1], 'y', label = "Validation Loss")
plt.plot(epochs, train_loss[:-1], 'b', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("/datadrive/speechtek/ephrem/E2E-ASR-pytorch/libri_exp1/ctc_bi_lr1e-5/loss.png")
#plt.show()
