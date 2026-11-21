"""
Activity Lens - Method 2 (Spectrogram + OpenCV + kNN)

Turns phone accelerometer signals from the UCI HAR dataset into
spectrogram pictures, processes them with OpenCV, turns each picture
into numbers, and guesses the activity with k-Nearest Neighbours.
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

# Paths
BASE = os.path.dirname(os.path.abspath(__file__)) # folder this file is in
DATA = os.path.join(BASE,"UCI HAR Dataset") # dataset folder next to it


# Step 1: Load the data
def load_signal(name,split):
    """Load one sensor file for train or test."""

    n = f"{name}_{split}.txt"
    path = os.path.join(DATA,split,"Inertial Signals",n)
    return np.loadtxt(path)

def load_labels(split):
    """Load the split labels."""

    s = f"y_{split}.txt" 
    path = os.path.join(DATA,split,s)
    return np.loadtxt(path,dtype=int)

# Load the 3 directions of movement and the activity labels for training
x = load_signal("total_acc_x", "train")
y = load_signal("total_acc_y", "train")
z = load_signal("total_acc_z", "train")
labels = load_labels("train")
formula = (x**2) + (y**2) + (z**2)
mag = np.sqrt(formula)
walk = mag[labels == 1][0]
lay = mag[labels == 6][0]

# Plot both signals: walking is wavy (steps), laying is flat (only gravity)
plt.plot(walk,label="Walking")
plt.plot(lay,label="Laying")
plt.legend()
plt.title("Walking vs Laying")
plt.xlabel("Reading(50 per seconds)")
plt.ylabel("Magnitude")
plt.savefig(os.path.join(BASE, "outputs", "walk_vs_lay.png"))
plt.close()


# Step 2: Turn one window into a spectrogram picture
def spectrogram(signal,frame=32,step=8):
    """Cut a signal into overlapping pieces and return the strength of each frequency over time."""
    result = []
    for i in range(0,len(signal)-frame+1,step):
        piece = signal[i:i+frame]
        freq = np.abs(np.fft.rfft(piece)) # frequency
        result.append(freq)
    return np.array(result).T # .T means transporse it flips the rows and columns

def make_image(xw,yw,zw):
    """Make one picture for a window by stacking the x, y and z spectrograms (shape 51 x 13)."""
    x_spectro = spectrogram(xw)
    y_spectro = spectrogram(yw)
    z_spectro = spectrogram(zw)
    stack = np.vstack([x_spectro,y_spectro,z_spectro])
    wrap_stack = np.log1p(stack)
    return wrap_stack

# Positions of the first walking and first laying windows
iw = np.where(labels==1)[0][0]
il = np.where(labels==6)[0][0]

# Make their pictures and show them side by side
walk_pic = make_image(x[iw],y[iw],z[iw])
lay_pic = make_image(x[il],y[il],z[il])
plt.subplot(1,2,1) #means 1 row, 2 columns
plt.imshow(walk_pic,origin="lower",aspect="auto")
plt.title("Walking")
plt.subplot(1,2,2)
plt.imshow(lay_pic,origin="lower",aspect="auto")
plt.title("Laying")
plt.savefig(os.path.join(BASE, "outputs", "spec_walk_vs_lay.png"))
plt.close()


# Step 3: Pictures for every window + OpenCV
def make_all(split):
    """Make a spectrogram picture for every window in train or test."""
    xs = load_signal("total_acc_x",split)
    ys = load_signal("total_acc_y",split)
    zs = load_signal("total_acc_z",split)
    result_all = []
    for i in range(len(xs)):
        result_all.append(make_image(xs[i],ys[i],zs[i]))
    return np.array(result_all)

img_train = make_all("train") 
img_test = make_all("test")
y_train = labels
y_test = load_labels("test")

# Darkest and brightest values, taken from TRAIN only
hi = img_train.max()
lo = img_train.min()

def to_uint8(imgs):
    """Scale pictures to the 0-255 image range (min-max scaling) using train's lo and hi."""
    values = (imgs-lo)/(hi-lo) * 255
    sort_values = np.clip(values,0,255) 
    return sort_values.astype(np.uint8)  
img_train = to_uint8(img_train)
img_test = to_uint8(img_test)

# OpenCV: one walking picture, made bigger and coloured for the slides
one_wpic = img_train[iw]
big = cv2.resize(one_wpic,(260,510),interpolation=cv2.INTER_NEAREST)   # (width, height), sharp blocks
flipped = np.flipud(big)                                                # low frequencies at the bottom
colored = cv2.applyColorMap(flipped,colormap=cv2.COLORMAP_JET)          # red = strong, blue = weak
cv2.imwrite(os.path.join(BASE,"outputs","opencv_walk.png"),colored)


# Step 4: Turn each picture into numbers
def pictures_features(imgs):
    """For each picture, take the mean and std of every row (frequency band): 51 + 51 = 102 numbers."""
    dec = imgs.astype(float)
    mean_ = dec.mean(axis=2)            # average strength of each band over time
    std_ = dec.std(axis=2)              # how much each band changes over time
    return np.hstack([mean_,std_])
F_train = pictures_features(img_train)  # (7352, 102)
F_test = pictures_features(img_test)    # (2947, 102)

# Normalize every feature using TRAIN mean and std (1e-8 avoids dividing by zero)
f_mean = F_train.mean(axis=0)
f_std = F_train.std(axis=0)
f_train_n = (F_train-f_mean) / (f_std + 1e-8)
f_test_n = (F_test-f_mean) / (f_std + 1e-8)


# Step 5: Guess the activity with kNN and check the results
def predict_knn(X_tr, y_tr, X_te, k=5):
    """For each test window, find the k closest train windows and let them vote on the activity."""
    preds = []
    for sample in X_te:
        distances = np.linalg.norm(X_tr - sample, axis=1) 
        nearest_idx = np.argsort(distances)[:k]  
        nearest_labels = y_tr[nearest_idx]  
        counts = np.bincount(nearest_labels)
        preds.append(np.argmax(counts))  
    return np.array(preds)
predict_ = predict_knn(f_train_n,y_train,f_test_n,k=5)

# Overall accuracy: fraction of test windows guessed correctly
print(np.mean(predict_ == y_test))

# Accuracy for each activity
tasks_ = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
for c in range(1,7):
    activities = predict_[y_test == c] 
    acc = np.mean(activities==c)    
    print(f"{tasks_[c-1]} and {acc * 100:.1f}%")

# Confusion matrix: rows = real activity, columns = guessed activity
cm = np.zeros((6,6),dtype=int)
for t, p in zip(y_test, predict_):
    cm[t - 1, p - 1] += 1 
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Method 2 Confusion Matrix")
plt.xticks(range(6), tasks_, rotation=45)
plt.yticks(range(6), tasks_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(BASE,"outputs","confusion_matrix_method2.png"))
plt.close()