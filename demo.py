import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE,"UCI HAR Dataset")
tasks_ = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]

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

def spectrogram(signal,frame=32,step=8):
    """Cut a signal into overlapping pieces and return the strength of each frequency over time."""
    result = []
    for i in range(0,len(signal)-frame+1,step):
        piece = signal[i:i+frame]
        result.append(np.abs(np.fft.rfft(piece)))
    return np.array(result).T

def make_image(xw,yw,zw):
    stack = np.vstack([spectrogram(xw),spectrogram(yw),spectrogram(zw)])
    return np.log1p(stack)

def make_all(xs,ys,zs):
    result_all = []
    for i in range(len(xs)):
        result_all.append(make_image(xs[i],ys[i],zs[i]))
    return np.array(result_all)

def pictures_features(imgs):
    dec = imgs.astype(float)
    return np.hstack([dec.mean(axis=2),dec.std(axis=2)])

print("Loading data and making pictures, please wait...")
x_tr = load_signal("total_acc_x","train")
y_tr = load_signal("total_acc_y","train")
z_tr = load_signal("total_acc_z","train")
x_te = load_signal("total_acc_x","test")
y_te = load_signal("total_acc_y","test")
z_te = load_signal("total_acc_z","test")
y_train = load_labels("train")
y_test = load_labels("test")

img_train = make_all(x_tr,y_tr,z_tr)
img_test = make_all(x_te,y_te,z_te)
lo = img_train.min()
hi = img_train.max()
img_train = np.clip((img_train-lo)/(hi-lo)*255,0,255).astype(np.uint8)
img_test = np.clip((img_test-lo)/(hi-lo)*255,0,255).astype(np.uint8)

F_train = pictures_features(img_train)
F_test = pictures_features(img_test)
f_mean = F_train.mean(axis=0)
f_std = F_train.std(axis=0)
f_train_n = (F_train-f_mean) / (f_std + 1e-8)
f_test_n = (F_test-f_mean) / (f_std + 1e-8)
mag_test = np.sqrt(x_te**2 + y_te**2 + z_te**2)
print("Ready!")

def guess_one(i,k=5):
    """Guess the activity of one test window and also return the votes of the k nearest windows."""
    distances = np.linalg.norm(f_train_n - f_test_n[i], axis=1)
    nearest_labels = y_train[np.argsort(distances)[:k]]
    votes = np.bincount(nearest_labels, minlength=7)[1:]
    return np.argmax(votes) + 1, votes

# window layout: signal | spectrogram | votes
fig = plt.figure(figsize=(14,6))
fig.canvas.manager.set_window_title("Activity Lens - Live Demo")
ax_sig = fig.add_axes([0.05,0.30,0.30,0.55])
ax_img = fig.add_axes([0.40,0.30,0.22,0.55])
ax_vote = fig.add_axes([0.70,0.30,0.27,0.55])
ax_slider = fig.add_axes([0.12,0.12,0.62,0.04])
ax_button = fig.add_axes([0.80,0.10,0.12,0.07])

slider = Slider(ax_slider,"Test window",0,len(y_test)-1,valinit=0,valstep=1)
button = Button(ax_button,"Random")

def update(val):
    i = int(slider.val)
    guess, votes = guess_one(i)
    real = y_test[i]

    ax_sig.clear()
    ax_sig.plot(mag_test[i])
    ax_sig.set_ylim(0,2.5)
    ax_sig.set_title("Movement signal")
    ax_sig.set_xlabel("Reading (50 per second)")
    ax_sig.set_ylabel("Magnitude")

    ax_img.clear()
    ax_img.imshow(img_test[i],origin="lower",aspect="auto",cmap="jet",vmin=0,vmax=255)
    ax_img.axhline(16.5,color="white",linewidth=1)
    ax_img.axhline(33.5,color="white",linewidth=1)
    ax_img.set_yticks([8,25,42])
    ax_img.set_yticklabels(["x","y","z"])
    ax_img.set_xlabel("Time piece")
    ax_img.set_title("Spectrogram picture")

    ax_vote.clear()
    colors = ["tab:green" if c == guess else "tab:gray" for c in range(1,7)]
    ax_vote.barh(tasks_,votes,color=colors)
    ax_vote.set_xlim(0,5)
    ax_vote.invert_yaxis()
    ax_vote.set_title("Votes of the 5 nearest windows")

    if guess == real:
        result = "CORRECT"
        color = "green"
    else:
        result = "WRONG"
        color = "red"
    fig.suptitle(f"Window {i}    Real: {tasks_[real-1]}    Guessed: {tasks_[guess-1]}    {result}",
                 fontsize=15,color=color,fontweight="bold")
    fig.canvas.draw_idle()

def random_window(event):
    slider.set_val(np.random.randint(0,len(y_test)))

slider.on_changed(update)
button.on_clicked(random_window)
update(0)
plt.show()