import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train = pd.DataFrame({
    "No": range(1, 17),
    "Texture": [19.83, 21.72, 20.20, 22.02, 17.35, 18.47, 25.27, 27.06,
                16.02, 18.68, 20.97, 16.32, 12.74, 19.76, 17.18, 19.86],
    "Smoothness": [0.09342, 0.09823, 0.14470, 0.09056, 0.09446, 0.11420,
                   0.10820, 0.10000, 0.10880, 0.09916, 0.11020, 0.11580,
                   0.09311, 0.08421, 0.08045, 0.10540],
    "y": [-1] * 8 + [1] * 8,
})
test = pd.DataFrame({
    "No": ["T1", "T2", "T3", "T4"],
    "Texture": [17.89, 18.70, 21.01, 19.94],
    "Smoothness": [0.09710, 0.11090, 0.10440, 0.10240],
    "y": [-1, -1, 1, 1],
})
feat = ["Texture", "Smoothness"]
X_tr_raw, y_tr = train[feat].values, train["y"].values
X_te_raw, y_te = test[feat].values, test["y"].values

scaler = StandardScaler().fit(X_tr_raw)
X_tr = scaler.transform(X_tr_raw)
X_te = scaler.transform(X_te_raw)

log = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)

C_MAL, C_BEN = "tab:red", "tab:blue"

def scatter_classes(ax, X, y, label=True, alpha=1.0):
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=C_MAL, marker="o", edgecolor="k",
               label="malignant (-1)" if label else None, alpha=alpha, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C_BEN, marker="s", edgecolor="k",
               label="benign (+1)" if label else None, alpha=alpha, zorder=3)

tx = np.linspace(10, 29, 300)
sy = np.linspace(0.07, 0.16, 300)
GX, GY = np.meshgrid(tx, sy)
grid_raw = np.c_[GX.ravel(), GY.ravel()]
grid_std = scaler.transform(grid_raw)

say("=" * 70, "\nQ1  Scatter plot and linear separability\n", "=" * 70)
fig, ax = plt.subplots(figsize=(7, 5))
scatter_classes(ax, X_tr_raw, y_tr)
for i, (x, s) in enumerate(X_tr_raw, 1):
    ax.annotate(str(i), (x, s), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("Mean texture")
ax.set_ylabel("Mean smoothness")
ax.set_title("Q1: Training data (raw units)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("q1_scatter.png", dpi=150)
plt.close(fig)

hard = SVC(kernel="linear", C=1e6).fit(X_tr, y_tr)
n_err_hard = int((hard.predict(X_tr) != y_tr).sum())
say(f"Linear SVM with C=1e6 (~hard margin): training errors = {n_err_hard} of 16")
say("-> linearly separable" if n_err_hard == 0 else
    "-> NOT linearly separable (even a nearly hard-margin linear SVM cannot "
    "classify every training point correctly)")

say("\nTraining feature ranges (raw):")
say(train[feat].describe().loc[["min", "max", "mean", "std"]].round(5).to_string())
say(f"\nScaler means = {scaler.mean_.round(5)}, stds = {scaler.scale_.round(5)}")

say("\n" + "=" * 70, "\nQ2  Linear SVM (C = 1)\n", "=" * 70)
lin = SVC(kernel="linear", C=1).fit(X_tr, y_tr)
w, b = lin.coef_[0], lin.intercept_[0]
sv_idx = lin.support_
pred = lin.predict(X_tr)
mis = np.where(pred != y_tr)[0]
say(f"w = {w.round(4)}, b = {b:.4f}")
say(f"Number of support vectors: {len(sv_idx)}  (observations {[int(v) for v in train['No'].values[sv_idx]]})")
say(f"Training misclassifications: {len(mis)}  (observations {[int(v) for v in train['No'].values[mis]]})")
margin = 2 / np.linalg.norm(w)
say(f"Geometric margin width 2/||w|| = {margin:.4f} (standardized units)")

def plot_linear(ax):
    f = lin.decision_function(grid_std).reshape(GX.shape)
    ax.contour(GX, GY, f, levels=[-1, 0, 1], colors=["gray", "k", "gray"],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2, 1.2])
    scatter_classes(ax, X_tr_raw, y_tr)
    ax.scatter(X_tr_raw[sv_idx, 0], X_tr_raw[sv_idx, 1], s=220, facecolors="none",
               edgecolors="gold", linewidths=2.5, label="support vectors", zorder=4)
    ax.set_xlim(tx[0], tx[-1]); ax.set_ylim(sy[0], sy[-1])
    ax.set_xlabel("Mean texture"); ax.set_ylabel("Mean smoothness")
    ax.set_title("Q2: Linear SVM (C=1) - boundary (solid), margins (dashed)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.3)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
plot_linear(ax)
fig.tight_layout(); fig.savefig("q2_linear_svm.png", dpi=150); plt.close(fig)

say("\n" + "=" * 70, "\nQ3  Hinge loss for the linear SVM (C = 1)\n", "=" * 70)
f_tr = lin.decision_function(X_tr)
yf = y_tr * f_tr
hinge = np.maximum(0, 1 - yf)
def category(v):
    if v < 0: return "misclassified"
    if v < 1: return "correct, inside margin"
    return "correct, outside margin"
tbl = pd.DataFrame({"No": train["No"], "y": y_tr, "f(x)": f_tr.round(4),
                    "y*f(x)": yf.round(4), "hinge": hinge.round(4),
                    "status": [category(v) for v in yf]})
say(tbl.to_string(index=False))

say("\nRepresentative observations:")
for name, mask in [("(i) correct, outside margin", yf >= 1),
                   ("(ii) correct, inside margin", (yf >= 0) & (yf < 1)),
                   ("(iii) misclassified", yf < 0)]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        say(f"{name}: none available")
        continue

    k = idx[np.argmin(np.abs(yf[idx] - 0.5))] if "inside" in name else idx[0]
    say(f"{name}: obs {train['No'][k]}  y={y_tr[k]:+d}  f={f_tr[k]:.4f}  "
        f"y*f={yf[k]:.4f}  hinge={hinge[k]:.4f}")
say(f"\nTotal hinge loss = {hinge.sum():.4f}")

say("\n" + "=" * 70, "\nQ4  Kernel comparison (C = 1)\n", "=" * 70)
gamma_default = 1.0 / X_tr.shape[1]
models = {
    "Linear":          SVC(kernel="linear", C=1),
    "Poly p=2":        SVC(kernel="poly", degree=2, gamma=1, coef0=1, C=1),
    "Poly p=3":        SVC(kernel="poly", degree=3, gamma=1, coef0=1, C=1),
    f"RBF g={gamma_default}": SVC(kernel="rbf", gamma=gamma_default, C=1),
}
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    rows.append({"model": name,
                 "train_acc": (m.predict(X_tr) == y_tr).mean(),
                 "test_acc": (m.predict(X_te) == y_te).mean(),
                 "n_SV": int(m.n_support_.sum()),
                 "test_pred": list(m.predict(X_te))})
res4 = pd.DataFrame(rows)
say(res4.to_string(index=False))

def region_plot(ax, m, title):
    Z = m.decision_function(grid_std).reshape(GX.shape)
    ax.contourf(GX, GY, Z > 0, levels=[-.5, .5, 1.5], colors=["#f6c4c4", "#c4d7f6"], alpha=.6)
    ax.contour(GX, GY, Z, levels=[-1, 0, 1], colors=["gray", "k", "gray"],
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    scatter_classes(ax, X_tr_raw, y_tr, label=False)
    ax.scatter(X_te_raw[y_te == -1, 0], X_te_raw[y_te == -1, 1], c=C_MAL, marker="X",
               s=110, edgecolor="k", linewidths=1.5, zorder=5)
    ax.scatter(X_te_raw[y_te == 1, 0], X_te_raw[y_te == 1, 1], c=C_BEN, marker="X",
               s=110, edgecolor="k", linewidths=1.5, zorder=5)
    ax.scatter(X_tr_raw[m.support_, 0], X_tr_raw[m.support_, 1], s=170, facecolors="none",
               edgecolors="gold", linewidths=2, zorder=4)
    ax.set_xlim(tx[0], tx[-1]); ax.set_ylim(sy[0], sy[-1])
    ax.set_title(title, fontsize=9); ax.grid(alpha=.3)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, (name, m), r in zip(axes.ravel(), models.items(), rows):
    region_plot(ax, m, f"{name} | train {r['train_acc']:.2f}, test {r['test_acc']:.2f}, SV={r['n_SV']}")
for ax in axes[1]: ax.set_xlabel("Mean texture")
for ax in axes[:, 0]: ax.set_ylabel("Mean smoothness")
fig.suptitle("Q4: Kernels (circles = train, X = test, gold rings = support vectors)")
fig.tight_layout(); fig.savefig("q4_kernels.png", dpi=150); plt.close(fig)

say("\n" + "=" * 70, "\nQ5  RBF SVM: C x gamma\n", "=" * 70)
Cs, Gs = [0.1, 1, 100], [0.1, 1, 10]
rows = []
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
for i, C in enumerate(Cs):
    for j, g in enumerate(Gs):
        m = SVC(kernel="rbf", C=C, gamma=g).fit(X_tr, y_tr)
        tr_acc = (m.predict(X_tr) == y_tr).mean()
        te_acc = (m.predict(X_te) == y_te).mean()
        rows.append({"C": C, "gamma": g, "train_acc": tr_acc, "test_acc": te_acc,
                     "n_SV": int(m.n_support_.sum()), "test_pred": list(m.predict(X_te))})
        region_plot(axes[i, j], m, f"C={C}, gamma={g} | train {tr_acc:.2f}, test {te_acc:.2f}, SV={int(m.n_support_.sum())}")
for ax in axes[-1]: ax.set_xlabel("Mean texture")
for ax in axes[:, 0]: ax.set_ylabel("Mean smoothness")
fig.suptitle("Q5: RBF SVM - rows: C = 0.1, 1, 100; columns: gamma = 0.1, 1, 10")
fig.tight_layout(); fig.savefig("q5_rbf_grid.png", dpi=150); plt.close(fig)
res5 = pd.DataFrame(rows)
say(res5.to_string(index=False))

with open("results.txt", "w") as fh:
    fh.write("\n".join(log))
print("\nSaved: q1_scatter.png, q2_linear_svm.png, q4_kernels.png, q5_rbf_grid.png, results.txt")
