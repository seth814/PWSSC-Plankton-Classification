import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def plot_conf_mat(conf_mat):
    plt.title('Confusion Matrix')
    sns.heatmap(conf_mat, cmap='hot', vmin=0.0, vmax=1.0, square=True, xticklabels=False, yticklabels=False)
    plt.show()

df = pd.read_csv('./model_results/inception_v3.csv')

acc = str(round(accuracy_score(df.y_true, df.y_pred), 4))
f1 = str(round(f1_score(df.y_true, df.y_pred, average='weighted'), 4))
conf_mat = confusion_matrix(df.y_true, df.y_pred)
conf_mat = conf_mat.astype(dtype=np.float16)
for row in range(conf_mat.shape[0]):
    conf_mat[row,:] = conf_mat[row,:] / sum(conf_mat[row,:])

plot_conf_mat(conf_mat)

print('Accuracy: {}'.format(acc))
print('F1 Score: {}'.format(f1))

for i in range(conf_mat.shape[0]):
    conf_mat[i,i] = 0.0
plot_conf_mat(conf_mat)

blank = np.zeros(conf_mat.shape)
x, y = np.where(conf_mat > 0.1)
for i, j in zip(x,y):
    blank[i,j] = 1.0

plot_conf_mat(blank)
