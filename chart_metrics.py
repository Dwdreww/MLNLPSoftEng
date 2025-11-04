import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average=None)

plt.figure(figsize=(10,6))
x = range(len(label_cols))
plt.bar(x, f1, color='skyblue')
plt.xticks(x, label_cols, rotation=45)
plt.title("F1-Score per Toxic Category")
plt.ylabel("F1-Score")
plt.ylim(0,1)
plt.show()
