"""IDS evaluation metrics — accuracy, F1, FPR, confusion matrix."""

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	fbeta_score,
	confusion_matrix,
	classification_report,
)


def compute_ids_metrics(
	y_true,
	y_pred,
	num_classes: int,
	beta: float = 1.0,
	class_weights: list[float] | None = None,
) -> dict:
	"""Compute standard IDS classification metrics.

	Args:
		y_true: Ground truth labels (array-like)
		y_pred: Predicted labels (array-like)
		num_classes: Number of classes (2 for binary, 10 for multi)

	Args:
		y_true: Ground truth labels (array-like)
		y_pred: Predicted labels (array-like)
		num_classes: Number of classes (2 for binary, 10 for multi)
		beta: F-beta parameter (<1 favors precision/low FPR, >1 favors recall)
		class_weights: Per-class importance weights for weighted F1 (None=uniform)

	Returns:
		Dict with accuracy, f1_macro, fbeta_macro (if beta!=1.0),
		f1_weighted (if class_weights), per_class_f1, per_class_recall,
		fpr (for binary), and confusion_matrix.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	acc = accuracy_score(y_true, y_pred)
	f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
	f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

	cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

	# Per-class recall from confusion matrix
	per_class_recall = []
	for i in range(num_classes):
		row_sum = cm[i].sum()
		per_class_recall.append(cm[i][i] / row_sum if row_sum > 0 else 0.0)

	result = {
		"accuracy": float(acc),
		"f1_macro": float(f1_macro),
		"per_class_f1": [float(f) for f in f1_per_class],
		"per_class_recall": per_class_recall,
		"confusion_matrix": cm.tolist(),
	}

	# F-beta score (when beta != 1.0)
	if beta != 1.0:
		fbeta = fbeta_score(y_true, y_pred, beta=beta, average="macro", zero_division=0)
		result["fbeta_macro"] = float(fbeta)
		result["beta"] = beta

	# Weighted F1 with custom class weights
	if class_weights is not None:
		# sklearn's "weighted" average uses sample counts; for custom weights,
		# compute manually: weighted avg of per-class F1
		weights = np.array(class_weights[:num_classes], dtype=float)
		weights = weights / weights.sum()  # normalize
		f1_weighted = float(np.dot(f1_per_class[:len(weights)], weights))
		result["f1_weighted"] = f1_weighted
		result["class_weights"] = [float(w) for w in weights]

	# FPR for binary classification (false positive rate)
	if num_classes == 2 and cm.shape == (2, 2):
		tn, fp, fn, tp = cm.ravel()
		result["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
		result["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

	return result


def format_ids_report(metrics: dict, class_names: list[str] | None = None) -> str:
	"""Format metrics into a readable report string.

	Args:
		metrics: Dict from compute_ids_metrics()
		class_names: Optional class names for labeling

	Returns:
		Formatted multi-line string.
	"""
	lines = []
	lines.append(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
	lines.append(f"F1 Macro:  {metrics['f1_macro']:.4f}")

	if "fbeta_macro" in metrics:
		lines.append(f"F{metrics['beta']:.1f} Macro: {metrics['fbeta_macro']:.4f}")
	if "f1_weighted" in metrics:
		lines.append(f"F1 Weighted: {metrics['f1_weighted']:.4f}")

	if "fpr" in metrics:
		lines.append(f"FPR:       {metrics['fpr']:.4f}")
		lines.append(f"FNR:       {metrics['fnr']:.4f}")

	lines.append("")
	lines.append("Per-class breakdown:")

	num_classes = len(metrics["per_class_f1"])
	for i in range(num_classes):
		name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
		f1 = metrics["per_class_f1"][i]
		recall = metrics["per_class_recall"][i]
		lines.append(f"  {name:20s}  F1={f1:.4f}  Recall={recall:.4f}")

	return "\n".join(lines)
