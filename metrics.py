import numpy as np
# also in sklearn
# from sklearn.metrics import precision_score


# def sub_results(confusion_matrix):
#     n = len(confusion_matrix)
#     total_sum = confusion_matrix.sum()
#     tp_all = np.diag(confusion_matrix)
#     sum_rows = confusion_matrix.sum(axis=1)  # Actual totals per class
#     sum_cols = confusion_matrix.sum(axis=0)  # Predicted totals per class

#     # sub_results = {i: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for i in range(n)}
#     sub_results = {
#         i: {
#             "TP": tp_all[i],
#             "FP": sum_rows[i] - tp_all[i],
#             "FN": sum_cols[i] - tp_all[i],
#             "TN": total_sum - sum_rows[i] - sum_cols[i] + tp_all[i]
#         } for i in range(n)
#     }
#     return sub_results

# Per-class metrics ##########################################################################
def per_class_precision(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_precision = np.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)

    return per_class_precision

def per_class_recall(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    actual_positives = confusion_matrix.sum(axis=1)

    # Calculate per-class precision safely (handles division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.where(actual_positives > 0, true_positives / actual_positives, 0.0)

    return per_class_recall

def per_class_specificity(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)
    sum_samples = actual_positives.sum()

    true_negatives = sum_samples - predicted_positives - actual_positives + true_positives
    actual_negatives = sum_samples - actual_positives

    # Calculate per-class precision safely (handles division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_specificity = np.where(actual_negatives > 0, true_negatives / actual_negatives, 0.0)

    return per_class_specificity

def per_class_fscore(confusion_matrix, beta):
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_class = np.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)
        rec_class = np.where(actual_positives > 0, true_positives / actual_positives, 0.0)
        
        beta_sq = beta ** 2
        denom = (beta_sq * prec_class) + rec_class
        per_class_fscore = np.where(denom > 0, (1 + beta_sq) * (prec_class * rec_class) / denom, 0.0)

    return per_class_fscore

def calculate_per_class_metrics(confusion_matrix, beta=1):
    results = {
        "precision": per_class_precision(confusion_matrix),
        "recall": per_class_recall(confusion_matrix),
        # "specificity": per_class_specificity(confusion_matrix),
        "fscore": per_class_fscore(confusion_matrix, beta=beta)
    }
    print(results)
    
    return results

# Class-averaged metrics ##########################################################################
# TP / TP + FP
def precision(confusion_matrix, average='weighted'):
    # Extract True Positives (diagonal) and Predicted Positives (column sums)
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)

    # Calculate per-class precision safely (handles division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_precision = np.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)

    if average == 'micro':
        tp_sum = true_positives.sum()
        fp_sum = predicted_positives.sum() - tp_sum
        return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    
    elif average == 'macro':
        return np.mean(per_class_precision)
    
    elif average == 'weighted':
        sum_samples = actual_positives.sum()
        return np.sum(per_class_precision * actual_positives) / sum_samples if sum_samples > 0 else 0.0
    
    else:
        raise ValueError("Accepted average modes are: 'micro', 'macro', 'weighted'")

# TP / TP + FN
def recall(confusion_matrix, average='weighted'):
    # Extract True Positives (diagonal) and Predicted Positives (column sums)
    true_positives = np.diag(confusion_matrix)
    actual_positives = confusion_matrix.sum(axis=1)

    # Calculate per-class precision safely (handles division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.where(actual_positives > 0, true_positives / actual_positives, 0.0)

    if average == 'micro':
        tp_sum = true_positives.sum()
        fn_sum = actual_positives.sum() - tp_sum
        return tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    
    elif average == 'macro':
        return np.mean(per_class_recall)
    
    elif average == 'weighted':
        sum_samples = actual_positives.sum()
        return np.sum(per_class_recall * actual_positives) / sum_samples if sum_samples > 0 else 0.0
    
    else:
        raise ValueError("Accepted average modes are: 'micro', 'macro', 'weighted'")
    

# TN / TN + FP
def specificity(confusion_matrix, average='weighted'):
    # Extract True Positives (diagonal) and Predicted Positives (column sums)
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)
    sum_samples = actual_positives.sum()

    true_negatives = sum_samples - predicted_positives - actual_positives + true_positives
    actual_negatives = sum_samples - actual_positives

    # Calculate per-class precision safely (handles division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_specificity = np.where(actual_negatives > 0, true_negatives / actual_negatives, 0.0)

    if average == 'micro':
        tn_sum = true_negatives.sum()
        tp_sum = true_positives.sum()
        fp_sum = predicted_positives.sum() - tp_sum
        return tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) > 0 else 0.0
    
    elif average == 'macro':
        return np.mean(per_class_specificity)
    
    elif average == 'weighted':
        sum_samples = actual_positives.sum()
        return np.sum(per_class_specificity * actual_positives) / sum_samples if sum_samples > 0 else 0.0
    
    else:
        raise ValueError("Accepted average modes are: 'micro', 'macro', 'weighted'")

# [(1 + β^2) * prec * rec] / [β^2 * prec + rec]
def fscore(confusion_matrix, beta=1, average='weighted'):
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    actual_positives = confusion_matrix.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_class = np.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)
        rec_class = np.where(actual_positives > 0, true_positives / actual_positives, 0.0)
        
        beta_sq = beta ** 2
        denom = (beta_sq * prec_class) + rec_class
        per_class_fscore = np.where(denom > 0, (1 + beta_sq) * (prec_class * rec_class) / denom, 0.0)

    if average == 'micro':
        prec_micro = precision(confusion_matrix, average='micro')
        rec_micro = recall(confusion_matrix, average='micro')
        denom_micro = (beta**2 * prec_micro) + rec_micro
        return (1 + beta**2) * (prec_micro * rec_micro) / denom_micro if denom_micro > 0 else 0.0
    elif average == 'macro':
        return np.mean(per_class_fscore)
    elif average == 'weighted':
        total_samples = actual_positives.sum()
        return np.sum(per_class_fscore * actual_positives) / total_samples if total_samples > 0 else 0.0
    else:
        raise ValueError("Accepted average modes are: 'micro', 'macro', 'weighted'")

def calculate_overall_metrics(confusion_matrix, beta=1, average='weighted'):
    results = {
        "precision": precision(confusion_matrix, average=average),
        "recall": recall(confusion_matrix, average=average),
        # "specificity": specificity(confusion_matrix, average=average),
        "fscore": fscore(confusion_matrix, average=average, beta=beta)
    }
    print(results)

    return results


# matrix = np.random.rand(10, 10)
# print(precision(matrix))