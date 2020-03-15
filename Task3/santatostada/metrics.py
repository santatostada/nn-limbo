def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(ground_truth.shape[0]):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                tp += 1
            else:
                fp += 1
                
        else:
            if prediction[i] == ground_truth[i]:
                tn += 1
            else:
                fn += 1
        
        if prediction.shape[0] != 0:
            accuracy = (tp + tn) / prediction.shape[0]
        else:
            accuracy = 0

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) != 0:
            recall = tp / (tp + fn)

        else:
            recall = 0

        if (recall + precision) != 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    tp = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            tp += 1

    if prediction.shape[0] != 0:
        accuracy = tp / prediction.shape[0]
    else:
        accuracy = 0

    return accuracy