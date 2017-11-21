from sklearn import metrics


def confusion_matrix(
    y_true, y_pred, neg_label='negative', pos_label='positive'
):
    matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    return {
        f'true {neg_label}': tn,
        f'false {pos_label}': fp,
        f'false {neg_label}': fn,
        f'true {pos_label}': tp
    }
