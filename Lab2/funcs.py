import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

import pandas as pd


def save_predicted(ids, predicted, method_name, res_col_name="Response"):
    df = pd.DataFrame(
        {
            "id": ids,
            res_col_name: predicted
        }
    )
    df.to_csv(f"./results/{method_name}.csv", index=False)


def cross_fit_clf(clf, x_train, y_train):
    cross = cross_validate(clf, x_train, y=y_train, return_estimator=True,
                           cv=5)

    return cross['estimator'][
        list(cross['test_score']).index(max(cross['test_score']))]


def test_by_cross_val(clf, x_train, y_train, x_test, y_test):
    clf = cross_fit_clf(clf, x_train, y_train)
    return get_probs_and_scores(clf, x_train, y_train, x_test, y_test)["test_score"]


def fit_classificator(classificator, params, x_train, y_train, x_test, y_test):
    clf = classificator()

    grid = GridSearchCV(clf, params, cv=5, scoring="roc_auc")
    grid.fit(x_train, y_train)

    best_params = grid.best_params_

    clf = grid.best_estimator_

    return {
        "model": clf,
        "best_params": best_params,
        "score": round(get_probs_and_scores(clf, x_train, y_train, x_test, y_test)
                       [
                           "test_score"
                       ], 3)
    }


def get_probs_and_scores(predictor, x_train, y_train, x_test, y_test):
    train_pred_proba = predictor.predict_proba(x_train)
    test_pred_proba = predictor.predict_proba(x_test)
    return {
        "train_pred_proba": train_pred_proba[:, 1],
        "test_pred_proba": test_pred_proba[:, 1],
        "train_score": roc_auc_score(y_train, train_pred_proba[:, 1]),
        "test_score": roc_auc_score(y_test, test_pred_proba[:, 1])
    }


def make_roc_curve_plot(predictor, title, x_train, y_train, x_test, y_test):
    resp = get_probs_and_scores(predictor, x_train, y_train, x_test, y_test)

    train_curve = roc_curve(y_train, resp["train_pred_proba"])

    test_curve = roc_curve(y_test, resp["test_pred_proba"])

    # строим график ROC-кривых
    plt.figure(figsize=(12, 6))
    plt.plot(*train_curve[:2],
             label='Train ROC curve (AUC={:.2f})'.format(resp["train_score"]))
    plt.plot(*test_curve[:2],
             label='Validation ROC curve (AUC={:.2f})'.format(resp["test_score"]))

    # добавляем диагональную линию
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # настраиваем параметры графика
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

    return resp["test_score"]


def make_mult_roc_curve_plots(predictors, x_train, y_train, x_test, y_test):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18, 16))

    for ax, (clf_model, label) in zip(axs.flatten(), predictors):
        probs_and_scores = get_probs_and_scores(clf_model,
                                                x_train, y_train,
                                                x_test, y_test)

        train_fpr, train_tpr, _ = roc_curve(y_train,
                                            probs_and_scores["train_pred_proba"])
        test_fpr, test_tpr, _ = roc_curve(y_test,
                                          probs_and_scores["test_pred_proba"])

        ax.plot(train_fpr, train_tpr,
                label=f'Training ROC curve ('
                      f'AUC = {probs_and_scores["train_score"]:.2f})')
        ax.plot(test_fpr, test_tpr,
                    label=f'Test ROC curve ('
                          f'AUC = {probs_and_scores["test_score"]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{label}')
        ax.legend(loc="lower right")

    plt.show()
