import pandas as pd
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, r2_score, roc_auc_score, mean_squared_error
import numpy as np

from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV

import pydotplus


def prepare_data(lab2: pd.DataFrame, lab3: pd.DataFrame):
    """
    Фнукция подготовки данных для работы деревьев
    :param lab2: данные из ЛР2
    :param lab3: данные из ЛР3
    :return: кортеж формата (x, y, x, y),
    где первая пара - данные 2 работы, вторая - данные 3
    """
    def prepare2():
        """
        Подготовка данных второй работы
        :return: x y для выборки
        """
        df_cat = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        df_num = ["Age", "Driving_License", "Previously_Insured", "Region_Code",
                  "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
        df_all = df_num + df_cat
        result_col = "Response"
        lab2.drop("id", axis=1, inplace=True, errors="ignore")

        le = LabelEncoder()
        for col in df_cat:
            lab2[col] = le.fit_transform(lab2[col])
            lab2[col] = le.fit_transform(lab2[col])

        scaler = MinMaxScaler()
        lab2[df_all] = scaler.fit_transform(lab2[df_all])
        lab2[df_all] = scaler.fit_transform(lab2[df_all])

        x = lab2[df_all]
        y = lab2[result_col]

        x, y = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),
                          random_state=7).fit_resample(x, y)
        return x, y

    def prepare3():
        """
        Подготовка данных 3 работы
        :return: x y для выборки
        """
        lab3.drop("id", axis=1, inplace=True, errors="ignore")
        res_col = "Цена"

        corr_matrix = lab3.corr()
        threshold = 0.29
        corr_price_matrix = lab3.corrwith(lab3[res_col]).abs()
        weak_corr_features = set(corr_matrix[corr_price_matrix <= threshold].index)
        lab3.drop(weak_corr_features, axis=1, inplace=True)

        feature_select = lab3.drop(res_col, axis=1)
        corr_series = feature_select.corr().abs().stack().sort_values()
        corr_series = corr_series[~corr_series.duplicated()]
        threshold = 0.75
        drops = set(
            [feature2 for (feature1, feature2), corr in corr_series.items() if
             corr > threshold and corr != 1])
        lab3.drop(drops, axis=1, inplace=True)
        x, y = lab3.drop(res_col, axis=1), lab3[res_col]
        return x, y

    return *prepare2(), *prepare3()


def get_probs_and_scores(predictor, x_train, y_train, x_test, y_test):
    """
    Функция получения probA и scores модели
    :param predictor: модель
    Остальные параметры в объяснении не нуждаются
    :param x_train: выборка
    :param y_train: выборка
    :param x_test: выборка
    :param y_test: выборка
    :return: словарь с результатами
    """
    train_pred_proba = predictor.predict_proba(x_train)
    test_pred_proba = predictor.predict_proba(x_test)
    return {
        "train_pred_proba": train_pred_proba[:, 1],
        "test_pred_proba": test_pred_proba[:, 1],
        "train_score": roc_auc_score(y_train, train_pred_proba[:, 1]),
        "test_score": roc_auc_score(y_test, test_pred_proba[:, 1])
    }


def make_roc_curve_plot(predictor, title, x_train, y_train, x_test, y_test):
    """
    Функция построения ROC-AUC кривой
    :param predictor: модель, для которой строим кривую
    :param title: Название модели
    :param x_train: выборка
    :param y_train: выборка
    :param x_test: выборка
    :param y_test: выборка
    """
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


def plot_scores_compare(clf_scores, reg_scores):
    """
    Функция построения сравнения результатов
    моделей классификаторов и регрессии
    :param clf_scores: результаты моделей - классификаторов
    :param reg_scores: резульаты моделей - регрессии
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.4)

    roc_ax = axs[0]
    names, scores = clf_scores["Model name"], clf_scores["ROC-AUC"]
    roc_ax.set_ylim([min(scores) - 0.01, max(scores) + 0.01])
    roc_ax.set_xticklabels(list(names), rotation=45)
    roc_ax.set_title("Classifiers")
    roc_ax.set_ylabel("ROC-AUC")

    roc_ax.bar(names, scores)
    for i, val in enumerate(scores):
        roc_ax.text(i, val, str(round(val, 3)), ha='center',
                    va='bottom')

    r2_ax = axs[1]
    names, scores = reg_scores["Model name"], reg_scores["R^2"]
    r2_ax.set_ylim([min(scores) - 0.01, max(scores) + 0.01])
    r2_ax.set_xticklabels(list(names), rotation=45)
    r2_ax.set_title("Regressions")
    r2_ax.set_ylabel("R^2")
    r2_ax.bar(names, scores)
    for i, val in enumerate(scores):
        r2_ax.text(i, val, str(round(val, 3)), ha='center',
                   va='bottom')
    plt.show()


def plot_param_comp(model, params, kind, x_train, x_test, y_train, y_test):
    """
    Функция построения графиков
    зависимости score модели от изменения ее параметров
    :param model: исследуемая модель
    :param params: параметры для исследования
    :param kind: тип модели (clf - классификатор, reg - регрессия)
    :param x_train: выборка
    :param x_test: выборка
    :param y_train: выборка
    :param y_test: выборка
    """
    res_dict = {}
    for param, vals in params.items():
        res_dict[param] = {"train_scores": [],
                           "test_scores": [],
                           "values": vals}
        for val in vals:
            to_fit = model(**{param: val})
            to_fit.fit(x_train, y_train)
            if kind == "clf":
                train_auc = roc_auc_score(y_train,
                                          to_fit.predict_proba(x_train)[:, 1])
                test_auc = roc_auc_score(y_test, to_fit.predict_proba(x_test)[:, 1])
            else:
                train_auc = to_fit.score(x_train, y_train)
                test_auc = to_fit.score(x_test, y_test)
            res_dict[param]["train_scores"].append(train_auc)
            res_dict[param]["test_scores"].append(test_auc)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    for ax, (param, stat_dict) in zip(axs.flatten(), res_dict.items()):
        ax.plot(stat_dict["values"], stat_dict["train_scores"], label="train_score")
        ax.plot(stat_dict["values"], stat_dict["test_scores"], label="test_score")
        ax.set_ylabel("ROC-AUC" if kind == "clf" else "R^2")
        ax.set_title(param)
        ax.legend()
    plt.show()


class ClfTree:
    def __init__(self, model, x, y, params, grid_plots_params):
        """
        Класс, реализующий необходимые в работе действия с деревом классификации
        :param model: класс модели
        :param x: выборка
        :param y: выборка
        :param params: сетка параметров для обучения модели
        :param grid_plots_params: параметры для построения графиков зависимости score
        от их изменения
        """
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.3, random_state=42)
        self.model = model
        # to_fit = model(**params)
        # to_fit.fit(self.x_train, self.y_train)
        # self.fitted_model = to_fit
        # self.best_params = params
        grid = GridSearchCV(estimator=model(), param_grid=params, cv=5,
                            scoring="roc_auc")
        grid.fit(self.x_train, self.y_train)
        self.fitted_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.grid_plots_params = grid_plots_params

        test_pred_proba = self.fitted_model.predict_proba(self.x_test)
        self.score = roc_auc_score(self.y_test, test_pred_proba[:, 1])

    def plot_roc_auc(self):
        """
        Метод построения кривой roc-auc
        """
        make_roc_curve_plot(self.fitted_model, "Classifier", self.x_train,
                            self.y_train, self.x_test, self.y_test)

    def params_compare(self):
        """
        Метод построения графиков зависимости score
        от изменения параметров
        """
        plot_param_comp(self.model, self.grid_plots_params, "clf",
                        self.x_train, self.x_test, self.y_train, self.y_test)

    def max_leaf_compare(self):
        """
        Метод визуализации зависимости качества дерева
        от изменения max_leaf_nodes и max_depth
        """
        max_leaf_vals = list(range(2, 20))
        max_depth_vals = list(range(2, 20))
        roc_auc_vals = []
        for leafs in max_leaf_vals:
            for depth in max_depth_vals:
                model = self.model(max_leaf_nodes=leafs, max_depth=depth)
                model.fit(self.x_train, self.y_train)
                roc_auc = roc_auc_score(self.y_test,
                                        model.predict_proba(self.x_test)[:, 1])

                roc_auc_vals.append(roc_auc)

        max_leaf_vals = np.array(max_leaf_vals)
        max_depth_vals = np.array(max_depth_vals)
        roc_auc_vals = np.array(roc_auc_vals)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('max_leaf_nodes')
        ax.set_ylabel('max_depth')
        ax.set_zlabel('roc_auc')
        ax.set_title('Leaf nodes compare')

        x, y = np.meshgrid(max_leaf_vals, max_depth_vals)
        z = np.array(roc_auc_vals).reshape(x.shape)

        ax.plot_surface(x, y, z)
        plt.show()

    def export_to_pdf(self):
        """
        Метод экспорта дерева в pdf
        :return: текстовое представления результата экспорта
        """
        dots = export_graphviz(self.fitted_model)
        graph = pydotplus.graph_from_dot_data(dots)
        return f'Сохранение дерева классификации: ' \
               f'{"Успешно" if graph.write_pdf(r"clf.pdf") else "Ошибка"}'


class RegTree:
    def __init__(self, model, x, y, params, grid_plots_params):
        """
        Класс, реализующий необходимые в работе действия с деревом регрессии
        :param model: класс модели
        :param x: выборка
        :param y: выборка
        :param params: сетка параметров для обучения модели
        :param grid_plots_params: параметры для построения графиков зависимости score
        от их изменения
        """
        self.model = model
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.3, random_state=42)

        self.model = model
        grid = GridSearchCV(estimator=model(), param_grid=params, cv=5,
                            scoring="roc_auc")
        grid.fit(self.x_train, self.y_train)
        self.fitted_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.grid_plots_params = grid_plots_params

        self.score = self.fitted_model.score(self.x_test, self.y_test)

    def params_compare(self):
        """
        Метод построения графиков зависимости score
        от изменения параметров
        """
        plot_param_comp(self.model, self.grid_plots_params, "reg",
                        self.x_train, self.x_test, self.y_train, self.y_test)

    def max_leaf_compare(self):
        """
        Метод визуализации зависимости качества дерева
        от изменения max_leaf_nodes и max_depth
        """
        max_leaf_vals = list(range(2, 20))
        max_depth_vals = list(range(2, 20))
        roc_auc_vals = []
        for leafs in max_leaf_vals:
            for depth in max_depth_vals:
                model = self.model(max_leaf_nodes=leafs, max_depth=depth)
                model.fit(self.x_train, self.y_train)
                roc_auc_vals.append(model.score(self.x_test, self.y_test))

        max_leaf_vals = np.array(max_leaf_vals)
        max_depth_vals = np.array(max_depth_vals)
        roc_auc_vals = np.array(roc_auc_vals)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('max_leaf_nodes')
        ax.set_ylabel('max_depth')
        ax.set_zlabel('R^2')
        ax.set_title('Leaf nodes compare')

        x, y = np.meshgrid(max_leaf_vals, max_depth_vals)
        z = np.array(roc_auc_vals).reshape(x.shape)

        ax.plot_surface(x, y, z)
        plt.show()

    def get_scores(self):
        """
        Метод получения оценки дерева в разных метриках
        :return: пандас серия с результатами оценки
        """
        y_pred = self.fitted_model.predict(self.x_test)

        r_sq = r2_score(self.y_test, y_pred)

        n, m = self.x_test.shape
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - m - 1)

        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        ll = -0.5 * (n * np.log(2 * np.pi) + n * np.log(
            np.sum(((self.y_test - y_pred) ** 2)) / n) + n)
        aic = 2 * m - 2 * ll
        bic = m * np.log(n) - 2 * ll

        res = pd.Series({
            "R^2": r_sq,
            "Adj R^2": adj_r_sq,
            "RMSE": rmse,
            "AIC": aic,
            "BIC": bic
        })
        return res

    def export_to_pdf(self):
        """
        Метод экспорта дерева в pdf
        :return: текстовое представления результата экспорта
        """
        dots = export_graphviz(self.fitted_model)
        graph = pydotplus.graph_from_dot_data(dots)
        return f'Сохранение дерева регрессии: ' \
               f'{"Успешно" if graph.write_pdf(r"reg.pdf") else "Ошибка"}'
