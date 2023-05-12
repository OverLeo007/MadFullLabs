import random
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

random.seed(42)


def subplots_args(n):
    args = {}
    if n == 1:
        return {"nrows": 1, "ncols": 1, "figsize": (6, 6)}
    for i in range(int(n ** 0.5), 0, -1):
        if n % i == 0:
            args["nrows"] = i
            args["ncols"] = n // i
            args["figsize"] = (args["ncols"] * 6, args["nrows"] * 6)
            return args


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


class Ensemble:
    def __init__(self, ensemble_est, base_ests,
                 xy_train, xy_test,
                 n_est_range, params_to_grid=None):
        self.xy_train, self.xy_test = xy_train, xy_test
        self.model = ensemble_est
        self.model_name = str(self.model).split(".")[-1][:-2]

        self.base_estimators = {str(est).split(".")[-1][:-2]: est for est in
                                base_ests}
        self.n_est_range = n_est_range
        self.params_to_grid = params_to_grid

        self.best_model = None
        self.best_score = None

    def score(self, model, xy):
        if self.model_name.endswith("Regressor"):
            return model.score(*xy)
        elif self.model_name.endswith("Classifier"):
            return roc_auc_score(xy[1], model.predict_proba(xy[0])[:, 1])
        else:
            raise ValueError("Unknown model type")

    def scoring_measure(self):
        if self.model_name.endswith("Regressor"):
            return "R^2"
        elif self.model_name.endswith("Classifier"):
            return "ROC-AUC"
        else:
            raise ValueError("Unknown model type")

    def params_count_compare(self):
        res_dict = {
            estimator_name: {
                "train_scores": [],
                "test_scores": [],
                "values": [],
                "models": []
            } for estimator_name in self.base_estimators.keys()
        }
        fig, axs = plt.subplots(**subplots_args(len(self.base_estimators)))
        fig.suptitle(self.model_name)
        for ax, (est_name, estimator) in zip(axs.flatten(),
                                             self.base_estimators.items()):
            for n in self.n_est_range:
                model = self.model(estimator(), n_estimators=n, random_state=42)
                model.fit(*self.xy_train)
                train_score = self.score(model, self.xy_train)
                test_score = self.score(model, self.xy_test)
                res_dict[est_name]["train_scores"].append(train_score)
                res_dict[est_name]["test_scores"].append(test_score)
                res_dict[est_name]["values"].append(n)
                res_dict[est_name]["models"].append(model)
            vals, train, test = res_dict[est_name]["values"], res_dict[est_name][
                "train_scores"], res_dict[est_name]["test_scores"]
            ax.plot(vals, train, label="train_score")
            ax.plot(vals, test, label="test_score")
            ax.set_ylabel(self.scoring_measure())
            ax.set_xlabel("n_estimators")
            ax.set_title(est_name)
            ax.legend()
        plt.show()

        best_est_name, best_val_dict = max(res_dict.items(),
                                           key=lambda x: x[1]["test_scores"])
        # self.best_base_estimator = self.base_estimators[best_est_name]
        self.best_score = max(best_val_dict["test_scores"])
        best_idx = best_val_dict["test_scores"].index(
            self.best_score
        )
        self.best_model = best_val_dict["models"][best_idx]

        print("Лучшая модель:", best_est_name)
        print("Лучшее значение n_estimators:", best_val_dict["values"][best_idx])
        print(f"{self.scoring_measure()}:", self.best_score)


class Stacking:
    def __init__(self, stacking_model, estimators, xy_train, xy_test, n_samples=3,
                 sample_len=3):
        self.xy_train, self.xy_test = xy_train, xy_test
        self.model = stacking_model
        self.model_name = str(self.model).split(".")[-1][:-2]
        if sample_len == "random":
            sample_len = random.randint(2, len(estimators) - 1)
        if n_samples == "all":
            self.estimator_combs = list(combinations(estimators, sample_len))
        else:
            self.estimator_combs = random.sample(
                list(combinations(estimators, sample_len)),
                n_samples
            )

        self.best_combination = None
        self.best_model = None
        self.best_score = None

    def score(self, model, xy):
        if self.model_name.endswith("Regressor"):
            return model.score(*xy)
        elif self.model_name.endswith("Classifier"):
            return roc_auc_score(xy[1], model.predict_proba(xy[0])[:, 1])
        else:
            raise ValueError("Unknown model type")

    def scoring_measure(self, to_cv=False):
        if self.model_name.endswith("Regressor"):
            return "R^2" if not to_cv else "r2"
        elif self.model_name.endswith("Classifier"):
            return "ROC-AUC" if not to_cv else "roc_auc"
        else:
            raise ValueError("Unknown model type")

    def test_na(self, estimators):
        for name, est in estimators:
            print("Пробуем", name)
            model = self.model(estimators=[(name, est)])
            try:
                model.fit(*self.xy_train)
            except ValueError:
                print(f"Виноват {name}!")
        print("Виновник не найден, беда(")

    @staticmethod
    def call_models(not_called):
        return [(m_name, m_cls()) for m_name, m_cls in not_called]

    def find_best_combination(self):
        comb_names = []
        train_scores = []
        test_scores = []
        models = []
        for comb in self.estimator_combs:
            model = self.model(estimators=self.call_models(comb))
            try:
                model.fit(*self.xy_train)
            except ValueError:
                print("Ошибка, тк один из est выдал NaN")
                self.test_na([(m_name, m_cls()) for m_name, m_cls in comb])
                return
            train_scores.append(self.score(model, self.xy_train))
            test_scores.append(self.score(model, self.xy_test))
            comb_names.append('\n'.join(map(lambda x: x[0], comb)))
            models.append(model)

        fig, ax = plt.subplots()
        fig.suptitle(self.model_name)

        ax.bar(comb_names, train_scores, color="b", label="train_score")
        ax.bar(comb_names, test_scores, color="r", label="test_score")
        ax.set_xticks(range(len(comb_names)))
        ax.set_xticklabels(comb_names)
        ax.set_xlabel("model combinations")
        ax.set_ylabel(self.scoring_measure())
        ax.set_ylim([min(test_scores) - 0.01, max(train_scores) + 0.01])
        ax.legend()
        plt.tight_layout()
        plt.show()

        best_idx = test_scores.index(max(test_scores))
        self.best_combination = self.estimator_combs[best_idx]
        self.best_model = models[best_idx]
        self.best_score = self.score(self.best_model, self.xy_test)
        print("Лучшая комбинация:", *map(lambda x: x[0], self.best_combination))
        print(f"{self.scoring_measure()}:", test_scores[best_idx])

    def cross_val_fit(self):
        to_cf = self.model(estimators=self.call_models(self.best_combination))
        cross = cross_validate(to_cf, *self.xy_train, cv=5, return_estimator=True)
        self.best_model = cross['estimator'][
            list(cross['test_score']).index(max(cross['test_score']))
        ]
        self.best_score = self.score(self.best_model, self.xy_test)
        print(f"{self.scoring_measure()} после cross_val: ", self.best_score)
