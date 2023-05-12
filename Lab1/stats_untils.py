import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sigmaclip
import pandas as pd


def visualisation(df, data_type: str, data_cols: list, data_titles: list = None):
    """
    Функция отрисовки графиков
    :param df: исходный датафрейм
    :param data_type: тип отрисовываемых данных
    :param data_cols: столбцы из датафрейма, для отрисовки
    :param data_titles: заголовки для столбцов
    """
    def _visual_qualitative(data, title):
        x_label = "Варианты"
        y_label = "Количество"
        fig, axes = plt.subplots(1, 2,
                                 constrained_layout=True,
                                 figsize=(16, 8))

        counts = data.value_counts()
        fig.suptitle(title, fontsize=20)

        bar_axe, pie_axe = axes

        bar_axe.bar(counts.index, counts.values)
        bar_axe.set_xticks(np.arange(len(counts.index)))
        bar_axe.set_xticklabels(counts.index)
        bar_axe.set_xlabel(x_label)
        bar_axe.set_ylabel(y_label)
        bar_axe.set_title("Столбчатая диаграмма")

        pie_axe.pie(counts.values, labels=counts.index, autopct='%1.2f%%')
        pie_axe.set_title("Круговая диаграмма")

        plt.show()

    def _visual_quantitative(data, title):
        data = pd.to_numeric(data, errors="coerce")

        fig, axes = plt.subplots(1, 2,
                                 constrained_layout=True,
                                 figsize=(16, 8))
        fig.suptitle(title, fontsize=20)

        hist_axe, box_axe = axes

        sns.histplot(data=data, kde=True, ax=hist_axe)
        hist_axe.set_xlabel("Значения")
        hist_axe.set_ylabel("Относительная плотность")
        hist_axe.set_title("Гистограмма с графиком функции распределения")

        sns.boxplot(data, ax=box_axe, orient='h')
        box_axe.set_xlabel("Значения")
        box_axe.set_ylabel("")
        box_axe.set_title("Ящик с усами")

        plt.show()

    visual_func = _visual_qualitative \
        if data_type == "qualitative" else \
        _visual_quantitative

    if data_titles is None:
        data_titles = data_cols

    for col, title in zip(data_cols, data_titles):
        visual_func(df[col], title)


def quartile_method(quartile_df: pd.DataFrame):
    """
    Метод квартилей
    :param quartile_df: датафрейм для применения метода квартилей
    :return: измененный датафрейм, с отсеченными значениями
    """
    num_cols = quartile_df.select_dtypes(include=['float']).columns
    q1 = quartile_df[num_cols].quantile(0.25)
    q3 = quartile_df[num_cols].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    quartile_df = quartile_df[
        ~((quartile_df[num_cols] < lower_bound) | (quartile_df[num_cols] > upper_bound)).any(
            axis=1)]
    quartile_df = quartile_df.reset_index()
    quartile_df.pop('index')
    return quartile_df


def sigma_method(sigma_df: pd.DataFrame):
    """
    Метод сигм
    :param sigma_df: датафрейм для применения метода сигм
    :return: измененный датафрейм, с отсеченными значениями
    """
    num_cols = sigma_df.select_dtypes(include=['float']).columns
    for col in num_cols:
        data = sigma_df[col].dropna()
        clean_data, low, high = sigmaclip(data, low=3, high=3)
        sigma_df = sigma_df.loc[(sigma_df[col].isin(clean_data)) | (sigma_df[col].isna())]

    sigma_df = sigma_df.reset_index()
    sigma_df.pop('index')
    return sigma_df


def fix_strings(data):
    """
    Функция очистки строковых серий от пустых значений
    :param data: серия с пустыми значениями
    :return: чистая серия
    """
    return data.replace({" ": None, "-": None}).replace({np.nan: ""}).apply(
        lambda x: x.capitalize()).replace({"": np.nan})


def is_na(serie: pd.Series):
    """
    Функция проверки на содержание NaN в серии
    :param serie: серия для проверки
    :return: True при наличии NaN, False при отсутствии
    """
    return serie.isna().any()
