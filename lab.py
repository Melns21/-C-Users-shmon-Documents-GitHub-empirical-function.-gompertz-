import numpy as np
from scipy.stats import binom
from scipy.stats import rankdata
from scipy.stats import norm

def wilcoxon(sample1, sample2):
    combined = sample1 + sample2
    combined_sorted = sorted(combined)
    ranks = {}
    for i, val in enumerate(combined_sorted):
        if val not in ranks:
            ranks[val] = [i + 1]
        else:
            ranks[val].append(i + 1)
    rank_sum1 = sum(ranks[val][0] for val in sample1)
    rank_sum2 = sum(ranks[val][0] for val in sample2)
    U = min(rank_sum1, rank_sum2)
    n1 = len(sample1)
    n2 = len(sample2)
    expected_U = n1 * (n1 + n2 + 1) / 2
    z1 = (U - expected_U) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    z = 2 * (1 - norm.cdf(abs(z1)))
    return U, z

def criteria_signs(data1, data2, alpha=0.05):
    min_len = min(len(data1), len(data2))
    if min_len == 0:
        raise ValueError("Обе выборки пусты.")
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    differences = np.array(data1) - np.array(data2)
    n_plus = np.sum(differences > 0)
    n_minus = np.sum(differences < 0)
    n = len(differences)
    p_value = 2 * min(binom.cdf(min(n_plus, n_minus), n, 0.5), 1 - binom.cdf(max(n_plus, n_minus) - 1, n, 0.5))
    return p_value

#read data
def read_data(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

#Data
data1 = read_data('var5.txt')
data2 = read_data('var6.txt')

#Значимости
alpha = 0.05

# Проверка гипотезы методом знаков
p_value = criteria_signs(data1, data2)
print("Тест знаков:")
print("p-значение:", p_value)
if p_value < alpha:
    print("Различия статистически значимы p <", alpha, "")
else:
    print("Различия не являются статистически значимыми p >=", alpha, "")

# Тест Вилкоксона
U, z = wilcoxon(data1, data2)
print("\nТест Вилкоксона:")
print("p-значение:", z)
if z < alpha:
    print("Отвергаем нулевую гипотезу: различия статистически значимы p <", alpha, "")
else:
    print("Не удалось отвергнуть нулевую гипотезу: Различия не являются статистически значимыми p >=", alpha, "")
