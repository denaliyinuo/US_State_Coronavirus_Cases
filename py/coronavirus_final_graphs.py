import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def read_file(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    return df


path = '/Users/nathanoliver/Desktop/Python/US State Coronavirus/csv/data.csv'

df = read_file(path)

df = df[df['state'] == 'yes']

df = df.reset_index(drop=True)

print(df)

print(df.columns)

num_col = ['total_cases', 'total_deaths', 'total_recovered',
           'active_cases', 'total_cases_per_capita', 'total_deaths_per_capita',
           'total_tests', 'total_tests_per_capita', 'population']


for i in num_col:
    df[i] = df[i].str.replace(',', '')


print(df['total_cases'])
print(df['total_deaths'])


num_col = ['total_cases', 'total_deaths', 'total_recovered',
           'active_cases', 'total_cases_per_capita', 'total_deaths_per_capita',
           'total_tests', 'total_tests_per_capita', 'population', 'rural',
           'exurban', 'suburban', 'urban']

for i in num_col:
    df[i] = pd.to_numeric(df[i])

df_pvi = df['pvi']
df_538 = df['lean_538']

df['pvi_new'] = 0
df['lean_538_new'] = 0

for i in range(len(df)):
    n = df_pvi[i]
    try:
        print(n[:1])
        print(n[2:])
        if n[:1] == 'R':
            df.loc[i, 'pvi_new'] = n[2:]
        elif n[:1] == 'D':
            df.loc[i, 'pvi_new'] = '-' + n[2:]
        elif n[:1] == 'EVEN':
            df.loc[i, 'pvi_new'] = '0'
    except TypeError:
        df.loc[i, 'pvi_new'] = ''

for i in range(len(df)):
    n = df_538[i]
    try:
        print(n[:1])
        print(n[2:])
        if n[:1] == 'R':
            df.loc[i, 'lean_538_new'] = n[2:]
        elif n[:1] == 'D':
            df.loc[i, 'lean_538_new'] = '-' + n[2:]
        elif n[:1] == 'EVEN':
            df.loc[i, 'lean_538_new'] = '0'
    except TypeError:
        df.loc[i, 'lean_538_new'] = ''


df['pvi_new'] = pd.to_numeric(df['pvi_new'])
df['lean_538_new'] = pd.to_numeric(df['lean_538_new'])

colors = ['#184e8f', '#49669f', '#6d80af', '#8e9bc0', '#afb7d0', '#d0d3e0',
          '#f1f1f1', '#f1d4d4', '#f0b8b8', '#ec9c9d', '#e67f83', '#de6069', '#d43d51']

print(df['pvi_new'])
# df['total_cases_per_capita'] = df['total_cases'] / df['population']

# print(df['total_cases_per_capita'])

# plt.scatter(df['suburban'] + df['urban'], df['total_cases_per_capita'])
# plt.show()


# x0 = [df['pvi_new'], df['pvi_new'], df['pvi_new']]
# y0 = [df['total_cases_per_capita'],
#       df['total_deaths_per_capita'], df['total_tests_per_capita']]


# y = np.ravel(y0[0])


def linear_regression(x1, y1):
    x = x1.to_numpy()
    y = y1.to_numpy()
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    print(model.score(x, y))
    print(model.intercept_)
    print(model.coef_)

    r2 = round(model.score(x, y), 2)
    b = model.intercept_
    m = model.coef_

    return m, b, r2, x, y


def plot_linear_regression(x, m, b):
    min_x = int(min(x))
    max_x = int(max(x))

    x_plot = range(min_x, 2 * max_x - min_x, max_x - min_x)
    y_plot = m * x_plot + b

    return x_plot, y_plot


fig, ax = plt.subplots(1, 1, sharex=True)

x = df['pvi_new']
y = df['total_cases_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax.scatter(x, y, label='US State COVID-19 Cases')
ax.plot(x_plot, y_plot, color='red',
        linestyle='--', label='$R^2$ = ' + str(r2))
ax.legend()
ax.set_title(
    'COVID-19 Cases vs. State\'s Political Leaning\n(data as of 12/08/20)')
ax.set_xlabel('Cook Partisan Voting Index')
ax.set_ylabel('COVID-19 Cases (per 1M people)')
ax.set_xlim(-27, 27)


# x1 = np.arange(-25, 30, 5)
# labels = ['D+25', 'D+20', 'D+15', 'D+10', 'D+5',
#           'EVEN', 'R+5', 'R+10', 'R+15', 'R+20', 'R+25']

x1 = np.arange(-20, 25, 10)
labels = ['D+20', 'D+10',
          'EVEN', 'R+10', 'R+20']

ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False)


df = df.set_index('state_name')
case = df['total_cases_per_capita'].sort_values(ascending=False)
print(case)

plt.show()
