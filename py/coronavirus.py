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


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

x = df['pvi_new']
y = df['total_cases_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax1.scatter(x, y)
ax1.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax1.legend()
ax1.set_title('Per Capita Cases vs. PVI')

x = df['pvi_new']
y = df['total_deaths_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax2.scatter(df['pvi_new'], df['total_deaths_per_capita'])
ax2.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax2.legend()
ax2.set_title('Per Capita Deaths vs. PVI')

x = df['pvi_new']
y = df['total_tests_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax3.scatter(df['pvi_new'], df['total_tests_per_capita'])
ax3.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax3.legend()
ax3.set_title('Per Capita Tests vs. PVI')
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

x = df['lean_538_new']
y = df['total_cases_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax1.scatter(df['lean_538_new'], df['total_cases_per_capita'])
ax1.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax1.legend()
ax1.set_title('Per Capita Cases vs. 538 Lean')

x = df['lean_538_new']
y = df['total_deaths_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax2.scatter(df['lean_538_new'], df['total_deaths_per_capita'])
ax2.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax2.legend()
ax2.set_title('Per Capita Deaths vs. 538 Lean')

x = df['lean_538_new']
y = df['total_tests_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

x = df['lean_538_new']
y = df['total_tests_per_capita']
ax3.scatter(df['lean_538_new'], df['total_tests_per_capita'])
ax3.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax3.legend()
ax3.set_title('Per Capita Tests vs. 538 Lean')
# plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

x = df['density_538']
y = df['total_cases_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax1.scatter(df['density_538'], df['total_cases_per_capita'])
ax1.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax1.legend()
ax1.set_title('Per Capita Cases vs. State Density')

x = df['density_538']
y = df['total_deaths_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax2.scatter(df['density_538'], df['total_deaths_per_capita'])
ax2.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax2.legend()
ax2.set_title('Per Capita Deaths vs. State Density')

x = df['density_538']
y = df['total_tests_per_capita']
m, b, r2, x, y = linear_regression(x, y)
x_plot, y_plot = plot_linear_regression(x, m, b)

ax3.scatter(df['density_538'], df['total_tests_per_capita'])
ax3.plot(x_plot, y_plot, color='red', linestyle='--', label='R2 = ' + str(r2))
ax3.legend()
ax3.set_title('Per Capita Tests vs. State Density')
plt.show()


# fig, (ax3, ax4) = plt.subplots(2, 1, sharex=True)

# ax3.scatter(df['suburban'] + df['urban'], df['total_cases_per_capita'])
# ax3.set_title('Per Capita Deaths vs. Density')

# ax4.scatter(df['suburban'] + df['urban'], df['total_deaths_per_capita'])
# ax4.set_title('Per Capita Deaths vs. Density')
# plt.show()
