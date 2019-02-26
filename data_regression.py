import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython

ipy = get_ipython()
if ipy  is not None:
    ipy.run_line_magic('matplotlib', 'inline')
auto_prices = pd.read_csv('Automobile price data _Raw_.csv')
def clean_auto_data(auto_prices):
    # recodificar nomes
    # arrumando os nomes das colunas 
    cols = auto_prices.columns
    auto_prices.columns = [str.replace('-', '_') for str in cols]
    # tratar valores perdidos
    # remove linhas com valores perdidos, e contas com  valor coded '?'
    cols =['price', 'bore', 'stroke',
          'horsepower', 'peak_rpm']
    for column in cols:
        auto_prices.loc[auto_prices[column] == '?', column] = np.nan
    auto_prices.dropna(axis = 0, inplace = True)

    # transforma o data type da coluna 
    # coverte alguma colunas em valores numericos 
    for column in cols:
        auto_prices[column] = pd.to_numeric(auto_prices[column])

    return auto_prices
auto_prices = clean_auto_data(auto_prices)
print(auto_prices.columns)
auto_prices.head()
auto_prices.dtypes
auto_prices.describe()

def count_unique(auto_prices, cols):
    for col in cols:
        print('\n' + 'For column' + col)
        print(auto_prices[col].value_counts())

cat_cols = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style',
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders',
            'fuel_system']
count_unique(auto_prices, cat_cols)

# visualização de data regression

def plot_histogram(auto_prices, cols, bins = 10):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) 
        ax = fig.gca() 
        auto_prices[col].plot.hist(ax = ax, bins = bins)
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel('Number of autos')
        plt.show()
        
num_cols = ['curb_weight', 'engine_size', 'city_mpg', 'price']    
plot_histogram(auto_prices, num_cols)

def plot_density_hist(auto_prices, cols,  bins= 10,  hist = False):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(auto_prices[col],  bins = bins, rug=True, hist=hist )
        plt.title('Histogram of' + col)
        plt.xlabel(col)
        plt.ylabel('Number of autos')
        plt.show()
plot_density_hist(auto_prices, num_cols, bins = 20, hist = True)

def plot_scatter(auto_prices, cols, col_y = 'price'):
    for col in cols:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of' + ' ' + col_y + 'vs' + ' ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()
num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']

def plot_scatter_t(auto_prices, cols, col_y = 'price', alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) 
        ax = fig.gca() 
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) 
        ax.set_xlabel(col) 
        ax.set_ylabel(col_y)
        plt.show()

plot_scatter_t(auto_prices, num_cols, alpha = 0.2)



# relação categorica e variaveis numericas


def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.show()

plot_desity_2d(auto_prices, num_cols, kind = 'hex')

def plot_box(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=auto_prices)
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.show()
        
cat_cols = ['fuel_type', 'aspiration', 'num_of_doors', 'body_style', 
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders']
plot_box(auto_prices, cat_cols)

def plot_violin(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices)
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.show()
        
plot_violin(auto_prices, cat_cols)

# usando a estetica para  projetar diferente dimensões

def plot_scatter_shape(auto_prices, cols, shape_col = 'fuel_type', col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] 
    unique_cats = auto_prices[shape_col].unique()
    for col in cols: 
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): 
            temp = auto_prices[auto_prices[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) 
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.legend()
        plt.show()
            
num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
plot_scatter_shape(auto_prices, num_cols)   

def plot_scatter_size(auto_prices, cols, shape_col = 'fuel_type', size_col = 'curb_weight',
                            size_mul = 0.000025, col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] 
    unique_cats = auto_prices[shape_col].unique()
    for col in cols: 
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): 
            temp = auto_prices[auto_prices[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha, "s":size_mul*auto_prices[size_col]**2}, 
                        fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) 
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']
plot_scatter_size(auto_prices, num_cols)

def plot_scatter_shape_size_col(auto_prices, cols, shape_col = 'fuel_type', size_col = 'curb_weight',
                            size_mul = 0.000025, color_col = 'aspiration', col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] 
    colors = ['green', 'blue', 'orange', 'magenta', 'gray'] 
    unique_cats = auto_prices[shape_col].unique()
    unique_colors = auto_prices[color_col].unique()
    for col in cols: 
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): 
            for j, color in enumerate(unique_colors):
                temp = auto_prices[(auto_prices[shape_col] == cat) & (auto_prices[color_col] == color)]
                sns.regplot(col, col_y, data=temp, marker = shapes[i],
                            scatter_kws={"alpha":alpha, "s":size_mul*temp[size_col]**2}, 
                            label = (cat + ' and ' + color), fit_reg = False, color = colors[j])
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) 
        plt.xlabel(col) 
        plt.ylabel(col_y)
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']        
plot_scatter_shape_size_col(auto_prices, num_cols)

# plots condicionados

def cond_hists(df, plot_cols, grid_col):
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
    return grid_col

# definindo colunas para fazer um histogram condicionado
plot_cols2 = ["length",
               "curb_weight",
               "engine_size",
               "city_mpg",
               "price"]

cond_hists(auto_prices, plot_cols2, 'drive_wheels')

def cond_plot(cols):
    for col in cols:
        g = sns.FacetGrid(auto_prices, col="drive_wheels", row = 'body_style', 
                      hue="fuel_type", palette="Set2", margin_titles=True)
        g.map(sns.regplot, col, "price", fit_reg = False)

num_cols = ["curb_weight", "engine_size", "city_mpg"]
cond_plot(num_cols)