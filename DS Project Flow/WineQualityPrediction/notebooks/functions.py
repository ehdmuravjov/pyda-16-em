import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

def fill_missings(data, cols, value='mean'):
    if value == 'mean':
        for col in cols:
            data[col].fillna(data[col].mean(), inplace=True)
    
    elif value == 'median':
        for col in cols:
            data[col].fillna(data[col].median(), inplace=True)
    
    elif value == 'type_transorm':
        for col in cols:
            data[col].fillna(data.groupby(['type'])[col].transform('mean'), inplace=True)
    
    elif isinstance(value, float) or isinstance(value, int):
        for col in cols:
            data[col].fillna(value, inplace=True)
            
    return data

def plot_scatter_and_lin(data, x_col, y_col, save_fig_dir):
    model = LinearRegression()
    X = data[[x_col]]
    y = data[y_col]
    model.fit(X, y)
    y_pred = model.predict(X)
    _, ax = plt.subplots(figsize=(8,6))
    plt.scatter(X, y, s=5, color='steelblue')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs. {y_col}')
    plt.grid()
    plt.plot(X, y_pred, c='black')
    plt.savefig(f'{save_fig_dir}/scatter_and_lin_{x_col}_vs_{y_col}.png')

def plot_variable_importance(X, y, save_fig_dir):
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)


def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns=['Importance'],
        index=X.columns
    )
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[:10].plot(kind='barh')
    
    
def plot_distribution(df, var, target, save_fig_dir, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set( xlim= (0, df[var].max()))
    facet.add_legend()
    plt.savefig(f'{save_fig_dir}/distribution{var}_{target}.png')
    
def plot_corr_heatmap(corr, save_fig_dir):
    _, ax = plt.subplots(figsize=(12,10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, annot=True, linewidths=0.5, center=0, )
    plt.savefig(f'{save_fig_dir}/corr_heatmap.png')
    
def plot_hist(data, var, save_fig_dir):
    _, ax = plt.subplots(figsize=(6,4))
    _ = sns.countplot(var, data=data, ax=ax, palette="Set2")
    ax.set_title(f"Распределение значений переменной {var}")
    plt.savefig(f'{save_fig_dir}/hist_{var}.png')
    plt.show()