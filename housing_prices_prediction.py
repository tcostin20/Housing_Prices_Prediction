'''
------------------------------------------------------------------------------------------------
Description: Analysis of Housing Prices in California using Machine Learning
Author: Thomas Costin (tcostin20@gmail.com)
Date started: 10/05/2026
Data source: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
------------------------------------------------------------------------------------------------
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path

#################################################################################################
def load_data() -> pd.DataFrame:
    '''
    Function that imports California Housing Price dataset, which has variables
    including median income and median house value for each block group in 
    California.
    
    Parameters:
    -----------
    None
    
    Returns:
    -----------
        df (DataFrame): pandas DataFrame containing the California housing data
    '''
    # load dataset from sklearn
    df = fetch_california_housing(as_frame = True).frame # type: ignore

    # remove rows with missing values
    df.dropna(inplace = True)

    # remove duplicate rows
    df.drop_duplicates(inplace = True)

    return df

#################################################################################################
def visualize_data(df: pd.DataFrame) -> None:
    '''
    Function that creates a heatmap of the correlation matrix for features in the California 
    housing dataset.

    Parameters:
    -----------
        df (DataFrame): pandas DataFrame containing the California housing data
    
    Returns:
    -----------
        None        
    '''
    # specify dimensions of the heatmap
    plt.figure(figsize=(10, 8))

    # exclude dependent variable from the correlation matrix data
    features = df.drop(columns = ['MedHouseVal'])

    # initialize heatmap using seaborn
    sns.heatmap(features.corr(), 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f',
                linewidths=2)

    # update x and y axis label orientation
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # define title for the heatmap
    plt.title('Feature Correlation Heatmap', pad = 20, font = {'size': 16, 'weight': 'bold'})

    # display heatmap
    plt.show()

#################################################################################################
def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Function that splits the raw dataset into a training dataset and a testing
    dataset

    Parameters:
    -----------
        df (DataFrame): pandas DataFrame containing the California housing data
    
    Returns:
    -----------
        X_train (DataFrame): pandas DataFrame containing the training subset
                             for the independent variables in the California 
                             housing dataset
        X_test (DataFrame):  pandas DataFrame containing the testing subset
                             for the independent variables in the California 
                             housing dataset
        Y_train (Series):    pandas Series containing the training subset
                             for the dependent variable in the California
                             housing dataset
        Y_test (Series):     pandas Series containing the testing subset
                             for the dependent variable in the California
                             housing dataset
    '''
    # create separate objects for the independent and dependent variables
    df_X = df.drop(columns = ['MedHouseVal'])
    df_Y = df['MedHouseVal']

    # split the two datasets into training and testing subsets
    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 0)

    return X_train, X_test, Y_train, Y_test

#################################################################################################
def run_Lasso(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.Series, Y_test: pd.Series) -> dict:
    '''
    Function that fits a lasso regression model using the training dataset and then uses the 
    trained model to predict outcomes from the testing dataset
    
    Parameters:
    -----------
        X_train (DataFrame): pandas DataFrame containing the training subset
                             for the independent variables in the California 
                             housing dataset
        X_test (DataFrame):  pandas DataFrame containing the testing subset
                             for the independent variables in the California 
                             housing dataset
        Y_train (Series):    pandas Series containing the training subset
                             for the dependent variable in the California
                             housing dataset
        Y_test (Series):     pandas Series containing the testing subset
                             for the dependent variable in the California
                             housing dataset

    Returns:
    -----------
        lasso_results (Dict): mean absolute percentage error for the SVR model after
                              being trained using the training dataset
    '''
    # standardize features for train and test datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # calculate alpha for Lasso model using cross validation
    lasso = LassoCV(cv = 5, max_iter = 10_000)
    lasso.fit(X_train_scaled, Y_train)

    # store lasso results in dictionary
    lasso_results = {'alpha': lasso.alpha_, 'R2': lasso.score(X_test_scaled, Y_test), 'Features': lasso.coef_ != 0}

    return lasso_results

#################################################################################################
def visualize_lasso_path(X_train: pd.DataFrame, Y_train: pd.Series, lasso_results: dict) -> None:
    '''
    TBU
    '''
    # standardize features for train and test datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # fit the lasso regression for a spectrum of alpha values
    alphas, path, *_ = lasso_path(X_train_scaled, Y_train)

    # transpose paths so that each row is a single alpha value and each column is a single feature
    paths = pd.DataFrame(path.T, columns = X_train.columns)

    # add column that labels which alpha each row corresponds to
    paths['alpha'] = alphas

    # reformat to paths DataFrame to group statistics by feature
    paths = paths.melt(id_vars = 'alpha', var_name = 'feature', value_name = 'coef')

    # create line plot that shows how each feature's coefficient changes as alpha changes
    sns.lineplot(data = paths, x = 'alpha', y = 'coef', hue = 'feature')

    # rescale x-axis to a log scale to account for alphas that range significantly in magnitude
    plt.xscale('log')

    # add dashed vertical line at the alpha that cross validation choose
    plt.axvline(lasso_results['alpha'], linestyle = '--', color = 'black')
    
    # display the plot
    plt.show()
    

#################################################################################################
def main():
    # load housing data
    df = load_data()
    
    # generate correlation matrix for dataset features
    visualize_data(df)

    # split dataset into training and testing subsets
    X_train, X_test, Y_train, Y_test = train_test(df)

    # run lasso regression regression and return relevant statistics
    lasso_results = run_Lasso(X_train, X_test, Y_train, Y_test)

    # visualize alpha cross validation
    visualize_lasso_path(X_train, Y_train, lasso_results)

if __name__ == "__main__":
    main()
