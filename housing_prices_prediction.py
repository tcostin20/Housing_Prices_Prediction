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
def visualize_data(df: pd.DataFrame, dep_var: str) -> None:
    '''
    Function that creates a heatmap of the correlation matrix for features in the California 
    housing dataset.

    Parameters:
    -----------
        df (DataFrame): pandas DataFrame containing the California housing data
        dep_var (str):  name of the dependent variable as a string
    
    Returns:
    -----------
        None        
    '''
    # specify dimensions of the heatmap
    plt.figure(figsize = (10, 8))

    # exclude dependent variable from the correlation matrix data
    features = df.drop(columns = [dep_var])

    # initialize heatmap using seaborn
    sns.heatmap(features.corr(), 
                annot = True, 
                cmap = 'coolwarm', 
                fmt = '.2f',
                linewidths = 2)

    # update x and y axis label orientation
    plt.xticks(rotation = 45, ha = 'right', fontsize = 10)
    plt.yticks(rotation = 0, fontsize = 10)

    # define title for the heatmap
    plt.title('Feature Correlation Heatmap', pad = 20, fontsize = 16, fontweight = 'bold')

    # display heatmap
    plt.show()

#################################################################################################
def split_data(df: pd.DataFrame, dep_var: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Function that splits the raw dataset into a training dataset and a testing
    dataset

    Parameters:
    -----------
        df (DataFrame): pandas DataFrame containing the California housing data
        dep_var (str):  name of the dependent variable as a string
    
    Returns:
    -----------
        x_train (DataFrame): pandas DataFrame containing the training subset
                             for the independent variables in the California 
                             housing dataset
        x_test (DataFrame):  pandas DataFrame containing the testing subset
                             for the independent variables in the California 
                             housing dataset
        y_train (Series):    pandas Series containing the training subset
                             for the dependent variable in the California
                             housing dataset
        y_test (Series):     pandas Series containing the testing subset
                             for the dependent variable in the California
                             housing dataset
    '''
    # create separate objects for the independent and dependent variables
    df_x = df.drop(columns = [dep_var])
    df_y = df[dep_var]

    # split the two datasets into training and testing subsets
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 0)

    return x_train, x_test, y_train, y_test

#################################################################################################
def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Function that scales feature training and test datasets using a standard scaler

    Parameters:
    -----------
        x_train (DataFrame): pandas DataFrame containing the training subset
                             for the independent variables in the California 
                             housing dataset
        x_test (DataFrame):  pandas DataFrame containing the testing subset
                             for the independent variables in the California 
                             housing dataset
    Returns:
    -----------
        x_scaled (tuple): tuple of scaled train and test data for the features
    '''
    # initialize standard scaler
    scaler = StandardScaler()
    scaler.set_output(transform = 'pandas')
    
    # scale train and test feature data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled # type: ignore

#################################################################################################
def run_lasso(x_train_scaled: pd.DataFrame, x_test_scaled: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    '''
    Function that fits a lasso regression model using the training dataset and then uses the 
    trained model to predict outcomes from the testing dataset
    
    Parameters:
    -----------
        x_train_scaled (DataFrame): pandas DataFrame containing the scaled training subset
                                    for the independent variables in the California 
                                    housing dataset
        x_test_scaled (DataFrame):  pandas DataFrame containing the scaled testing subset
                                    for the independent variables in the California 
                                    housing dataset
        y_train (Series):           pandas Series containing the training subset
                                    for the dependent variable in the California
                                    housing dataset
        y_test (Series):            pandas Series containing the testing subset
                                    for the dependent variable in the California
                                    housing dataset

    Returns:
    -----------
        lasso_results (Dict): dictionary with the alpha, r-squared, coefficients, and 
                              feature names the fitted lasso model returns
    '''
    # calculate alpha for Lasso model using cross validation
    lasso = LassoCV(cv = 5, max_iter = 10_000)
    lasso.fit(x_train_scaled, y_train)

    # store lasso results in dictionary
    lasso_results = {'alpha': lasso.alpha_, 
                     'R2': lasso.score(x_test_scaled, y_test), 
                     'coefs': pd.Series(lasso.coef_),
                     'features': pd.Series(x_train_scaled.columns)}

    return lasso_results

#################################################################################################
def visualize_coef_path(x_train_scaled: pd.DataFrame, y_train: pd.Series, lasso_results: dict) -> None:
    '''
    Function that creates a line graph showing how the value of coefficients by feature changes
    as alpha changes. Includes a dashed vertical line showing the value of alpha that is
    ultimately selected after cross validation.
    
    Parameters:
    -----------
        x_train_scaled (DataFrame):  pandas DataFrame containing the scaled training subset
                                     for the independent variables in the California 
                                     housing dataset
        y_train (Series):            pandas Series containing the training subset
                                     for the dependent variable in the California
                                     housing dataset
        lasso_results (dict):        dictionary containing alpha, r-squared, and the
                                     coefficients for each feature obtained from a CV
                                     lasso model
    
    Returns:
    -----------
        None
    '''
    # fit the lasso regression for a spectrum of alpha values
    alphas, path, *_ = lasso_path(x_train_scaled, y_train)

    # transpose paths so that each row is a single alpha value and each column is a single feature
    paths = pd.DataFrame(path.T, columns = x_train_scaled.columns)

    # add column that labels which alpha each row corresponds to
    paths['alpha'] = alphas

    # reformat to paths DataFrame to group statistics by feature
    paths = paths.melt(id_vars = 'alpha', var_name = 'feature', value_name = 'coef')

    # create line plot that shows how each feature's coefficient changes as alpha changes
    sns.lineplot(data = paths, x = 'alpha', y = 'coef', hue = 'feature')
    
    # update x-axis and y-axis labels
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient (in Standardized Units)')

    # rescale x-axis to a log scale to account for alphas that range significantly in magnitude
    plt.xscale('log')

    # add dashed vertical line at the alpha that cross validation selects
    plt.axvline(lasso_results['alpha'], linestyle = '--', color = 'black')

    # add plot title
    plt.title("Visualization of Coefficient Path")
    
    # display the plot
    plt.show()

#################################################################################################
def visualize_coef_selection(features: pd.Series, coefs: pd.Series) -> None:
    '''
    Function that creates a bar graph showing all non-zero coefficients sorted from least to 
    greatest

    Parameters:
    -----------
        features (Series): pandas Series containing the names of each feature in the dataset

        coefs (Series):    pandas Series containing the coefficients associated with each
                           feature in the dataset

    Returns:
    -----------
        None
    '''
    # create pandas DataFrame pairing each feature with its coefficient
    df = pd.DataFrame({'feature': features, 'coef': coefs})

    # keep only nonzero coefficients and sort remaining coefficients by signed value
    df = df[df['coef'] != 0].sort_values('coef')

    # create bar plot showing relative magnitude of each coefficient
    bargraph = sns.barplot(data = df, x = 'coef', y = 'feature')

    # add data labels to bar graph
    for container in bargraph.containers:
        bargraph.bar_label(container, fmt = '{:,.2f}', padding = 3) # type: ignore
    
    # update x-axis and y-axis labels
    plt.xlabel('Coefficient (in Standardized Units)')
    plt.ylabel('Feature')

    # add plot title
    plt.title("Nonzero Lasso Coefficients")

    # display the plot
    plt.show()

#################################################################################################
def main():
    # load housing data
    df = load_data()

    # initialize variable to store dependent variable name
    dep_var = 'MedHouseVal'
    
    # generate correlation matrix for dataset features
    visualize_data(df, dep_var)

    # split dataset into training and testing subsets
    x_train, x_test, y_train, y_test = split_data(df, dep_var)

    # scale feature training and testing datasets
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    
    # run lasso regression and return relevant statistics
    lasso_results = run_lasso(x_train_scaled, x_test_scaled, y_train, y_test)

    # display model performance
    print(f"Test R-squared: {lasso_results['R2']:0.3f}")

    # visualize coefficient path across alphas
    visualize_coef_path(x_train_scaled, y_train, lasso_results)

    # visualize coefficient selection
    visualize_coef_selection(lasso_results['features'], lasso_results['coefs'])

if __name__ == "__main__":
    main()
