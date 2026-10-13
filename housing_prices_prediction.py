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
from sklearn.metrics import mean_absolute_error

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
def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        Y_train (DataFrame): pandas DataFrame containing the training subset
                             for the dependent variable in the California
                             housing dataset
        Y_test (DataFrame):  pandas DataFrame containing the testing subset
                             for the dependent variable in the California
                             housing dataset
    '''
    # create separate DataFrames for the independent and dependent variables
    df_X = df.drop(columns = ['MedHouseVal'])
    df_Y = df[['MedHouseVal']]

    # split the two datasets into training and testing subsets
    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 0)

    return X_train, X_test, Y_train, Y_test

###############################################################################
def main():
    # load housing data
    df = load_data()
    
    # generate correlation matrix for dataset features
    #visualize_data(df)

    # split dataset into training and testing subsets
    X_train, X_test, Y_train, Y_test = train_test(df)

if __name__ == "__main__":
    main()
