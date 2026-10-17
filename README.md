## Introduction

The availability, or lack thereof, of housing in the United States has been widely reported on and studied over the past couple of years [[1]](#references) [[2]](#references). In light of this, we were interested in which features are most important when predicting home prices. While median income and geography are commonly given as the two most meaningful features, we wanted to investigate whether the data supported this assertion. To answer this question, we used a lasso regression to predict median home values across California using the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn. We note that this dataset is from the 1990 U.S. Census and may not be fully reflective of the current California housing market.

## Data Selection, Cleaning, and Manipulation

Our dataset was constructed from 1990 U.S. Census data and contains 20,640 block groups across nine different variables. Our dependent variable was `MedHouseVal`, the median house value by block group. All nine variables and their descriptions are listed below. 

| Variable | Description |
|---|---|
|`MedHouseVal` | Median house value in the block group (in hundreds of thousands of dollars) |
| `MedInc` | Median household income in the block group (tens of thousands of dollars) |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

After loading the dataset directly from scikit-learn, we removed rows with missing values and any rows that were duplicates of other rows. In practice, the dataset contains no missing values or duplicate rows, but we kept the checks in place for completeness. Once cleaning was complete, we generated a heatmap of the feature correlation matrix and checked visually for multicollinearity. From this matrix we observed that the pairs `AveRooms` and `AveBedrms` and `Latitude` and `Longitude` were both highly correlated. This was unsurprising as latitude and longitude are correlated because of California's diagonal shape, and homes with more rooms are likely to have more bedrooms as well. The issue of multicollinearity is discussed in the findings section.

We then split the cleaned dataset into training and testing sets using an 80/20 split and specifying a random state for reproducibility. After this, features were standardized using the training set's mean and standard deviation. This step was particularly important because of our choice of the lasso model, which penalizes each coefficient as if it were measured in the same units. Lack of a common scale would cause the penalty for some features to be unfair and make interpreting the model output difficult.

## Model Selection

We selected the lasso model for two primary reasons: (i) it is a linear model, which makes it easier to interpret and gives each coefficient a direct meaning (i.e., the change in predicted median house value for a one standard deviation change in a specific feature) and (ii) its structure allows it to shrink coefficients to exactly zero, effectively dropping features that don't add predictive value. The magnitude of the penalty term alpha was chosen using 5-fold cross validation on the training set. After cross validation, we created a line graph showing each feature's coefficient at varying values of alpha. This was helpful when comparing the cross-validated choice of alpha against other values.

## Findings

Cross validation selected a very small value for alpha (≈ 0.002), which suggested that only light regularization was needed and left the collinear pairs mostly unaddressed. As a result, all eight features had nonzero coefficients and none were dropped after the model was fitted.

$$
\begin{aligned}
\widehat{\text{MedHouseVal}} = 2.07 &+ 0.81\,\text{MedInc} - 0.88\,\text{Latitude} - 0.85\,\text{Longitude} \\
&+ 0.26\,\text{AveBedrms} - 0.22\,\text{AveRooms} + 0.12\,\text{HouseAge} \\
&- 0.03\,\text{AveOccup} - 0.01\,\text{Population}
\end{aligned}
$$

The fitted model above had an R-squared of 0.59, meaning that it explains 59% of the variance in median house value on the test set. As expected, median income had the largest positive effect on house value - a one standard deviation increase in median income was associated with an ~$81,000 increase in median house value, holding other features constant. Also confirming our initial suspicions were the coefficients on the geography variables, both of which were negative and slightly larger in magnitude than the coefficient on median income. We believe this makes sense, as it implies that home values are higher further south and further west (i.e., toward the coastline and the major Southern California and Bay Area markets). We also noted that `AveRooms` and `AveBedrms` have opposite signs, which is likely because of their high correlation.

## Extensions

Possible extensions for this project include the use of non-linear models, which may better account for the non-linear relationship between location and price (e.g., prices tend to cluster around cities and the coast). Depending on data availability, it would also be interesting to replace the latitude and longitude variables with more granular features like distance from the coast or to the nearest major city. From a data standpoint, we would like to spend more time looking at extreme outliers and whether they could be trimmed from the dataset to improve predictive accuracy. Additionally, we could explore log transforms of house value to combat error term heteroskedasticity (e.g., errors on a $500,000 home are far larger than errors on a $150,000 home).

## References

1. DeSilver, D. (2024, October 25). *A look at the state of affordable housing in the U.S.* Pew Research Center. https://www.pewresearch.org/short-reads/2024/10/25/a-look-at-the-state-of-affordable-housing-in-the-us/
2. Center on Budget and Policy Priorities. (2026, June 10). *Addressing the housing affordability crisis requires increasing housing supply and expanding rental assistance.* https://www.cbpp.org/research/housing/addressing-the-housing-affordability-crisis-requires-increasing-housing-supply-and

## Author

Thomas Costin · [tcostin20@gmail.com](mailto:tcostin20@gmail.com)
