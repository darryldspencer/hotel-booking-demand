# hotel-booking-demand
Predicts which hotel reservations will be canceled. The data was taken from the [Kaggle website](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
The data dictionary describing the data fields can be found in the original [paper](https://www.sciencedirect.com/science/article/pii/S2352340918315191).

## Skills Demonstrated
  - Python
  - Data cleaning and wrangling
  - Descriptive Statistics
  - ANOVA analysis
  - Exploratory Data Analysis (EDA)
  - Data Visualization
  - Machine Learning Classification Algorithms: Logistic Regression, Random Forest, Artificial Neural Network, KNN, Naive Bayes, Support Vector Machines, Ensemble Method

It is part of a data science portfolio for Darryl D Spencer (darryl.d.spencer@gmail.com). You can see my professional profile on [LinkedIn](https://www.linkedin.com/in/darryldspencer/).

## The Business Problem - Cancellations
Reservation cancellations cost hotels money. Hotel reservations are an opportunity cost. A room reserved by one customer cannot be sold to another. If a customer cancels the reservation, the hotel risks being unable to rent it out in time to another customer. In this data set, about 37% of reservations are canceled. If the hotels could predict which reservations were likely to cancel, they might use some risk mitigation strategies such as charging a non-refundable deposit or overbooking a certain percentage of the rooms. This notebook will create and evaluate a predictive model for booking cancellations.


A description of the steps and findings from the Jupyter Notebook is as follows:

## Data Cleaning

### Wrong Data Types
The variables Agent, Children, and Company were imported as floats and needed to be converted to integers. 

### Imputing Missing Values
Out of 119,390 rows, `agent` had 16,340 missing values, `children` had four missing values, `country` had 488 missing values, and `company` had 112,593. 
The paper says that NULL values for `agent` or `company` ID are not applicable (i.e., there was no agent or company). Since 0 is unused, an ID of 0 will mean not applicable.
For `children`, missing values will be imputed using the median value, 0.
The unused category "UNK" for UNKNOWN will be used for `country`.

### Fixing Data Types & Data Transformations
  - The variables `agent` and `company` required casting to integer types because they are IDs.
  - The variable `children` was cast from float to integer because you can't have fractional children.
  - The variable `lead_time` was transformed by log1p to `lead_time_log1p` to make it closer to a normal distribution.
  - The numeric variables `agent`, `babies`, `children`, and `company` were transformed into logical categories `has_agent`, `has_babies`, `has_children`, and `has_company`.

### ANOVA Findings

The ANOVA table below shows that all the categorical variables showed statistically significant different means between category values so all were submitted to an algorithm to choose the best for classification analysis.


| Variable name        |     sum_sq    |      df  |       F     |     PR(>F)     |
|----------------------|---------------|----------|-------------|----------------|
| arrival_date_month   |     88.526834 |     11.0 |   52.108848 |  1.467621e-115 |
| country              |   1630.838226 |    177.0 |   59.657746 |  0.000000e+00  |
| customer_type        |     52.317152 |      3.0 |  112.915111 |  5.179361e-73  |
| deposit_type         |   2018.886281 |      2.0 | 6535.985561 |  0.000000e+00  |
| distribution_channel |    10.673845  |      4.0 |   17.277867 |  3.528990e-14  |
| market_segment       |   395.440498  |      7.0 |  365.773575 |  0.000000e+00  |
| meal                 |    31.369759  |      4.0 |   50.778563 |  8.751435e-43  |
| reserved_room_type   |    28.736823  |      9.0 |   20.674045 |  2.927160e-35  |
| booking_changes      |   109.033169  |      1.0 |  705.972617 |  4.270900e-155 |
| is_repeated_guest    |    63.762457  |      1.0 |  412.851879 |  1.256144e-91  |
| has_agent            |    30.285927  |      1.0 |  196.096613 |  1.610768e-44  |
| has_company          |     3.590474  |      1.0 |   23.247754 |  1.425919e-06  |
| has_children         |     8.940124  |      1.0 |   57.885899 |  2.797975e-14  |
| has_babies           |     7.374200  |      1.0 |   47.746787 |  4.873997e-12  |
| Residual             | 18404.462520  | 119166.0 |         NaN |           NaN  |

### Variable Selection Algorithm
- Eliminate the following variables:
  - `reservation_status_date`: Removed due to its date format, leading to too many categories.
  - `reservation_status`: Excluded as it contains a redundant category value of is_canceled.
  - `assigned_room_type`: Excluded because it is not known until the customer shows up, making it unsuitable for prediction.
  - `agent` and `company`: Removed due to their numerous ID values. Instead, we will use has_agent and has_company.
  - `lead_time`: Dropped in favor of lead_time_log1p.
  - `deposit_type`: Likely a value created by the hotel, making it unnecessary for prediction.
- One hot encode the remaining categorical variables
- Measure variable goodness by their impact on logistic regression AIC fit value p-values
- Incrementally add one variable at a time based upon which one increased the AIC fit the most and had a p-value > 0.02.
- To reduce algorithm time, do this in two matches. All variables except country codes in one and county codes in the second.
  - At the end of each batch, refit and remove any variables one at a time, in order of largest p-values, that still have p-values > 0.02, refitting each time.
  - This can happen if a variable added later is too co-linear with a previously added variable.
- Merge the two batches and eliminate variables with p-values > 0.02 as described above.

The remaining list had 54 variables.

### Classification Algorithm Performance Summary
- The random forest classifier had the highest accuracy at 86.6%.
- Naive Bayes had the lowest accuracy at 63.6% and was the only one with an accuracy below 80%.
- The unweighted ensemble method, excluding naive Bayes, achieved 84.9%.

Confusion Matrix
[[13753  1280] [ 1908  6937]]

Classification Report for the Random Forest

|    Label    | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
|           0 |      0.88 |  0.91  |    0.90  |   15033 |
|           1 |      0.84 |  0.78  |    0.81  |    8845 |   
|    accuracy |           |        |    0.87  |   23878 |
|   macro avg |      0.86 |  0.85  |    0.85  |   23878 |
|weighted avg |      0.87 |  0.87  |    0.87  |   23878 |


### Conclusions and Recommendations for the Hotel
## Recommendations for the Hotel
1. Use a model to predict room cancellations and incorporate that into your pricing or deposit policy. Each model achieved an accuracy much better than random guessing.
2. The Random Forest algorithm performed the best. The variable, `feature_importances`, contains the ordered list in descending order of which variables were the most important in the decision-making process.
3. If more precision information is desired on the impact of each variable, the coefficients in the logistic regression give you the [log-odds](https://www.statisticshowto.com/log-odds/) impact of each one.
