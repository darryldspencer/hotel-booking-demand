# hotel-booking-demand
Simple project to demonstrate data cleaning, descriptive statistics, and visualization. Data taken from the [Kaggle website.](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
The data dictionary describing the data fields can be found in the original [paper.](https://www.sciencedirect.com/science/article/pii/S2352340918315191) 

## Data Cleaning

### Wrong Data Types
The variables Agent, Children, and Company were imported as floats and needed to be converted to integers. 

### Imputing Missing Values
Out of 119,390 rows, Agent had 16,340 missing values, Children had 4 missing values, Country had 488 missing values, and Company had 112,593. 
The paper says that NULL values for Agent or Company ID means not applicable (i.e., there was no agent or company). Since 0 is unused, an ID of 0 will mean not applicable.
For Children, missing values will be imputed using the median value, 0.
For Country, the unused category "UNK" for UNKNOWN will be used.

