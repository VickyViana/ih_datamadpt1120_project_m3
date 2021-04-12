# ih_datamadpt1120_project_m3
Ironhack Madrid - Data Analytics Part Time - March 2021 - Project Module 3


# **Kaggle competition: What price is this diamond?**

The participation in the kaggle competition 'dapt202011mad - Predict diamond prices!' requires to create a machine learning model that could predict the price of diamonds depending on theis properties. The model that gets the most accurate results will be the winner.

In this repository it will be explained how the model is designed step by step.

<p align="center"><img src="https://www.mtievents.com/wp-content/uploads/2019/04/competition-in-the-workplace-illustration.jpg"></p>

## **Data to train** 

The file diamonds_train.csv is provided to use it to train the model. It consists of 40454 rows (different diamonds) and 10 columns that represent the diamonds properties. These columns are: carat, cut, color, clarity, depth, table, price, x, y and z. These features and their influence in diamonds price are explained in this repository [Link](https://github.com/VickyViana/ih_datamadpt1120_project_m2). 
For the prediction model some modifications have been made to this dataset:

-Two new columns have been created:
	-Column 'volume': is calculated as the multiplication of the three dimensions of the diamond (x, y and z)
	-Column 'bright_relation': is calculated as the division between the table and the depth. This factor could be important as it indicates the type of bright the diamond will have.

-Elimination of outliers: The outliers of columns 'volume' and 'bright_relation' are removed, as they can influence wrongly in the training model.

-Columns x, y and z will not be take into account, as they will be represented by the column 'volume'.


## **Data to predict** 

The file diamonds_test.csv includes the diamonds which price is going to be predicted. It consists of 13484 rows of diamonds and the same columns as diamonds_train.csv. The columns 'volume' and 'bright_relation' have been added too, as calculated in train dataset.


## **Prediction model** 

For the preprocessing part of the data, a scikit-learn pipeline transformer is used.
First of all the columns are categorized: 

-Columns 'carat', 'depth', 'table, 'volume' and 'bright_relation' are considered as numerical columns. A preprocessing transformer is applied to them implementing a SimpleImputer for the missing values (introducing the mean of the column) and RobustScaler for the scaling.

-Columns 'cut', 'color' and 'clarity' are considered as categorical columns. Column 'price' will be the target of the model. A preprocessing transformer is applied to them implementing a SimpleImputer for the missing values (introducing the constant "missing") and encoding the strings to integers with OrdinalEncoder.

Both preprocessing transformers are join in one ColumnTransformer, called preprcessor.
The model is defined with a pipeline, using the previous explained preprocessor and a regressor, that in this case the chosen one has been LGBMRegressor, that seem to give the better results.

After consider this transformation to the columns and define the prediction model, the dataset is split in train part and test part, and these parts are used to train the model. 

The following step is to check how good is the model. The first check is done with mean_squared_error, and the following results are obtained:
-test error: 551.94
-train error: 475.51
These are not bad results as both values are prety near to 0.

A second check with cross validation (from sklearn) is performed, considering as scoring 'neg_root_mean_squared_error' The mean of the scores obtained is 537.39, an acceptable score.

##**Model optimization**

The scores obtained in the previous checks are not bad, but after some optimization process they could be better. An optimization of the hyper parameters of the model with grid search is performed. The parameter grid is defined with the following variables:
-preprocessor__num__imputer__strategy = ['mean', 'median']
-regressor__n_estimators = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
-regressor__max_depth = [2, 4, 8, 16, 32, 64, 128]
And the grid search is defined with RandomizedSearchCV class and the following parameters:
-cv = 10
-verbose = 10
-scoring= neg_root_mean_squared_error
-n_iter = 50

After this optimization, the best parameters obtained are:
-regressor__n_estimators = 256
-regressor__max_depth = 16
-preprocessor__num__imputer__strategy = median

And the best score is 528, what considerably improves the scores obtained before.

As this model has the best score of 25 different models proved, is the chosen one to get the predicted prices of diamonds in the dataset 'diamonds_test.csv' for submission.


## **Technology stack**

- **Programming Language**: Python 3.8
- **Libraries in Use**: pandas, numpy, matplotlib, seaborn, sklearn and lightgbm.



## **Folder structure**
```
└── ih_datamadpt1120_project_m2
    ├── .gitignore
    ├── README.md
    ├── Model
    │   ├── Diamonds_kaggle_17.ipynb
    │   └── diamonds_prediction_17
    └── dapt202011mad
        ├── sample_submission.csv
        ├── diamonds_test.csv
        └── diamonds_train.csv

:gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: 
:gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem: :gem:
:gem: :gem: :gem: :gem: :gem: :gem:


