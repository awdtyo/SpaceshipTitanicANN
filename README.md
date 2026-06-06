# SpaceshipTitanicANN
This project aims to predict whether passengers on the Spaceship Titanic were transported to an alternate dimension, based on a dataset containing various personal and travel-related features. An Artificial Neural Network (ANN) model is developed using TensorFlow/Keras, preceded by a robust data preprocessing pipeline implemented with sklearn.pipeline and sklearn.compose.
## Data Preprocessing
The preprocessing pipeline handles feature engineering, missing value imputation, and categorical feature encoding. The key steps include:

1.  **Feature Engineering:**
    *   **Group Size:** Extracted from `PassengerId` by grouping passengers with the same `gggg` prefix (indicating a group). The size of each group is then calculated.
    *   **Cabin Details:** `CabinDeck` and `CabinSide` are extracted from the `Cabin` feature. Missing cabin values are temporarily filled with a placeholder to allow splitting, then reverted to `NaN` for proper imputation.
    *   **Total Spent:** A new feature, `TotalSpent`, is engineered by summing expenditures across `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`.

2.  **Feature Removal:** Irrelevant text identifiers such as `PassengerId`, `Cabin`, and `Name` are dropped from the dataset.

3.  **Target Transformation:** The `Transported` boolean target variable is converted to a numerical representation (1 for `True`, 0 for `False`).

4.  **Data Splitting:** The dataset is split into training, validation, and test sets with a 70%/15%/15% ratio, ensuring stratification on the `Transported` variable to maintain class balance.

5.  **Pipelined Transformations using `ColumnTransformer`:**
    *   **Categorical Features:** `HomePlanet`, `CryoSleep`, `VIP`, `CabinDeck`, and `CabinSide` are identified as categorical. These undergo `SimpleImputer` with a `most_frequent` strategy, followed by `OneHotEncoder` (dropping the first category to avoid multicollinearity, and `sparse_output=False`).
    *   **Numerical Features:** `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `GroupSize`, and `TotalSpent` are identified as numerical. These are imputed using `SimpleImputer` with a `median` strategy, then scaled using `StandardScaler`.

The `ColumnTransformer` (`preprocessor`) is fitted exclusively on the training data and then used to transform the training, validation, and test sets. The resulting processed datasets are cast to `float32` for compatibility with TensorFlow.

## Model Architecture
The Artificial Neural Network (ANN) is constructed using `tf.keras.Sequential`:

*   **Input Layer:** A `Dense` layer with 64 units and `relu` activation, expecting an `input_shape` dynamically determined by the number of features after preprocessing (`X_train.shape[1]`).
*   **Dropout Layer 1:** A `Dropout` layer with a rate of 0.2 to prevent overfitting.
*   **Hidden Layer:** A `Dense` layer with 32 units and `relu` activation.
*   **Dropout Layer 2:** Another `Dropout` layer with a rate of 0.2.
*   **Output Layer:** A `Dense` layer with 1 unit and `sigmoid` activation, suitable for binary classification.

## Results Visualization
Training and validation loss and accuracy curves are plotted against epochs to visually assess model convergence, identify overfitting/underfitting, and understand performance trends over the training duration. These plots are generated using `matplotlib.pyplot`.

## Persistence
The `preprocessor` object, encapsulating all data transformation logic (imputers, encoders, scalers), is saved to `preprocessor.pkl` using `pickle`. This ensures that the exact same preprocessing steps can be applied to new, unseen data, maintaining consistency between training and inference.
