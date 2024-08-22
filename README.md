# MSU-ML-Prac

This repository is devoted to **Machine Learning MSU Practicum**.

**Mastered themes:**
1. Handling **tabular data** using the **Pandas** library, visualization using the **Matplotlib** library, **Seaborn**, **Plotly**,
2. **Vector computation** using the **NumPy** library,
3. **K Nearest Neighbors (KNN)** algorithm for solving **classification** and **regression** tasks,
4. **Linear models**
    - **Overtraining** experience,
    - **Dealing** with **overtraining**,
    - **Regularization** Techniques,
    - **Regression** issue.
5. **Preprocessing** categorical features:
    - **One-Hot** Encoding,
    - **Count** Encoding.
6. **Support Vector Machine (SVM)**:
    - Plotting of **nonlinear decision boundary**,
    - **Optimal selection** of the **hyperparameter**,
    - **Principal Component Analysis (PCA)** for dimensionality reduction,
    - The **Posterior Probability** for SVM,
    - **Solving ML task**, , the task was solved with the use of ***ensemble learning***.
7. **Decision Trees**:
   > Used to predict the *real estate prices in California*, using **RandomForestRegressor**, **ExtraTreesRegressor** and **LinearSVR**
   - Training, predicting and visualizing **DecisionTreeRegressor**,
   - Improving prediction result using **Ensemble learning**, including testing **stacking**, **bagging** and **boosting** techniques,
   - **Transforming** multidimensional matrix into 1d-vectors,
   - **Pipeline** use to chain multiple estimators into one,
   - **GridSearchCV** use to tune the hyper-parameters of an estimator,
   - **Solving ML tasks**, *predict the value of some energy for each physical potential*, using the ExtraTreesRegressor, the task was solved with the use of ***PotentialTransformer*** and ***data preprocessing*** (centering).
8. **Gradient Boosting**:
   > Used to predict the *price of used cars in a number of countries*, using **XGBoost**, **LightGBM**, **Catboost**, **HyperOpt**,
   - **Dataset preprocess**: missing values replaces with average ones, cells separation, feature selection and encoding
   - **Hyperopt** use to tune the hyper-parameters of an estimator,
   - **Solving ML tasks**, *predict the number of awards for the film*, the task was solved with the use of ***ensemble learning***.
9. **Clusterization**:
   - **Unsupervised machine learning methods** — **clustering** and **dimensionality reduction**.
   - **PyTorch** and **Tensorflow** use,
   - Using the dimensionality reduction algorithms **TSNE**, **UMAP**, **Isomap**, **KernelPCA**,
   - Using **Transfer Learning** to transform to more representative feature space, where objects will be located in a variety that is easier to represent.

## Project Tree
```
.
├── Clustarization
│       └── clusterization.ipynb
├── Decision Trees
│       ├── decision_trees.ipynb
│       ├── decision_trees_ml.py
│       └── decision_trees_unit-tests.py
├── Gradient Boosting
│       ├── gradient_boosting.ipynb
│       └── gradient_boosting_ml.py
├── KNN
│       ├── cross_val.py
│       ├── KNN_2023.ipynb
│       └── scalers.py
├── Linear Models: classification
│       ├── Linear_Models_classification .ipynb
│       └── Task.py
├── Linear Models: regression
│       └── Linear_Models_regression.ipynb
├── numpy-pandas-matplotlib
│       ├── functions.py
│       ├── functions_vectorised.py
│       └── Numpy_pandas_matplotlib.ipynb
├── Python Introduction
│       ├── task15.py
│       ├── task6.py
│       └── task7.py
├── README.md
└── SVM
    ├── SVM.ipynb
    └── svm_solution.py
```


