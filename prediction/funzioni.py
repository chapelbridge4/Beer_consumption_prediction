import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
# validazione
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# ottimizzazione
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

# Modulo che contiene tutte le funzioni per la predizione e visualizzazione di dati che potrebbero essere rilevanti

def interquartile_range_limits(data: pd.DataFrame):
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)

    iqr = (q75 - q25)
    up_limit = q75 + iqr * 1.5
    low_limit = q25 - iqr * 1.5

    return up_limit, low_limit

def plot_generic_distribution(data: pd.DataFrame, up_limit: float, low_limit: float):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(len(data)), data, alpha=0.5)
    # up limit
    plt.axhline(up_limit, color='brown', linestyle='-', label='up_limit')
    # low limit
    plt.axhline(low_limit, color='brown', linestyle='-', label='low limit')
    plt.legend()
    plt.show()



def normal_distribution_limits(data: pd.DataFrame):
    mean = data.mean()
    std = data.std()

    up_limit = mean + std
    low_limit = mean - std

    return mean, up_limit, low_limit


def plot_normal_distribution(data: pd.DataFrame, mean: float, up_limit: float, low_limit: float):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(len(data)), data, alpha=0.5)
    # terza deviazione standard postitiva
    plt.axhline(up_limit, color='brown', linestyle='-', label='up limit')
    # media
    plt.axhline(mean, color='r', linestyle='-', label='mean')
    # terza deviazione standard negativa
    plt.axhline(low_limit, color='brown', linestyle='-', label='low limit')
    plt.legend()
    plt.show()

def plot_scatter(data):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(len(data)), data, alpha=0.5)
    plt.show()


def plot_scatter_2(x, y):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, alpha=0.5)
    plt.show()


# testa e valida una pipeline generica
def pipeline_validation(pipeline, x, y):
    # split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    # trainging
    pipeline.fit(x_train, y_train)

    # predict
    y_pred = pipeline.predict(x_val)

    # error test
    mse = mean_squared_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)

    scores_df = pd.DataFrame({'rmse': [np.sqrt(mse)], 'r2': [r2]})
    print(scores_df)
    print("\n")

def plot_learning_curve(pipeline, x, y):

  x_len = len(x)
  train_size_60 = int((x_len * 60) / 100)
  train_size_70 = int((x_len * 70) / 100)
  train_size_80 = int((x_len * 80) / 100)

  train_sizes = [train_size_60, train_size_70, train_size_80]

  train_size_abs, train_scores, valid_scores = learning_curve(pipeline,
                                                           x,
                                                           y,
                                                           train_sizes = train_sizes)

  train_mean = np.mean(train_scores, axis=1)
  valid_mean = np.mean(valid_scores, axis=1)


  scores_df = pd.DataFrame({'train': train_mean, 'validation': valid_mean})
  print(scores_df)

  plt.figure(figsize=(12, 8))
  plt.plot(train_size_abs, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
  plt.plot(train_size_abs, valid_mean, color='red', marker='o', markersize=5, label='CrossValidation Accuracy')
  plt.legend()
  plt.show()


def test_poly(x_train, x_test, y_train, y_test, n):
  for n in range(1, n):
    # [1] generatore di feature polinominali
    polyfeatures = PolynomialFeatures(degree=n)
    x_train_poly = polyfeatures.fit_transform(x_train)
    x_test_poly = polyfeatures.transform(x_test)


    # [2] stnadardizzazione del set di training e del set di test
    scaler = StandardScaler()
    x_train_poly_std = scaler.fit_transform(x_train_poly)
    x_test_poly_std  = scaler.transform(x_test_poly)


    # [3] eseguo la regressione lineare
    reg = LinearRegression()
    reg.fit(x_train_poly_std, y_train)


    # [4] calcolo la predizione sul modello addestrato
    y_pred = reg.predict(x_test_poly_std)


    # [5] test
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print("Grado:", n, " - MSE:", mse, " - R2:", r2)



# visualizza gli errori accumulati durante
# un processo di validazione iterativo
def plot_errors_curves(mse_errors, r2_errors, y):
  max_y = np.max(y)
  min_y = np.min(y)
  plt.figure(figsize=(12, 8))
  plt.axhline(max_y, color = 'brown', linestyle = '-', label = f'max y={max_y}')
  plt.plot(np.sqrt(mse_errors), "b", label= 'rmse')
  plt.axhline(min_y, color = 'brown', linestyle = '-', label = f'min y={min_y}')
  plt.legend()
  plt.show()

  plt.figure(figsize=(12, 8))
  plt.plot(r2_errors, "g", label="r2")
  plt.legend()
  plt.show()



# testa e valida una pipeline con regressione polinominale
def pipeline_poly_validation(regressor, x, y, n):
  # split
  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

  mse_errors = np.zeros(n - 1)
  r2_errors  = np.zeros(n - 1)


  for i in range(1,n):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree = i), regressor)

    # training
    pipe.fit(x_train, y_train)

    # predict
    y_pred = pipe.predict(x_val)

    # error test
    mse = mean_squared_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)

    print("Grado:", i, " - MSE:", mse, " - R2:", r2)
    mse_errors[i - 1] = mse
    r2_errors[i - 1] = r2

  plot_errors_curves(mse_errors, r2_errors, y_val)

def ottimization(x, y):
  # split
  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

  pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

  grid_parameter = {
      'polynomialfeatures__degree': [1, 2, 3, 4],
      'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1., 10.],
      'ridge__fit_intercept': [True, False]
  }

  grid_search = GridSearchCV(pipe, grid_parameter)
  grid_search.fit(x_train, y_train)

  print("MIGLIOR SCORE: ", grid_search.best_score_)
  print("I MIGLIORI IPERPARAMETRI: ", grid_search.best_params_)

  return grid_search.best_params_["polynomialfeatures__degree"], \
      grid_search.best_params_["ridge__alpha"], \
      grid_search.best_params_["ridge__fit_intercept"]


def ottimization_lasso(x, y):
    # split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Lasso())

    grid_parameter = {
        'polynomialfeatures__degree': [1, 2, 3],
        'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1., 10.],
        'lasso__fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(pipe, grid_parameter)
    grid_search.fit(x_train, y_train)

    print("MIGLIOR SCORE: ", grid_search.best_score_)
    print("I MIGLIORI IPERPARAMETRI: ", grid_search.best_params_)

    return grid_search.best_params_["polynomialfeatures__degree"], \
           grid_search.best_params_["lasso__alpha"], \
           grid_search.best_params_["lasso__fit_intercept"]


def test_pipeline_ridge(x, y, polyDegree, maxAlpha):
    # split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    mse_errors = np.zeros(maxAlpha - 1)
    r2_errors = np.zeros(maxAlpha - 1)

    for i in range(1, maxAlpha):
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=polyDegree), Ridge(alpha=i))

        # training
        pipe.fit(x_train, y_train)

        # predict
        y_pred = pipe.predict(x_val)

        # error test
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print("Grado:", i, " - MSE:", mse, " - R2:", r2)
        mse_errors[i - 1] = mse
        r2_errors[i - 1] = r2

    plot_errors_curves(mse_errors, r2_errors, y_val)

def test_pipeline_lasso(x, y, polyDegree, maxAlpha):
    # split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    mse_errors = np.zeros(maxAlpha - 1)
    r2_errors = np.zeros(maxAlpha - 1)

    for i in range(1, maxAlpha):
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=polyDegree), Lasso(alpha=i))

        # training
        pipe.fit(x_train, y_train)

        # predict
        y_pred = pipe.predict(x_val)

        # error test
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print("Alpha:", i, " - MSE:", mse, " - R2:", r2)
        mse_errors[i - 1] = mse
        r2_errors[i - 1] = r2

    plot_errors_curves(mse_errors, r2_errors, y_val)

def test_plot(test, pred):
  # andamento valori reali e valori predetti
  plt.figure(figsize=(10, 8))
  plt.plot(np.arange(test.size), test,  alpha=0.8, label="valori reali")
  plt.plot(np.arange(pred.size), pred, alpha=0.8, label="valori predetti")
  plt.xlabel('Numero di osservazioni')
  plt.ylabel('Valori')
  plt.legend()
  plt.show()

  # andamento media mobile valori reali e valori predetti
  r = 20
  maverage_y_test = pd.Series(test).rolling(r).mean()
  maverage_y_pred = pd.Series(pred).rolling(r).mean()

  plt.figure(figsize=(10, 8))
  plt.plot(np.arange(maverage_y_test.size), maverage_y_test,  alpha=0.8, label="valori reali")
  plt.plot(np.arange(maverage_y_pred.size), maverage_y_pred, alpha=0.8, label="valori predetti")
  plt.xlabel('Numero di osservazioni')
  plt.ylabel('Valori')
  plt.show()

  # varianza tra valori reali e valori predetti
  plt.figure(figsize=(10, 8))
  plt.scatter(test, pred)
  plt.show()





