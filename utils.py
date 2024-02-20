import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
color_set1 = [
    [8,51,110],
    [16,92,164],
    [56,136,192],
    [104,172,213],
    [170,207,229],
    [210,227,243],
    [244,249,254]
]
color_set1 = [[i/256 for i in color] for color in color_set1]
color_set2 = [
    [78,98,171],
    [70,158,180],
    [135,207,164],
    [203,233,157],
    [245,251,177],
    [254,232,154],
    [253,185,106]
]
color_set2 = [[i/256 for i in color] for color in color_set2]

color_set = color_set1

def convert_to_period(y,period ,info = False): # convert the data into yearly data
    yearly_data = []
    for i in range(0, len(y), period):
        yearly_data.append(y[i:i+period])
    yearly_data = np.array(yearly_data)
    if info:
        print("Yearly data shape: ", yearly_data.shape)
    return yearly_data

def load_data(data, target, info = False):
    y = data[target]
    y = y.values
    if info:
        print("processing lake", target, "with shape: ", y.shape)
    return y

def STL_decompose(y, info = False, fig = False):
    stl = STL(y, period = 12, robust = True)
    res = stl.fit()
    if info:
        print("Seasonal component shape: ", res.seasonal.shape)
    return res

def plot_decompose(y, res, target, info = False,save = False):
    if info:
        print(target, "trend shape: ", res.trend.shape)
        print(target, "seasonal shape: ", res.seasonal.shape)
        print(target, "residual shape: ", res.resid.shape)
    fig, ax = plt.subplots(4, 1, figsize = (10, 10))
    ax[0].plot(y)
    ax[0].set_title(f"Original data of {target}")
    ax[1].plot(res.trend)
    ax[1].set_title(f"Trend of {target}")
    ax[2].plot(res.seasonal)
    ax[2].set_title(f"Seasonal of {target}")
    ax[3].scatter(np.arange(0,len(res.resid)),res.resid, alpha = 0.5,marker = ".")
    ax[3].set_title(f"Residual of {target}")
    if save:
        plt.savefig(f"../pics/{target}_decompose.png")
    plt.subplots_adjust(hspace=0.6)
    plt.show()
    return fig

def decompose_node(data,target, info = False, save = False):
    y = load_data(data, target, True)
    res = STL_decompose(y, True, True)
    fig = plot_decompose(y,res, 'Superior', info, save)
    return res, fig 

def func1(x, A1, A2, w, b1, b2):
    return A1*(np.cos(A2*np.cos(w*x + b1)+b2))

def func2(x, a0, a1, a2, a3, a4):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4



def fit_curve(y, period,f, info = False, plot = False):
    para, cov =  curve_fit(f, np.arange(0,period), y, maxfev=10000)
    y_ = f(np.arange(0,period), *para)
    #calculate the mean square error 
    mse = np.mean((y - y_)**2)
    if info:
        print("Parameter: ", para)
        print("Covariance: ", cov)
    if plot: 
        plt.plot(np.arange(0,period), y, label = "Original")
        plt.plot(np.arange(0,period), y_, label = "Fitted")
        plt.legend()
        plt.show()
    return para, cov, mse

def fit_node(res, period, f, info = False, plot = False ):
    y = res.seasonal
    yearly_data = convert_to_period(y, period, info)
    para = []
    cov = []
    mse = []
    for i in range(yearly_data.shape[0]):
        p, c, m = fit_curve(yearly_data[i], period, f, info, plot)
        para.append(p)
        cov.append(c)
        mse.append(m) # function fitting mse
    return np.array(para), np.array(cov), np.array(mse)

def plot_parameters(p, together = False):
    if together:
        p = np.array(p)
        fig = plt.figure()
        plt.plot(p[:,0], color = color_set[0],marker = '.')
        plt.plot(p[:,1],color = color_set[1], marker = '.')
        plt.plot(p[:,2], color = color_set[2], marker = '.')
        plt.plot(p[:,3],color = color_set[3], marker = '.')
        plt.plot(p[:,4],color = color_set[4], marker = '.')
        plt.legend(['a0', 'a1', 'a2', 'a3', 'a4'])
    else:
        p = np.array(p)
        fig, ax = plt.subplots(p.shape[0],1,figsize=(10,10))
        ax[0].plot(p[:,0], color = color_set[0],marker = '.')
        ax[0].set_title("a0")
        ax[1].plot(p[:,1], color = color_set[1],marker = '.')
        ax[1].set_title("a1")
        ax[2].plot(p[:,2], color = color_set[2],marker = '.')
        ax[2].set_title("a2")
        ax[3].plot(p[:,3], color = color_set[3],marker = '.')
        ax[3].set_title("a3")
        ax[4].plot(p[:,4], color = color_set[4],marker = '.')
        ax[4].set_title("a4")


shift = 5
split = 90
color_set = color_set2
def trainer(p, model, res, f, period,  plot = False, info = False):
    X = []
    p = np.array(p)
    for i in range(shift, len(p)):
        X.append(p[i-shift:i].flatten())
    X = np.array(X)
    y = p[shift:]
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    y_real = convert_to_period(res.seasonal, 12)
    y_real = y_real[split+shift:]
    if info :
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)
        print("y_real shape: ", y_real.shape)
    model = model()
    model.fit(X_train, y_train)
    mse_validation = []
    mse_total = []
    paras = []
    
    for i in range(len(y_test)):
        para_predicted = model.predict(X_test[i].reshape(1,-1))
        para_predicted = np.array(para_predicted).flatten()
        paras.append(para_predicted)
        y_predicted = f(np.arange(0,period), *para_predicted.flatten())
        y_predicted = np.array(y_predicted).flatten()
        # append loss
        mse_valid = mean_squared_error(y_test[i], para_predicted)
        mse_validation.append(mse_valid)
        mse = mean_squared_error(y_real[i], y_predicted)
        mse_total.append(mse)
        if plot and i == 0:
            plt.plot(np.arange(0,period), y_real[i], color = color_set2[0], marker = '.',label = "Real")
            plt.plot(np.arange(0,period), y_predicted, color = color_set2[len(color_set)-1], marker = '.',label = "Predicted")
            plt.legend()
            plt.show()
    return paras, mse_validation, mse_total 

def plot_mse(mse, title):
    #box plot
    fig = plt.figure()
    plt.title(title)
    plt.boxplot(mse)
    plt.show()

def trend_data_prepare(res):
    y = res.trend
    X = []
    for i in range(len(y)-shift):
        X.append(y[i:i+shift])
    X = np.array(X)
    y = y[shift:]
    X_train = X[:1215]
    X_test = X[1215:]
    y_train = y[:1215]
    y_test = y[1215:]
    return X_train, X_test, y_train, y_test

def trend_node(res, model):
    X_train, X_test, y_train, y_test = trend_data_prepare(res)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    mse = np.mean((y_predict - y_test)**2)
    return mse, y_predict, y_test





        