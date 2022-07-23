from pip import main
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
import joblib
from yaml import dump

def linear_model1():
    
    data=load_boston()
    x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,random_state=22,test_size=0.2)
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.fit_transform(x_test)

    
    # estimator = Ridge(alpha=1)
    # estimator.fit(x_train, y_train)
    # joblib.dump(estimator, "./data/test.pkl")
    estimator=joblib.load("./data/test.pkl")
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)
    pass




if __name__ == '__main__':
    linear_model1()
    

