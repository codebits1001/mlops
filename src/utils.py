import os
import sys
import pickle
# import dill

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_and_load(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train,X_test,y_test,model, param):
    try:
        
        report = []
        gs = GridSearchCV(model, cv = 3, param_grid= param, scoring = 'neg_mean_squared_error')
        gs.fit(X_train, y_train)
        
        # Do prediciton:
        prediction = gs.predict(X_test)
        r2 = r2_score(y_test, prediction)
        print(gs.best_score_)
       
        print(r2)
        report = r2
        return report
        
        
    
        
    
    except Exception as e:
        raise CustomException(e, sys)
    



            