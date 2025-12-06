import os
import sys
import pickle
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param = params[model_name]

            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            gs = RandomizedSearchCV(
                model,
                param_distributions=param,
                n_iter=30,
                scoring="balanced_accuracy",
                cv=cv_splitter,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            print(f"Best parameters for {model_name}: {gs.best_params_}")
            print(f"Best CV balanced_accuracy for {model_name}: {gs.best_score_}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
            test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)

            print(f"{model_name} train balanced_accuracy: {train_bal_acc:.4f}")
            print(f"{model_name} test balanced_accuracy:  {test_bal_acc:.4f}")

            report[model_name] = test_bal_acc

        return report

    except Exception as e:
        raise CustomException(e, sys)