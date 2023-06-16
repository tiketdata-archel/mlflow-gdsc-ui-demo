# MLFLOW-GDSC-UI-DEMO

## How to Use
1. Create new environment and install the dependencies
```
pip install -r requirements.txt
```
2. Since MLflow can setup a local server to host your experiments, you can fire it up by typing this in your terminal (make sure you are inside the `MLFLOW-GDSC-UI-DEMO` directory):
```
mlflow ui
```
3. Open another terminal and make sure you are inside the `MLFLOW-GDSC-UI-DEMO` directory. To submit an experiment run:
```
python train.py
```
4. To load one of the models from your run directly:
```
python predict.py
```