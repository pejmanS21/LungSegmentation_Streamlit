import os 

def mlflow_server(model_name):
    os.system("../Records/" + f"{model_name}" + "/mlflowui.sh")

