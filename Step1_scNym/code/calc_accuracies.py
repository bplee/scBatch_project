from run_scnym import *
import os


x_dim = 784
train_patient = 2

model_dir = "./201103_single_train_domain_tests"

os.chdir(f"{model_dir}/train_domain_{train_patient}")

trained_models = os.listdir()
print(f"trained models: {trained_models}")

models_calculated = []
accuracies = []
for test_patient in range(6):
    if train_patient == test_patient:
        continue
    model_name = f"scnym_test_domain_{test_patient}_train_domain_{train_patient}"
    
    if model_name not in trained_models:
        print(f"{model_name} not found in {trained_models}")
        continue
    print(f"Current model name: {model_name}")
    models_calculated.append((train_patient, test_patient))
    adata, d = get_Rcc_adata(test_patient, train_patient, x_dim, True)
    try:
        predict_from_scnym_model(adata, model_name)
        accur, weighted_accur = get_accuracies(adata)
        models_calculated.append((train_patient, test_patient))
        accuracies.append((accur, weighted_accur))
        print(f"Train Patient: {train_patient}\tTest Patient: {test_patient}\n Accuracy: {accur}\n Weighed Accuracy: {weighted_accur}")
    except:
        print("Some error occured, not breaking loop.")
for i, patients in enumerate(models_calculated):
    train_patient, test_patient = patients
    accur, weighted_accur = accuracies[i]
    print(f"Train Patient: {train_patient}\tTest Patient: {test_patient}\n Accuracy: {accur}\n Weighed Accuracy: {weighted_accur}")
