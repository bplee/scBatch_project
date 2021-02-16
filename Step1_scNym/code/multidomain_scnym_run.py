"""
This file is to do a run of the multi-domain functionality.
I made some change to the source code to let this run since there were changes with the API file
"""
import argparse
from new_prescnym_data_load import get_Rcc_adata
from run_scnym import *
from scnym.api import scnym_api

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scNym')
    parser.add_argument('--test_patient', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_patient', type=int, default=None,
                        help='test domain')
    args_scnym = parser.parse_args()
    print(args_scnym)

    print(f"Current Working Dir: {os.getcwd()}")

    train_pat = args_scnym.train_patient
    test_pat = args_scnym.test_patient

    outpath = f"./210202_multi_domain_test_pat_{test_pat}"


    adata = get_Rcc_adata(test_patient=test_pat, x_dim=784)

    adata = adata[0]

    np.unique(adata.obs.batch)
    pats = np.unique(adata.obs.batch)

    # making dict from patient name to new train_i
    # a = {pats[i]: "train_"+str(i) for i in range(len(pats))}
    a = {}
    count = 0
    for i in range(len(pats)):
        if i == test_pat:
            a[pats[i]] = 'target_0'
        else:
            a[pats[i]] = f"train_{count}"
            count += 1

    # # manually setting patient 't4' to be target_0
    # a['t4'] = 'target_0'

    # checking the mapping
    np.array(list(map(lambda x: a[x], np.array(adata.obs.batch))))

    # using dict to map patient names to domain label names
    adata.obs['domain_label'] = np.array(list(map(lambda x: a[x], np.array(adata.obs.batch))))

    # saving the patient names and deleting the names from the adata obj
    # batch = adata.obs.batch
    # batch
    # del adata.obs['batch']


    # running training
    scnym_api(adata=adata, task='train', groupby='annotations',
              domain_groupby='domain_label', out_path= outpath,
              config='no_new_identity')

    predict_from_scnym_model(adata, trained_model=outpath)

    accur, weighted_accur = get_accuracies(adata)

    print(f"Weighted Accur: {weighted_accur}\nUnweighted Accur: {accur}")