import time
import os
import sys
import numpy as np
import pandas as pd

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

class PdRccAllData:
    """
    this class will load the pkl pandas file from ../Step0_Data/data/
    into a pandas df of raw counts and the last two columns are the 'patient' and 'cell_type'
    strings

    same_cell_types is used for only getting cells such that each patient has the same label set
    """
    def __init__(self, train=True, take_cell_label_intersection=True,
                 pkl_path='/data/leslie/bplee/scBatch/Step0_Data/data/201002_6pat_proto4_raw_counts.pkl'):
        self.pkl_path = pkl_path
        self.train = train
        self.take_cell_label_intersection = take_cell_label_intersection

        self.init_time = time.time()
        self.data = self._load_data()
        if self.take_cell_label_intersection:
            print("Subsetting for labels that are present in every patient")
            self.data = self.ssl_label_data_clean(self.data)
        self.load_time = time.time()
        print(f"Loading time: {self.load_time - self.init_time}")

    @staticmethod
    def _create_pkl(pkl_path):
        print("Importing necessary libraries...")
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        readRDS = robjects.r['readRDS']  # this is a function
        pandas2ri.activate()  # this is so readRDS loads into pandas df's

        print('Loading annotations...')
        annot = readRDS('/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds')
        print("Loading raw counts...")
        raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds').transpose()

        cell_labels = np.array(annot.cluster_name)
        patient_labels = np.array(annot.Sample)
        gene_names = raw_counts.columns.values  # np array

        # making dataframe to save
        data = pd.DataFrame(raw_counts)
        data['cell_type'] = cell_labels
        data['patient'] = patient_labels

        print(f"Saving .pkl to {pkl_path}")
        data.to_pickle(pkl_path, protocol=4)
        return data

    def _load_data(self):
        if not os.path.isfile(self.pkl_path):
            print(f"Could not find pkl file: {self.pkl_path}")
            # saving new pkl_path
            self.pkl_path = '/data/leslie/bplee/scBatch/Step0_Data/data/temp_6pat_raw_counts.pkl'
            return self._create_pkl(self.pkl_path)
        else:
            rtn = pd.read_pickle(self.pkl_path)
            return rtn

    @staticmethod
    def ssl_label_data_clean(data_df):
        """
        Removes data such that every SSL training/test set have the same labels
            ie. taking the

        Parameters
        ----------
        data_df

        Returns
        -------
        pandas df, where the above condition is true
        """
        patients = np.unique(data_df.patient)
        cell_types_to_keep = set(data_df.cell_type)
        init_num_labels = len(cell_types_to_keep)
        for pat, pat_df in data_df.groupby("patient"):
            cell_types_to_keep = cell_types_to_keep.intersection(set(pat_df.cell_type))
        print(f"Reducing total number of labels from {init_num_labels} to {len(cell_types_to_keep)}.")
        bool_subset = data_df.cell_type.isin(cell_types_to_keep)
        return data_df[bool_subset]

    def get_label_counts(self):
        return data_obj.data[["patient", "cell_type"]].value_counts(sort=False).to_frame().pivot_table(index="patient",
                                                                                                columns="cell_type")

if __name__ == "__main__":

    data_obj = PdRccAllData()
    
    # readRDS = robjects.r['readRDS']
    # pandas2ri.activate()
    # print('Loading annotations...')
    # annot = readRDS('/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds')
    # print("Loading raw counts...")
    #     # raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds')
    # raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds').transpose()
    # cell_types = np.unique(annot.cluster_name)
    # cell_labels = np.array(annot.cluster_name)
    # patient_labels = np.array(annot.Sample)
    # gene_names = raw_counts.columns.values # np array
    # rtn = pd.DataFrame(raw_counts)
    # rtn['cell_type'] = cell_labels
    # rtn['patient'] = patient_labels

    # rtn.to_pickle('/data/leslie/bplee/scBatch/Step0_Data/data/200930_6pat_raw_counts.pkl')

    # from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

    # getting training and testing data
    TEST_PATIENT = 4
    X_DIM = 16323# 784 is the magic number for DIVA; 16323 is the max
    X_DIM = 784

    # getting training and testing data
    # data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)

    # new_data_obj = NewRccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True)

    annot_filepath = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds'
    # new_batch_corrected_data = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_June2020.rds'
    raw_counts_filepath = '/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds'

    # annot = readRDS(annot_filepath)
    # raw_counts = readRDS(raw_counts_filepath) # this list of counts was generated by opening the seurat obj in R and saving the raw counts to a df for saveRDS
    # raw_counts = readRDS(raw_counts_filepath).transpose()
    # test = pyreadr.read_r(raw_counts_filepath)[None].transpose()

    # so this contains 31 labels instead of 16

    #data = readRDS(new_batch_corrected_data)

    
