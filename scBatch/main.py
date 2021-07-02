"""
Main class for trianing a DIVA model, stores the model runs training, testing, creation of figs
"""
import torch
import os

# importing package contents
from . import dataprep
from . import train
from . import visualization
# from . import model

from .model import DIVA
# from .model_2layers import DIVA
# from .model_tanh import DIVA

import numpy as np
import torch.utils.data as data_utils


class DIVAObject:
    def __init__(self, args=None):
        if args is None:
            from .args import default_args
            self.set_args(default_args)
        else:
            # allow user to input their own args file
            self.set_args(args)

        # model architecture is dependent on data dimension so can't set it yet
        self.model = None
        self.device = None
        # data loaders are stored in the obj
        self.train_loader, self.test_loader, self.valid_loader = None, None, None

        # directory where the model output directory will live
        # set in self.fit
        self.outpath = None

        # name of model (used for name of .config file, .model file, and figure names
        self.model_name = None


    def __repr__(self):
        rtn = f"DIVAObject:\n{self.args}"
        if self.model is None:
            return rtn
        else:
            return f"{rtn}\nModel:\n{self.model}"
        # print("Data")
        # if self.train_loader is not None:
        #     print(f" Training set shape: {self.train_loader.train_data.shape}")
        # if self.valid_loader is not None:
        #     print(f" Validation set shape: {self.valid_loader.train_data.shape}")
        # if self.test_loader is not None:
        #     print(f" Testing set shape: {self.test_loader.test_data.shape}")

    def set_data_loaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def create_model_from_args(self):
        self.set_model(DIVA(self.args))

    def load_model_from_args(self):
        print(f" loading model from args: {self.args}")
        self.model = DIVA(self.args)

    def set_args(self, args):
        self.args = args

    def set_args_d_dim(self, value):
        print(f" changing d_dim from {self.args.d_dim} to {value}")
        self.args.d_dim = value

    def set_args_y_dim(self, value):
        print(f" changing y_dim from {self.args.y_dim} to {value}")
        self.args.y_dim = value

    def set_args_x_dim(self, value):
        print(f" changing x_dim from {self.args.x_dim} to {value}")
        self.args.x_dim = value

    def set_args_ssl(self, x):
        if isinstance(x, bool):
            print(f' Setting model ssl to {x}')
            self.args.ssl = x
        else:
            raise TypeError(f"Expecting ssl arg to be bool not {type(x)}")

    def set_outpath(self, name):
        self.outpath = name

    def set_model_name(self, name):
        self.model_name = name

    def set_model(self, model):
        self.model = model

    @staticmethod
    def adata_to_diva_loaders(adata):
        train_loader, test_loader = dataprep.get_diva_loaders(adata,
                                                              domain_name="domain",
                                                              label_name="cell_type",
                                                              shuffle=True)
        new_train_loader, validation_loader = dataprep.get_validation_from_training(train_loader)

        return new_train_loader, validation_loader, test_loader

    def fit(self, adata, model_name=None, outpath="./", ssl=None):
        """

        runs training procedure for n epochs (set in args)

        Parameters
        ----------
        adata : anndata obj
        model_name : str
        outpath : str
        ssl : bool
            if True, will use unsupervised data (as specified in adata.obs.batch)

        Returns
        -------
        None

        """
        if 'batch' not in adata.obs:
            # if no batches specified then ssl is automatically False
            print(f'"batch" not listed as column in anndata obj, running supervised learning on entire adata')
            ssl = False
            adata.obs['batch'] = "0"
        else:
            if ssl is None:
                # if it wasnt set in .fit then use the ssl from the args
                ssl = self.args.ssl
            # otherwise ssl arg was already set, and it will be fine to use
        train_loader, validation_loader, test_loader = DIVAObject.adata_to_diva_loaders(adata)
        self.set_data_loaders(train_loader, validation_loader, test_loader)
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': False} if self.args.cuda else {}

        # Model name
        self.set_outpath(outpath)
        print(self.outpath)
        if model_name is None:
            model_path = os.path.join(self.outpath, "DIVAModel")
            model_name = f"DIVAModel"
        else:
            model_path = os.path.join(self.outpath, model_name)
        print(model_name)
        self.set_model_name(model_name)

        # Set seed
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        # get dataloader from adata
        # can change the name of the domain and label columns
        train_loader, test_loader = dataprep.get_diva_loaders(adata, domain_name="domain", label_name="cell_type", shuffle=True)

        # splitting train loader into a training and validation set
        # can change the percent you want to make into a validation set
        train_loader, validation_loader = dataprep.get_validation_from_training(train_loader)

        data_loaders = {}
        # No shuffling here
        data_loaders['sup'] = data_utils.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        data_loaders['unsup'] = data_utils.DataLoader(test_loader, batch_size=self.args.batch_size, shuffle=True)
        data_loaders['valid'] = data_utils.DataLoader(validation_loader, batch_size=self.args.batch_size, shuffle=False)

        labels = train_loader.labels
        domains = train_loader.domains
        # these are set after the DIVA model is initiated

        num_labels = len(train_loader[0][1])
        num_domains = len(train_loader[0][2])
        num_dims = train_loader[0][0].shape[1]
        self.set_args_d_dim(num_domains)
        self.set_args_y_dim(num_labels)
        self.set_args_x_dim(num_dims)
        self.load_model_from_args()

        # setting the model to retain the integer to name matches
        self.model.labels = labels
        self.model.domains = domains

        self.set_args_ssl(ssl)
        train.epoch_procedure(model_path, self.args, self.model, data_loaders, self.device, ssl=ssl)

        print(f"Model name: {model_name}")
        # print(f"Train domain: {self.args.train_patient}")
        # print(f"Test domain: {self.args.test_patient}")
        print(f"Train domain: {self.args.train_domain}")
        print(f"Test domain: {self.args.test_domain}")
        print("Training Accuracy")
        print(train.get_accuracy(data_loaders['sup'], self.model, self.device))
        print("Validation Accuracy")
        print(train.get_accuracy(data_loaders['valid'], self.model, self.device))
        if test_loader is not None:
            print("Testing Accuracy")
            print(train.get_accuracy(data_loaders['unsup'], self.model, self.device))
        else:
            print("No test loader found")
        self.predict(adata)
        true_labels = adata.obs.cell_type[adata.obs.batch=="1"]
        predicted = adata.obs.label_preds[adata.obs.batch=="1"]
        visualization.save_cm(true=true_labels, preds=predicted, name=self.model_name, sort_labels=True, reduce_cm=False)

        visualization.plot_embeddings(self.model, data_loaders, self.device, self.model_name)

    def predict(self, adata):
        """
        returns predictions for label and domain within given adata
        alters adata, doesnt change anything in self

        Parameters
        ----------
        adata : AnnData

        Returns
        -------
        AnnData
            fills in adata.obs columns "label_preds" "domain_preds'

        """
        tensor_data = torch.from_numpy(adata.X).to(self.device)
        d_preds, y_preds = self._predict(tensor_data)
        d_preds_str = self.model.domains[d_preds]
        y_preds_str = self.model.labels[y_preds]
        adata.obs['label_preds'] = y_preds_str
        adata.obs['domain_preds'] = d_preds_str

    def _predict(self, tensor):
        """
        These return torch tensors of ints
        """
        self.model.eval()
        classifier_fn = self.model.classifier
        d_onehots, y_onehots = classifier_fn(tensor)
        d_preds, y_preds = torch.argmax(d_onehots, axis=1), torch.argmax(y_onehots, axis=1)
        return d_preds.cpu().numpy(), y_preds.cpu().numpy()


def load_model_from_file(name):
    if torch.cuda.is_available():
        device = 'cuda'
    device = 'cpu'
    args = torch.load(name + ".config")
    model = torch.load(name + ".model", map_location=torch.device(device))
    obj = DIVAObject(args)
    obj.set_model(model)
    obj.set_model_name(name)
    return obj
