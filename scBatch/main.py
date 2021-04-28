"""
Main class for trianing a DIVA model, stores the model runs training, testing, creation of figs
"""
import model
import dataprep
import train
import torch
import numpy as np
import torch.utils.data as data_utils

class DIVAModel:
    def __init__(self, args=None):
        self.model = model.DIVA(args=args)
        self.args = self.args

    def adata_to_diva_loader(self, adata):
        train_loader, test_loader = dataprep.get_diva_loaders(adata, domain_name="patient", label_name="cell_type", shuffle=True)
        new_train_loader, validation_loader = dataprep.get_validation_from_training(train_loader)
        self.train_loader = new_train_loader
        self.test_loader = test_loader
        self.valid_loader = validation_loader

        return new_train_loader, validation_loader, test_loader

    @staticmethod
    def fit(self, model, adata, epochs=200):
        train_loader, validation_loader, test_loader = DIVAModel.adata_to_diva_loader(adata)

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if self.args.cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': False} if self.args.cuda else {}

        # Model name
        print(self.args.outpath)
        # TODO change the name stuff, put it in as an arg or something
        model_name = f"{self.args.outpath}210317_rcc_to_crc_no_conv_semi_sup_seed_{self.args.seed}"
        fig_name = f"210317_rcc_to_crc_no_conv_semi_sup_seed_{self.args.seed}"
        print(model_name)

        # Set seed
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        # get dataloader from adata
        # can change the name of the domain and label columns
        train_loader, test_loader = dataprep.get_diva_loaders(adata, shuffle=True)

        # splitting train loader into a training and validation set
        # can change the percent you want to make into a validation set
        train_loader, validation_loader = dataprep.get_validation_from_training(train_loader)

        data_loaders = {}
        # No shuffling here
        data_loaders['sup'] = data_utils.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        data_loaders['unsup'] = data_utils.DataLoader(test_loader, batch_size=self.args.batch_size, shuffle=True)
        data_loaders['valid'] = data_utils.DataLoader(validation_loader, batch_size=self.args.batch_size, shuffle=False)

        train.epoch_procedure(epochs, self.args, model, data_loaders)


        pass

    def transform(self, y):
        pass

    # def fit_transform(self, d, x, y):
    #     fit(d, x, y)
    #     transform()


def run_DIVA(adata, args):
    domain_name = "patient"
    label_name = "cell_type"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    args.d_dim = len(np.unique(adata.obs[domain_name]))
    args.y_dim = len(np.unique(adata.obs[label_name]))

    print(args.outpath)
    model_name = f"{args.outpath}210317_rcc_to_crc_no_conv_semi_sup_seed_{args.seed}"
    fig_name = f"210317_rcc_to_crc_no_conv_semi_sup_seed_{args.seed}"
    print(model_name)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    obj = DIVAModel(args=args)


