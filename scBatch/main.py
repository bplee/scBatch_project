"""
Main class for trianing a DIVA model, stores the model runs training, testing, creation of figs
"""
import model
import dataprep
import train
import torch
import visualization
import numpy as np
import torch.utils.data as data_utils

class DIVAModel:
    def __init__(self, args=None):

        if args is None:
            from args import default_args
            self.args = default_args
        else:
            # allow user to input their own args file
            self.args = args

        self.model = None
        self.train_loader, self.test_loader, self.valid_loader = None, None, None

    def set_data_loaders(self, train, valid, test):
        self.train_loader = train
        self.valid_loader = valid
        self.test_loader = test

    def create_model_from_args(self):
        self.model = model.DIVA(self.args)

    def create_model_from_config(self, config_filepath):
        # TODO
        pass

    def load_model_from_args(self):
        print(f" loading model from args: {self.args}")
        self.model = model.DIVA(self.args)

    # def set_model_arg(self, arg, value):
    #     if self.args.arg is not None:
    #         print(f" changing {arg} from {self.args.arg} to {value}")
    #         self.args.arg = value
    #     else:
    #         print(f" {arg} does not exist. No changes.")

    def set_model_d_dim(self, value):
        print(f" changing d_dim from {self.args.d_dim} to {value}")
        self.args.d_dim = value

    def set_model_y_dim(self, value):
        print(f" changing y_dim from {self.args.y_dim} to {value}")
        self.args.y_dim = value

    def set_model_x_dim(self, value):
        print(f" changing x_dim from {self.args.x_dim} to {value}")
        self.args.x_dim = value

    def set_model_name(self, name):
        self.model_name = name

    @staticmethod
    def adata_to_diva_loaders(adata):
        train_loader, test_loader = dataprep.get_diva_loaders(adata, domain_name="patient", label_name="cell_type", shuffle=True)
        new_train_loader, validation_loader = dataprep.get_validation_from_training(train_loader)

        return new_train_loader, validation_loader, test_loader

    def fit(self, adata, epochs=200, model_name=None):
        train_loader, validation_loader, test_loader = DIVAModel.adata_to_diva_loaders(adata)
        self.set_data_loaders(train_loader, validation_loader, test_loader)

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if self.args.cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': False} if self.args.cuda else {}

        # Model name
        print(self.args.outpath)
        if model_name is None:
            model_name = f"{self.args.outpath}DIVAModel_seed_{self.args.seed}"
            fig_name = f"DIVAModel_seed_{self.args.seed}"
        else:
            model_name = f"{self.args.outpath}{model_name}_seed_{self.args.seed}"
            fig_name = f"{model_name}_seed_{self.args.seed}"
        print(model_name)
        self.set_model_name(model_name)

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

        cell_types = test_loader.labels
        patients = test_loader.domains

        num_labels = len(train_loader[0][1])
        num_domains = len(train_loader[0][2])
        num_dims = train_loader[0][0].shape[1]
        self.set_model_d_dim(num_domains)
        self.set_model_y_dim(num_labels)
        self.set_model_x_dim(num_dims)
        self.load_model_from_args()

        train.epoch_procedure(self.model_name, self.args, self.model, data_loaders, device)

        print("Training Accuracy")
        print(train.get_accuracy(data_loaders['sup'], self.model, device))
        print("Validation Accuracy")
        print(train.get_accuracy(data_loaders['valid'], self.model, device))
        print("Testing Accuracy")
        print(train.get_accuracy(data_loaders['unsup'], self.model, device))

        visualization.plot_embedings(self.model, data_loaders, device, self.model_name)

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


