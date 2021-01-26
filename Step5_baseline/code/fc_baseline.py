import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code,pkl_load_data import PdRccAllData

class Network(nn.module):
    def __init__(self, n_genes, n_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_genes, n_classes)
    def forward(self, x):
        h = self.fc1(x)
        return F.log_softmax(a, dim=1)




if __name__ == "__main__":
    data_obj = PdRccAllData()  # default args for this function will give me what I want
    raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)
    patients = data_obj.data.patient
    y_onehot = np.eye(dim_out)[pd.factorize(patients)]
    cell_types = data_obj.data.cell_type

    test_pat = 0

    dim_in = raw_counts.shape[0]
    dim_out = len(np.unique(cell_types))

    # creating model
    model = Network(dim_in, dim_out)


    # training
    loss_fn = nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    x = raw_counts
    y = np.eye(dim_out)[pd.factorize(patients)]

    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # prediction
