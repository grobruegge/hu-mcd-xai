import torch
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
import scipy.stats as stats

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a simple linear model with a sigmoid activation function
class LogClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(LogClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        torch.nn.init.kaiming_uniform_(self.linear.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def test_cav(cav_model, dataloader):

    cav_model.eval()
    # Evaluate on the testing set
    total_correct_test = 0
    total_samples_test = 0

    with torch.no_grad():
        for b_data, b_labels in dataloader:
            b_data, b_labels = b_data.to(DEVICE), b_labels.to(DEVICE)

            b_output_test = cav_model(b_data).squeeze()

            total_correct_test += ((b_output_test > 0.5).float() == b_labels).sum().item()
            total_samples_test += b_labels.size(0)
        
    return total_correct_test / total_samples_test

def cav_pytorch_training(X:np.ndarray, y:np.ndarray, batch_size:int, num_epochs:int)->dict:
    # do train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    concept_dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.float)
    )
    concept_dataset_test = torch.utils.data.TensorDataset(
        torch.tensor(test_data, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.float)
    )

    # define CAV model
    cav_model = LogClassifier(X.shape[1]).to(DEVICE)

    # Define the loss function
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer (e.g., Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(cav_model.parameters(), lr=1e-2)
    # optimizer = torch.optim.Adam(cav_model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_loss_history = []
    train_acc_history = []
    test_acc_history = []
    best_test_acc = 0.0
    current_patience = 0

    for epoch in range(num_epochs):
        # put model in training mode
        cav_model.train()
        total_correct_train = 0
        total_samples_train = 0
        sum_loss_train = 0  
        for b_data, b_labels in  torch.utils.data.DataLoader(
            concept_dataset_train, batch_size, shuffle=True
        ):
            b_data, b_labels = b_data.to(DEVICE), b_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            b_outputs = cav_model(b_data).squeeze(-1)
                    
            loss = criterion(b_outputs, b_labels)
            loss.backward()

            optimizer.step()

            sum_loss_train += float(loss.item())
            total_correct_train += ((b_outputs > 0.5).float() == b_labels).sum().item()
            total_samples_train += b_labels.size(0)
        
        train_loss_history.append(sum_loss_train)
        train_acc_history.append(total_correct_train / total_samples_train)
        
        scheduler.step() 
        
        # obtain accuracy on held-out test dataset
        test_acc_history.append(
            test_cav(
                cav_model, 
                dataloader=torch.utils.data.DataLoader(
                    concept_dataset_test, batch_size, shuffle=False
                )
            )
        )

        # Check if test accuracy has improved
        if test_acc_history[-1] > best_test_acc:
            # set current (best) test accuracy
            best_test_acc = test_acc_history[-1]
            # reset waiting counter
            current_patience = 0
            # save the best model
            best_model_state = cav_model.state_dict()
        else:
            current_patience += 1
            # If the model hasn't improved for 5 epochs, stop training
            if current_patience >= 10: # TODO: make this a parameter
                break
    
    cav_model.load_state_dict(best_model_state)
    
    # extract weight vector of CAV model
    weight_vector = cav_model.linear.weight.data.detach().clone().cpu().numpy().squeeze()
    
    cav_dict = {
        # save normalized weight vector
        'weight_vector': weight_vector, #/np.linalg.norm(weight_vector),
        # save test accuracy
        'test_accuracy_history': test_acc_history,
        'train_accuracy_history': train_acc_history,
        'train_loss_history': train_loss_history
    }
        
    return cav_dict

def cav_sklearn_training(X:np.ndarray, y:np.ndarray)->dict:
    # settings taken from original TCAV implementation
    model = linear_model.SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cav = {
        'weight_vector': model.coef_[0],
        'accuracy': metrics.accuracy_score(y_pred, y_test)
    }
    return cav

def statistical_testings(i_ups_concept, i_ups_random):
    """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
    min_len = min(len(i_ups_concept), len(i_ups_random))
    _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
    return p