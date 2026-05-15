from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

# Khởi tạo hệ số random cho CPU và GPU
random_state = 42
np.random.seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)

class BaseMLP(ABC):
    def __init__(self):
        self.Layers = []
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = None
    
    @abstractmethod
    def predict(self,X):
        pass
    @abstractmethod
    def get_accuracy(self,logits,y):
        pass
    @abstractmethod
    def compute_loss(self,logits, y):
        pass
    
    def Add_layer(self,layer):
        self.Layers.append(layer)
        # dấu * ở đây có nghĩa là mỗi phần tử của list là 1 tham số của hàm nn.Sequential
        self.model = nn.Sequential(*self.Layers)
    def forward(self,X):
        if self.model is not None:
            return self.model(X)
        raise ValueError("BaseMLP.model is None !")

    def print_fmt(self,Value):
        if Value is None or len(Value) ==0:
            return float('nan')
        return Value[-1]
    def fit(self,X = None, y = None, dataset = None, lr = 0.01, n_epochs = 100, batch_size = 1, verbose = 0, validation_split = None, is_shuffle = True, optimizer = 'SGD', criterion = 'MSE'):
        if X is not None and y is not None:
            self.dataset = TensorDataset(X,y)
        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError("BaseMLP.dataset is empty !")
        # Chuyển mô hình lên GPU
        if self.model is not None:
            self.model = self.model.to(self.device)
        else:
            raise ValueError("BaseMLP.model is None !")
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.is_shuffle = is_shuffle
        optimizers = {
            'sgd':torch.optim.SGD,
            'adam':torch.optim.Adam
        }
        
        criterions = {
            'mse': nn.MSELoss,
            'ce': nn.CrossEntropyLoss,
            'bce': nn.BCEWithLogitsLoss,
        }
        # Kiểm tra một biến có kiểu Dl là gì đó
        if isinstance(criterion,str):
            crit_class = criterions.get(criterion.lower())
            if crit_class is None:
                raise ValueError(f"criterion unsupport {criterion}")
            self.criterion = crit_class()
        else:
            self.criterion = criterion
        
        if isinstance(optimizer,str):
            optim_class = optimizers.get(optimizer.lower())
            if optim_class is None:
                raise ValueError(f"optimizer unsupport {optimizer}")
            self.optimizer = optim_class(self.model.parameters(),lr = self.lr)
        else:
            self.optimizer = optimizer(self.model.parameters(),lr = self.lr)
        self.Val_Loader = None
        self.Train_Loader = None
        self.Losses = []
        self.Accuracies = []
        self.Val_Losses = []
        self.Val_Accuracies = []
        if self.validation_split is not None and self.validation_split>0:
            '''
            Val_size = int(self.validation_split*len(self.dataset))
            Train_size = len(self.dataset) - Val_size
            # Chia DL trong train set và val set nếu có
            # Cách chia theo random_split sẽ làm mất cân bằng nhãn -> mô hình học không tốt
            self.TrainSet, self.ValSet = random_split(self.dataset,[Train_size,Val_size])
            self.Val_Loader = DataLoader(self.ValSet,batch_size=Val_size)
            '''
            idxs = range(len(self.dataset))
            labels = list(zip(*self.dataset))[1]
            # Tham số stratify=labels sẽ giữ nguyên phân bố của nhãn khi split giữa các tập được chia -> phân bố nhãn giữa các tập giống nhau 
            try: 
                # Split train, test follow index of dataset then use function Subset to split Traindaset and Valdataset
                train_idx, val_idx = train_test_split(idxs, train_size= (1- self.validation_split), stratify=labels, shuffle=True ,random_state=random_state)
            except:
                train_idx, val_idx = train_test_split(idxs, train_size= (1- self.validation_split), shuffle=True ,random_state=random_state)
            self.TrainSet, self.ValSet = Subset(self.dataset,train_idx), Subset(self.dataset,val_idx)
            self.Val_Loader = DataLoader(self.ValSet,batch_size=len(val_idx))
        else:
            self.TrainSet = self.dataset
        self.Train_Loader = DataLoader(self.TrainSet,batch_size=self.batch_size, shuffle= self.is_shuffle)
        num_batch = len(self.Train_Loader)

        for epoch in tqdm(range(self.n_epochs)):
            if self.verbose:
                print(f"Epoch [{epoch+1:>4}/{self.n_epochs}]")
            batch_count = 1
            Loss_epoch = 0
            Acc_epoch = 0
            for X_batch_train, y_batch_train in self.Train_Loader:
                X_batch_train, y_batch_train = X_batch_train.to(self.device), y_batch_train.to(self.device)

                # forward 
                logits = self.forward(X_batch_train)
                loss = self.compute_loss(logits,y_batch_train)
                Loss_epoch += loss.item()
                acc = self.get_accuracy(logits, y_batch_train)
                Acc_epoch += acc.item()

                # 
                self.optimizer.zero_grad()

                # compute Gradient
                loss.backward()

                # update weight
                self.optimizer.step()
                
                if self.verbose == 1 and batch_count< num_batch:
                    print(f"Batch {batch_count:>4}/{num_batch} - Loss = {loss.item():.4f} - Accuracy = {acc.item():.4f}")
                
                batch_count +=1
            self.Losses.append(Loss_epoch/num_batch)
            self.Accuracies.append(Acc_epoch/num_batch)
            if self.validation_split is not None and self.validation_split > 0:
                X_Val, y_val = next(iter(self.Val_Loader))
                X_Val, y_val = X_Val.to(self.device), y_val.to(self.device)
                with torch.no_grad():
                    logits_Val = self.model(X_Val)
                    val_loss = self.compute_loss(logits_Val,y_val)
                    val_acc = self.get_accuracy(logits_Val, y_val)
                    self.Val_Losses.append(val_loss.item())
                    self.Val_Accuracies.append(val_acc.item())
            if self.verbose == 1:
                print(f"Batch {num_batch}/{num_batch} - Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = {self.print_fmt(self.Val_Losses):.4f} - Accracy_Validation = {self.print_fmt(self.Val_Accuracies):.4f}")
            if self.verbose >1:
                print(f"Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = {self.print_fmt(self.Val_Losses):.4f} - Accracy_Validation = {self.print_fmt(self.Val_Accuracies):.4f}")