## BackGround
This repo is run using a dataset that contains the information of the bank account data to classify the users' credit level.
Notice that the data size of different classes are distributed unevenly. 

## Dataset Selection
To change the dataset , please go to config/config.py. 
Notice the dataset must be in .csv format and the final columns must be the label of the dataset

```
data_path = "./data/BankChurners.csv"
num_input = 10
num_class = 10
```

## Run Files
To run the repo:

## 1. a) For centralized training with Neural Network, 

   ```
   run train.py
   ```
   
  To use a different network , you can edit the code in train.py 
  
  ```
  def train_nn(model_save_path, acc_save_path, train_loader, test_loader):
      model = FcNet()
      # model = DeeperFcNet()
      params_to_update = model.parameters()
  ```

   ## b) For testing the ensemble learning performance, 
   ```
   run ensemble_learning/test_all.py
   ```
   
   ![text](/ensemble_learning_result/EnsembleLearningResult.png) ![text](/ensemble_learning_result/NeuralNetworkResult.png)
   
   
   ## b) For training the ensemble learning's models, 
   ```
   run ensemble_learning/train_all.py
   ```
       
   The ensemble learning method is defined below: 
   ![text](/ensemble_learning_result/el_equation.png)
   
## 2. For training with Federated Learning

   ```
   run fl_train.py
   ```
   
   The results will be saved as diagram in names like "Test Accuracy of Federated Learning Alpha=0.02,User = 5.jpg" and "Test Loss of Federated Learning Alpha=0.02,User = 5.jpg"
   
   The dataset distribution will be named as name like "Alpha=0.02,User = 5.jpg" 
   
## To edit the hyperparameter, go to fl_train.py 

The alpha value and user_num define how data is distributed to local trainer

```
if __name__ == '__main__':
    alpha = 0.1
    user_num = 5
    global_rounds = 30
    local_epoches = 5
    batch_size = 64

    spliter = \
    DatasetSplitByDirichletPartition(file_path=config.data_path,
                                       alpha=alpha,
                                       user_num=user_num,
                                       train_ratio=.7)
```

## To turn on or off the Robust Learning Rate

please change the rlr value to True or False

```
    train(
            spliter = spliter,
            alpha = alpha,
            user_num = user_num,
            global_rounds = global_rounds,
            local_epoches = local_epoches,
            batch_size = batch_size,
            rlr = True,
            log=True)
```
The Robust Learning Rate is defined as follow

![text](/federated_learning_result/rlr_equation.png)

## To change the method for aggregating local weights

simple average and weighted average functions are defined as below 
```
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def weighted_average_weights(w, dataset_proportions):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] += torch.mul(w[i][key], dataset_proportions[i])
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg
   
```
Replace the functions below

```
            if len(new_local_weights) != 0:
                global_weight = weighted_average_weights(new_local_weights,                  new_dataset_proportions)
            else:
                global_weight = initial_global_weight
        else:
            global_weight = weighted_average_weights(local_weights, dataset_proportions)
```
