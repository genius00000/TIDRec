# Triple-graph Interactive Distillation Recommendation (TIDRec).

## Requirements
- Python 3.8.8
- pytorch 1.9.0
- Numpy 1.20.1
- dgl 0.6.1

## Usage
Execute the following scripts to train and test TIDRec on the wos dataset with default hyper-parameters:

```
python main.py --conf_name=tidrec --data_name=wos --train_model --test_model
```

There are some key options of these scrips:

--conf_name: Choose the corresponding framework for training/testing. By default we use tidrec. If one wants to train each model separately, one can set this value to tidrec_no. Also, One can modify the configuration file in tidrec/configs.

--data_name: Choose the dataset for training/testing. 

--train_model: Perform training process.

--test_model: Perform testing process.