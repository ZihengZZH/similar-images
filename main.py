from src.data_mnist import DataMNIST
from src.siameseNet import SiameseNet

DataMNIST().vis()
new_inst = SiameseNet()
new_inst.train_model()
new_inst.random_test()

