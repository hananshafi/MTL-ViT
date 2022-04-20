class ModelParams():

  def __init__(self):
    self.dataset = 'CelebA'   #['CelebA', 'CIFAR10']
    self.n_classes = 2
    self.num_tasks = 9
    self.epochs = 100
    self.lr = 1e-3
    self.BS = 128
    #self.iter_epoch = len(X_train)// self.BS
    self.save_period = 10
    self.weight_decay = 0.0001
    self.celeba_tasks = ['5_o_Clock_Shadow','Black_Hair','Blond_Hair','Brown_Hair','Goatee','Mustache','No_Beard','Rosy_Cheeks', 'Wearing_Hat']  #celeba tasks: 
    self.cifar10_tasks = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck'] #cifar10 tasks