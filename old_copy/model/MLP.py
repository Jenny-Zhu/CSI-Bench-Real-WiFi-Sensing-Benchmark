



class OW_MLP(nn.Module):

  """
    A class to store the Multi-layer perceptron model
    ...

    Attributes
    ----------
    class_num:int
      the number of classes for training models

    Methods
    -------
    forward(x)
      computes output Tensors from input Tensors.

  """
  def __init__(self,class_num,win_len,feature_size):

    super(OW_MLP,self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(win_len*feature_size,1024),
      nn.ReLU(),
      nn.Linear(1024,128),
      nn.ReLU(),
      nn.Linear(128,class_num)
    )

  def forward(self,x):
    x = x.view(-1,win_len*feature_size)
    x = self.fc(x)
    return x
