


class MLP_Parrallel(nn.Module):
    def __init__(self,class_num,win_len,feature_size):
        super(MLP_Parrallel, self).__init__()
        self.encoder_1 = MLP_encoder(win_len,feature_size)
        self.encoder_2 = MLP_encoder(win_len,feature_size)

        self.classifier = nn.Linear(128,class_num)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class MLP_encoder(nn.Module):
    def __init__(self,win_len,feature_size,hidden_states = 256):
        super(MLP_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1*win_len*feature_size,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
        )
        self.mapping = nn.Linear(128, hidden_states)

        self.bn = nn.BatchNorm1d(hidden_states)

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1, 1*win_len*feature_size)
        x = self.encoder(x)
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x
