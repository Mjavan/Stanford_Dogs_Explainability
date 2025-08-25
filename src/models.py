import torch
import torchvision.models as models
import torch.nn as nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(len(x), -1)

class EncoderClassifier(nn.Module):
    '''Covolutional classification NN with same feature map as autoencoder'''
    def __init__(self, d=100, c_out=118):
        super(EncoderClassifier, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 40, 3, padding=0),
                nn.BatchNorm2d(40),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(40, 80, 3, padding=0),
                nn.BatchNorm2d(80),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(80, 160, 3, padding=0),
                nn.BatchNorm2d(160),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(160, 240, 3, padding=0),
                nn.BatchNorm2d(240),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(240, 360, 3, padding=0),
                nn.BatchNorm2d(360),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(360, d, 3, padding=0),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                FlattenLayer(),
                nn.Linear(d, c_out),
                )
    def forward(self, x):
        return self.features(x)


# Here is another unsupervised model that matthias used
class CAE224(nn.Module):
    def __init__(self, d=2048):
        super(CAE224, self).__init__()
        p = 0
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 40, 3, padding=p),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(2),
                
                nn.Conv2d(40, 80, 3, padding=p),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(80, 160, 3, padding=p),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(160, 240, 3, padding=p),
                nn.BatchNorm2d(240),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(240, 360, 3, padding=p),
                nn.BatchNorm2d(360),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(360, d, 3, padding=p),
                nn.BatchNorm2d(d),
                nn.MaxPool2d(2),
                nn.Tanh(),
                )
        p2 = 0
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(d, 360, 3, padding=p2),
                nn.BatchNorm2d(360),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(8, 8)),

                nn.ConvTranspose2d(360, 240, 3, padding=p2),
                nn.BatchNorm2d(240),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(24, 24)),

                nn.ConvTranspose2d(240, 160, 3, padding=p2),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(56, 56)),

                nn.ConvTranspose2d(160, 80, 3, padding=p2),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(112, 112)),

                nn.ConvTranspose2d(80, 40, 3, padding=p2),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(222, 222)),

                nn.ConvTranspose2d(40, 3, 3, padding=p2),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Let's check models to see if they are working properly or not
# The output of EncoderClassifier is as follows:
# In conelutional classifier, the output embedding has dimension: 118
# The output of autoencode is as follows: [1,3,224,224]
if __name__=="__main__":

    model1 = EncoderClassifier()
    a = torch.randn(1,3,224,224)
    out = model1(a)
    print(f'out:{out.shape}')

    model2= CAE224()
    b = model2(a)
    print(f'b:{b.shape}')