"""
Architecture declaration of the models used in the project will appear here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# In order to use pre-trained resnet
import torchvision.models as models

class ResNet18(torch.nn.Module):
    """
    Pretrained ResNet18 on ImageNet. Some slight changes have been made:

    - First convolution (in_channels = 1 and not in_channels = 3)
    - Last linear layer have out_features given by __init__ parameter
    """

    def __init__(self, embedding_dimension: int):

        super(ResNet18, self).__init__()

        # Dimension del embedding que la red va a calcular
        self.embedding_dimension = embedding_dimension

        # Tomamos el modelo pre-entrenado ResNet18
        self.pretrained = models.resnet18(pretrained=True)

        # Cambiamos la primera convolucion para que en vez
        # de tres canales acepte un canal para las imagenes
        # de entrada
        self.pretrained.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Cambiamos la ultima capa fc Linear(in_features=512, out_features=1000, bias=True)
        # para calcular un embedding de dimension mucho menor, especificada por parameatro
        # TODO -- comentar en la memoria el cambio de ERROR que hacer esto nos ha supuesto
        self.pretrained.fc = nn.Linear(in_features=512, out_features=self.embedding_dimension, bias=True)

        # Por defecto siempre realizamos la permutacion del tensor de entrada
        self.should_permute = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tenemos como entrada tensores (1, DATALOADER_BACH_SIZE, 28, 28) y
        # queremos tensores (DATALOADER_BACH_SIZE, 1, 28, 28) para poder trabajar
        # con la red pre-entrenada
        # Usamos permute en vez de reshape porque queremos que tambien funcione al
        # realizar inferencia con distintos tamaños de minibatch (ie. 1)
        if self.should_permute is True:
            x = torch.permute(x, (1, 0, 2, 3))

        # Usamos directamente la red pre-entrenada para hacer el forward
        x = self.pretrained.forward(x)

        return x

    def set_permute(self, should_permute: bool):
        self.should_permute = should_permute


class LightModel(torch.nn.Module):
    """
    Very light model. Used mainly to test ideas with a fast model,
    """

    def __init__(self, embedding_dimension: int):

        super(LightModel, self).__init__()

        # Dimension del embedding que la red va a calcular
        self.embedding_dimension = embedding_dimension

        # Bloques convolucionales
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.fc = nn.Linear(in_features = 3200, out_features = self.embedding_dimension)

        self.should_permute = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Tenemos como entrada tensores (1, DATALOADER_BACH_SIZE, 28, 28) y
        # queremos tensores (DATALOADER_BACH_SIZE, 1, 28, 28) para poder trabajar
        # con la red
        # Usamos permute en vez de reshape porque queremos que tambien funcione al
        # realizar inferencia con distintos tamaños de minibatch (ie. 1)
        if self.should_permute is True:
            x = torch.permute(x, (1, 0, 2, 3))

        # Pasamos el tensor por los distintos bloques de nuestra red
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))


        # Max pooling y seguido flatten de todas las dimensiones menos la del batch
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x,1)

        # Fully connected para llevar el vector aplanado a la dimension del
        # embedding
        x = self.fc(x)

        return x

    def set_permute(self, should_permute: bool):
        self.should_permute = should_permute
