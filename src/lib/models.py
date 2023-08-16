"""
Architecture declaration of the models used in the project will appear here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# In order to use pre-trained resnet
import torchvision.models as models

import src.lib.core as core

class ResNet18(torch.nn.Module):
    """
    Pretrained ResNet18 on ImageNet, for MNIST dataset. Some slight changes have been made:

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
    Very light model. Used mainly to test ideas with a fast model. For MNIST dataset
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
        x = torch.flatten(x, 1)

        # Fully connected para llevar el vector aplanado a la dimension del
        # embedding
        x = self.fc(x)

        return x

    def set_permute(self, should_permute: bool):
        self.should_permute = should_permute


class LFWResNet18(torch.nn.Module):
    """
    Pretrained ResNet18 on ImageNet, for MNIST dataset. Some slight changes have been made:

    - Last linear layer have out_features given by __init__ parameter
    """

    def __init__(self, embedding_dimension: int):

        super(LFWResNet18, self).__init__()

        # Dimension del embedding que la red va a calcular
        self.embedding_dimension = embedding_dimension

        # Tomamos el modelo pre-entrenado ResNet18
        self.pretrained = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        # Cambiamos la primera convolucion para que en vez
        # de tres canales acepte un canal para las imagenes
        # de entrada
        #  self.pretrained.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False)

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


class LFWLightModel(torch.nn.Module):
    """
    Very light model. Used mainly to test ideas with a fast model. For LFW dataset
    """

    def __init__(self, embedding_dimension: int):

        super(LFWLightModel, self).__init__()

        # Dimension del embedding que la red va a calcular
        self.embedding_dimension = embedding_dimension

        # Bloques convolucionales
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.fc = nn.Linear(in_features = 468512, out_features = self.embedding_dimension)

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
        x = torch.flatten(x, 1)

        # Fully connected para llevar el vector aplanado a la dimension del
        # embedding
        x = self.fc(x)

        return x

    def set_permute(self, should_permute: bool):
        self.should_permute = should_permute


# TODO -- remove spanish docs and comments!
class FGLigthModel(torch.nn.Module):
    """
    Very light model. Used mainly to test ideas with a fast model. For FG-Net dataset
    """

    def __init__(self, embedding_dimension: int):

        super(FGLigthModel, self).__init__()

        # Embedding dimension that the network is going to compute
        self.embedding_dimension = embedding_dimension

        # Basic structure for the network
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.fc = nn.Linear(in_features = 36992, out_features = self.embedding_dimension)
        self.should_permute = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # We have input tensors with shape (1, DATALOADER_BACH_SIZE, 28, 28)
        # and we wnat to work with shapes (DATALOADER_BACH_SIZE, 1, 28, 28)
        #
        # We use `permute` instead of `reshape` because we want this code to work
        # with different values for `DATALOADER_BATCH_SIZE`
        if self.should_permute is True:
            x = torch.permute(x, (1, 0, 2, 3))

        # Go trought the conv blocks and some max pools
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)


        # Flatten for future fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected that gets the output with the proper dimension
        x = self.fc(x)

        return x

    def set_permute(self, should_permute: bool):
        self.should_permute = should_permute


class RandomNet(torch.nn.Module):
    """
    Random net that we are going to use in some tests and benchmarks
    """

    def __init__(self, embedding_dimension: int):

        super(RandomNet, self).__init__()

        # Dimension del embedding que la red va a calcular
        self.embedding_dimension = embedding_dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the batch size so we return the same number of vectors as
        # number of given images
        batch_size = x.shape[0]

        # Random values with the embedding_dimension specified in __init__
        return torch.rand([batch_size, self.embedding_dimension])


class NormalizedNet(torch.nn.Module):
    """
    Use a base model to compute outputs. This model gets that ouputs and
    normalizes them, dividing each vector by its euclidean norm
    """

    def __init__(self, base_model):
        super(NormalizedNet, self).__init__()

        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_model(x)
        normalized_output = torch.nn.functional.normalize(
            output,
            p = 2,
            dim = 1,
            eps = 0.1
        )
        return normalized_output

    def set_permute(self, should_permute: bool):
        self.base_model.set_permute(should_permute)


class RetrievalAdapter(torch.nn.Module):
    """
    Takes a network that computes embeddings of the input images and adapts it
    to a network that does a retrieval task

    That is to say, we end up with a network that, given a query image and a set
    of candidates, returns a ranked list of the most promising candidates
    """

    def __init__(self, base_net: torch.nn.Module):
        super(RetrievalAdapter, self).__init__()
        self.base_net = base_net

        # Put the base model on the proper device
        device = core.get_device()
        self.base_net.to(device)

    def query(self, query: torch.Tensor, candidates: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Given a `query` image, and a list of image `candidates` (list in the form
        of a pytorch tensor), returns the `k` most promising candidate indixes, ranked
        by relevance (that is to say, the first element of the list should be more
        similar to the `query` than the last element of the list)

        The list of indixes is returned as a `torch.Tensor`
        """

        # Check the dimensions of the query
        # Query is a single image so `query.shape == [channels, width, height]`
        # and `channels in [1, 3]`
        if len(query.shape) != 3:
            raise ValueError(f"`query` should be a tensor with three modes, only {len(query.shape)} modes were found")

        if query.shape[0] != 3 and query.shape[1] != 1:
            raise ValueError(f"`query` must be an image with one or three channels, got {query.shape[0]} channels")

        # Check the dimensions of the candidates
        # `candidates` is a list of images, so `candidates.shape == [n, channels = 1 | 3, width, height]`
        # Also, as we are querying for the best `k` candidates, we should have at least
        # `k` candidates
        if len(candidates.shape) != 4:
            raise ValueError(f"`candidates` should be a tensor with four modes, only {len(candidates.shape)} modes were found")

        if candidates.shape[1] != 3 and candidates.shape[1] != 1:
            raise ValueError(f"Candidates must be images of one or three channels, got {candidates.shape[1]} channels")

        if candidates.shape[0] < k:
            raise ValueError(f"Querying for the best {k} candidates, but we only have {candidates.shape[0]} candidates in total")

        # Our network only accepts batches of images. Query is a single image,
        # so create a batch with a single image:
        query = query[None, ...]

        # Make sure that both query and candidates tensors are in the proper device
        device = core.get_device()
        candidates = candidates.to(device)
        query = query.to(device)

        # Compute the embeddings of the images
        query_embedding = self.base_net(query)
        candidate_embeddings = self.base_net(candidates)

        # Take advantage of the method that computes the best k candidates
        # using the embeddings that we have computed
        return self.query_embedding(query_embedding, candidate_embeddings, k)

    def query_embedding(self, query_embedding: torch.Tensor, candidate_embeddings: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Does the same as `Self.query_embedding` but taking as parameters the
        embeddings, and not the images that we are going to embed

        NOTE: in this computation, as we already have the embeddings computed,
        we are not using `self.base_net`
        """

        # Check the shapes of the embeddings given as parameters
        if len(query_embedding.shape) != 1:
            raise ValueError(f"We were expecting a single embedding, thus only one mode, got {len(query_embedding.shape)} modes")

        if len(candidate_embeddings.shape) != 2:
            raise ValueError(f"We were expecting a batch of embeddings, thus two modes, got {len(candidate_embeddings.shape)} modes")

        # Check that embedding dimension are the same for the query and the candidates
        if query_embedding.shape[0] != candidate_embeddings.shape[1]:
            err_msg = "Embedding dimensions are not the same for candidates and query\n"
            err_msg += f"Embedding dimension for query is {query_embedding.shape[0]}\n"
            err_msg += f"Embedding dimension for candidates is {candidate_embeddings.shape[1]}\n"
            raise ValueError(err_msg)

        # We must have at least `k` candidates to be able to compute the search
        if candidate_embeddings.shape[0] < k:
            raise ValueError(f"Querying for the best {k} candidates, but we only have {candidate_embeddings.shape[0]} candidates in total")

        # Compute the euclidean distances between the query and the candidates
        #
        # In fist step, `query - candidates` expands query to be a torch list of
        # `k` copies of the query tensor to match `candidates` dimensions
        # We have `k` tensors of `embedding_dimension` entries.
        #
        # In the second step, we sum all the squares of each diff entry and then
        # compute the square root of each square sum, which actually is the
        # euclidean distance of each entry
        diff_squared = (query_embedding - candidate_embeddings) ** 2
        distances = torch.sqrt(diff_squared.sum(1))

        # Now get the best `k` indixes
        sorted_indixes = torch.sort(distances, descending = False)[1]
        best_k_indixes = sorted_indixes[:k]

        return best_k_indixes

    def set_permute(self, should_permute: bool):
        self.base_net.set_permute(should_permute)
