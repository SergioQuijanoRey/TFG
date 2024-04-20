import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

from . import core


# TODO -- trasnlate documentation to english
class EmbeddingToClassifier:
    """
    Our models generate an embedding
    Using this class, we use K-NN algorithm to adapt our model to a classification task
    """

    def __init__(
        self, embedder: nn.Module, k: int, data_loader, embedding_dimension: int
    ):
        # El modelo que calcula los embeddings
        self.embedder = embedder

        # Dataloader que representa el dataset que usamos para k-nn
        self.data_loader = data_loader

        # Tamaño del embedding con el que estamos trabajando
        self.embedding_dimension = embedding_dimension

        # Tomamos el dispositivo en el que esta el modelo y los
        # datos, porque nos va a hacer falta durante todo el codigo
        self.device = core.get_device()

        # Calculamos todos los embeddings de los puntos
        self.dataset_embedded = self.__calculate_embedding()

        # Modelo de clasificacion k-nn
        self.k = k
        self.knn = self.__fit_knn()

    def predict_proba(self, img, batch_mode: bool = False) -> int:
        # Ponemos la red en modo evaluacion
        self.embedder.eval()

        # Tenemos una unica imagen, lo que queremos es
        # tener un batch de una imagen para que la red
        # pueda trabajar con ello
        single_img_batch = torch.Tensor(img)

        # Calculamos el embedding de la imagen
        img_embedding = None
        if batch_mode is False:
            img_embedding = self.embedder(single_img_batch[None, ...].to(self.device))
        else:
            img_embedding = self.embedder(single_img_batch.to(self.device))

        # Pasamos el embedding a cpu que es donde esta
        # el modelo knn de scikit learn
        img_embedding = img_embedding.cpu().detach().numpy()

        # Antes de salir de la funcion volvemos a poner
        # a la red en modo entrenamiento
        self.embedder.train()

        # Usamos dicho embedding para clasificar con knn
        return self.knn.predict(img_embedding)

    def predict(self, img, batch_mode: bool = False) -> int:
        # Ponemos la red en modo evaluacion
        self.embedder.eval()

        # Tenemos una unica imagen, lo que queremos es
        # tener un batch de una imagen para que la red
        # pueda trabajar con ello
        single_img_batch = torch.Tensor(img)

        # Calculamos el embedding de la imagen
        img_embedding = None
        if batch_mode is False:
            img_embedding = self.embedder(single_img_batch[None, ...].to(self.device))
        else:
            img_embedding = self.embedder(single_img_batch.to(self.device))

        # Pasamos el embedding a cpu que es donde esta
        # el modelo knn de scikit learn
        img_embedding = img_embedding.cpu().detach().numpy()

        # Antes de salir de la funcion volvemos a poner
        # a la red en modo entrenamiento
        self.embedder.train()

        # Usamos dicho embedding para clasificar con knn
        return self.knn.predict(img_embedding)

    def predict_using_embedding(self, embedding: np.ndarray) -> int:
        """
        Realizamos la prediccion, pero en vez de usando la imagen
        pasamos directamente el embedding de la imagen (en ocasiones
        podemos mejorar el rendimiento pre-computando el embedding de
        todo un conjunto de imagenes)
        """

        # Usamos dicho embedding para clasificar con knn
        return self.knn.predict(embedding)

    def predict_proba_using_embedding(self, embedding: np.ndarray) -> int:
        """
        Realizamos la prediccion, pero en vez de usando la imagen
        pasamos directamente el embedding de la imagen (en ocasiones
        podemos mejorar el rendimiento pre-computando el embedding de
        todo un conjunto de imagenes)
        """

        # Usamos dicho embedding para clasificar con knn
        return self.knn.predict_proba(embedding)

    def __calculate_embedding(self):
        """Dado el conjunto de imagenes con sus etiquetas, calculamos
        el conjunto de embedding con sus etiquetas"""

        embedded_imgs = []
        labels = []

        # Por motivos que desconocemos, ahora los tensores vienen
        # en el formato que espera la red, asi que no tenemos que
        # realizar la permutacion del tensor
        self.embedder.set_permute(False)

        for img, img_class in self.data_loader:
            # TODO -- esto hay que borrarlo
            if np.random.rand() < 0.01:
                break

            # Calculamos el embedding de la imagen
            img_embedding = self.embedder(img.to(self.device))

            # Añadimos el embedding asociado a la etiqueta
            embedded_imgs.append(img_embedding)
            labels.append(img_class)

        # Antes de devolver los datos, volvemos a colocar la opcion
        # de que permute los tensores
        self.embedder.set_permute(True)

        return embedded_imgs, labels

    def __fit_knn(self):
        # Tomamos los datos en el formato que espera sklearn
        # para realizar el fit
        x, y = self.prepare_data_for_sklearn()

        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(x, y)
        return knn

    def scatter_plot(self):
        """
        Hacemos un scatter plot del embedding obtenido
        """

        # Solo hacemos este plot cuando la dimension del
        # embedding es 2
        if self.embedding_dimension != 2:
            print(
                f"Skipping the scatter plot, as we have {self.embedding_dimension} dimensions"
            )
            return

        # Tomamos los datos en el formato adecuado para hacer el plot
        x, y = self.prepare_data_for_sklearn()

        # Los ejes x,y son los datos de nuestro vector x
        # El color de los puntos lo dan las etiquetas almacenadas en y
        plt.scatter(x=x[:, 0], y=x[:, 1], c=y)
        plt.show()

    def prepare_data_for_sklearn(self):
        """
        Tomamos las imagenes y las etiquetas, y las devolvemos en un
        formato adecuado para sklearn y matplotlib. Esto es:
            - Pasar los datos a memoria RAM
            - Aplanar los datos (tenemos los datos agrupados en minibatches)
        """

        # Separamos los datos segun espera sklearn
        x = self.dataset_embedded[0]
        y = self.dataset_embedded[1]

        # Pasamos de una lista de sublistas (por los minibatches)
        # a una lista. Tomamos la idea de:
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        x = [item for sublist in x for item in sublist]
        y = [item for sublist in y for item in sublist]

        # Forzamos a usar la memoria RAM (podrian estar los datos
        # en memoria GPU)
        x = np.array([element.cpu().detach().numpy() for element in x])
        y = np.array([element.cpu().detach().numpy() for element in y])

        return x, y
