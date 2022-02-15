"""
Module for displaying visualizations that are not included in tensorboard
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

def is_black_and_white(img: np.ndarray) -> bool:
    """Checks if a given image is in color or in black & white format"""
    if len(img.shape) == 2 or img.shape[0] == 1:
        return True

    return False

def show_img(img: np.ndarray, color_format_range = (0, 255)):
    """
    Displays an image

    Parameters:
    ===========
    img: the image. If its in color, use RGB format
    color_format_range: range for the colors, ie. [0, 255]
    """

    # Security check
    if img.min() < color_format_range[0] or img.max() > color_format_range[1]:
        raise Exception(f"Expected img in range [{color_format_range[0]}, {color_format_range[1]}], got range [{img.min()}, {img.max()}]")

    if is_black_and_white(img):
        # B/W images displayed in gray scale
        plt.imshow(img, cmap = "gray")

    else:
        plt.imshow(img, vmin = color_format_range[0], vmax = color_format_range[1])

    # Make it visible right now
    # This is specially useful when working in Jupyter Notebooks
    plt.show()

def show_images(images: List[np.ndarray], color_format_range: Tuple[int, int] = (0, 255), columns: int = 4, figsize: Tuple[int, int] = None):
    """
    Shows multiple images in one figure, using rows & cols layout

    TODO -- color_format_range implies that all images must have same color range

    @param images: list of images to be displayed
    @param color_format_range: range for the colors, ie. [0, 255]
    @param columns: number of columns of the layout
    @param figsize: size of the window to show the img
                    If None, default figsize is set
    """

    # Tamaño de cada imagen que mostramos en el mosaico
    # Si no se pasa por parametro, usamos un valor por defecto
    if figsize is None: figsize = (15, 15)
    fig = plt.figure(figsize=figsize)

    # Calculamos el numero de filas necesarias para nuestras imagenes que depende
    # de la cantidad de imagenes que tengamos y del numero de columnas establecido
    rows = len(images) // columns + 1

    # Recorro las imagenes junto con los titulos y las voy colocando en el mosaico
    for index, img in enumerate(images):
        # Añado el subplot del mosaico para esta imagen concreta en
        # la posicion que le corresponde
        # Ademas le estamos pasando los titulos dados por parametro
        fig.add_subplot(rows, columns, index + 1)

        # Añado la imagen a la posicion especificada
        # Tenemos que cambiar los canales si la imagen es tricolor, pues
        # estamos usando formato cv2 BGR
        if is_black_and_white(img) == True:
            plt.imshow(img)
        else:
            plt.imshow(img[:,:,::-1])

    # Para el espaciado entre filas
    # Esta es la orden que menciono en el enlace al problema con el espaciado
    plt.subplots_adjust(top = 0.5, bottom=0.2, hspace=0.5, wspace=0.5)

    # Mostramos la composicion y espero en caso de estar en local
    plt.show()

def show_images_with_titles_same_window(images: List[np.ndarray], titles: List[str], columns: int = 4, figsize: Tuple[int, int] = None):
    """
    Muestra varias imagenes con sus respectivos titulos en una misma ventana

    Inspirado en
        https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
    En ese enlace, se generan un numero fijado de imagenes. En mi caso, añado
    la funcionalidad de fijar el numero de columnas y variar el numero de filas
    segun la cantidad de imagenes que pasemos en la lista

    Tuve un problema con un espaciado demasiado grande entre columnas, que resolvi gracias a:
        https://stackoverflow.com/a/41011910


    @param images: lista de imagenes que queremos representar
    @param titles: titulos respectivos de las imagenes que queremos representar
                   Debe tener el mismo tamaño que la lista de imagenes
    @param figsize: tamaño de cada imagen. Si no se especifica (None) se da un
                    valor por defecto
    """

    # Comprobacion de seguridad
    if len(images) != len(titles):
        err_msg = "La lista de imagenes debe tener el mismo tamaño que la lista de titulos\n"
        err_msg += f"{len(images)} imagenes, {len(titles)} titulos"
        raise Exception(err_msg)

    # Tamaño de cada imagen que mostramos en el mosaico
    # Si no se pasa por parametro el valor, tomamos un figsize por defecto
    if figsize is None: figsize = (15, 15)
    fig = plt.figure(figsize=figsize)

    # Calculamos el numero de filas necesarias para nuestras imagenes que depende
    # de la cantidad de imagenes que tengamos y del numero de columnas establecido
    rows = len(images) // columns + 1

    # Recorro las imagenes junto con los titulos y las voy colocando en el mosaico
    for index, (img, title) in enumerate(zip(images, titles)):
        # Añado el subplot del mosaico para esta imagen concreta en
        # la posicion que le corresponde
        # Ademas le estamos pasando los titulos dados por parametro
        fig.add_subplot(rows, columns, index + 1, title = title)

        # Añado la imagen a la posicion especificada
        # Tenemos que cambiar los canales si la imagen es tricolor, pues
        # estamos usando formato cv2 BGR
        if is_black_and_white(img) == True:
            plt.imshow(img)
        else:
            plt.imshow(img[:,:,::-1])

    # Para el espaciado entre filas
    # Esta es la orden que menciono en el enlace al problema con el espaciado
    plt.subplots_adjust(top = 0.5, bottom=0.2, hspace=0.5, wspace=0.5)

    # Mostramos la composicion y espero en caso de estar en local
    plt.show()
