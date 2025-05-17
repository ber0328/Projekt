"""
    Modul obstarávající práci s grafy.
"""

import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class Visual:
    """
        Tato třída obsahuje metody pro grafické znázorňování sítě. Uživatel si může vytvořit její instanci,
        v níž specifikuje její parametry, které ovlivňují výsledné grafy.
    """
    def __init__(self, figsize: Tuple[int, int], line_col: str, axes_offset: Tuple[float, float, float, float]):
        self.figsize = figsize
        self.line_col = line_col
        self.axes_offset = axes_offset

    def plot_loss(self, epoch: int, loss_per_iteration: List[float]) -> None:
        """ Vykreslí graf průběhu chybové funkce, dle poskytnutých dat. """
        n = range(0, len(loss_per_iteration) * 100, 100)
        fig = plt.figure(figsize=self.figsize)
        axes = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        axes.semilogy(n, loss_per_iteration, color=self.line_col)

        axes.set_xlabel('Number of iterations')
        axes.set_ylabel('Loss')
        axes.set_title(f"Epoch {epoch}")

        plt.show()

    def plot_loss_per_epoch(self, loss_per_epoch: Dict[str, List[float]]) -> None:
        """ Vykreslí chybovou funkci pro více epoch v jedné funkci. """
        for label, data in loss_per_epoch.items():
            self.plot_loss(label, data)

    def plot_layer(self, layer: str, data: torch.Tensor) -> None:
        """ Vykreslí konkrétní vrstvu. """
        fig = plt.figure(figsize=self.figsize)
        axes = fig.add_axes(self.axes_offset)
        axes.set_title(f"{layer} data")
        axes.imshow(data)

    def plot_model_lin_weights(self, model: torch.nn.Module) -> None:
        """ Vykreslí všechny lineární vrstvy v modelu. """
        for layer, data in model.state_dict().items():
            if layer.find("lin") != -1 and layer.find("bias") == -1:
                self.plot_layer(layer, data)

    def plot_output(self, out_tesnor: torch.Tensor) -> None:
        fig = plt.figure(figsize=self.figsize)
        axes = fig.add_axes(self.axes_offset)
        axes.set_title("Output")
        axes.plot(range(10), out_tesnor.detach().numpy().reshape(10, 1))
