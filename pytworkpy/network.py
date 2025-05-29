"""
    Modul pro definici a práci s neuronovou sítí.
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import h5py
from typing import Tuple, List, Dict
from numpy import array


class SimpleMNISTClassifier(nn.Module):
    """
        Tato třída modeluje základní neuronovou síť určenou k rozpoznávání ručně psaných číslic databáze MNIST.
    """
    def __init__(self, ns: Dict[str, Tuple]) -> None:
        """Konstruktor vytvoří vrstvy neuronové sítě."""
        super(SimpleMNISTClassifier, self).__init__()
        self.con1 = nn.Conv2d(ns['con1'][0], ns['con1'][1], ns['con1'][2])
        self.pool = nn.MaxPool2d(ns['pool'][0], ns['pool'][1])
        self.con2 = nn.Conv2d(ns['con2'][0], ns['con2'][1], ns['con2'][2])
        self.lin1 = nn.Linear(ns['lin1'][0], ns['lin1'][1])
        self.lin2 = nn.Linear(ns['lin2'][0], ns['lin2'][1])
        self.lin3 = nn.Linear(ns['lin3'][0], ns['lin3'][1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Tato funkce provede dopřednou propagaci přes síť. Zároveň sestaví výpočetní graf pro zpětnou automatickou derivaci.
        """
        x = self.pool(F.relu(self.con1(x)))
        x = self.pool(F.relu(self.con2(x)))
        x = x.view(-1, self.lin1.in_features)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


class Network:
    """
        Tato třída zabaluje neuronovou síť do jednoho objektu, který poskytuje 
        základní nástroje pro manipulaci s ní.

        Uživatel může volit konkrétní optimizery a loss_fn.
        Defaultně třída pracuje s chybovou funkcí: "torch.nn.CrossEntropyLoss()" a optimizérem "torch.optim.SGD".
    """
    def __init__(self, path: str, model, optimizer, loss_fn) -> None:
        """Konstruktor třídy."""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._create_MNIST_data_loader(path)

    def _create_MNIST_data_loader(self, path: str) -> None:
        """
            Vytvoří data loadery a transformy dat, které bude síť využívat. 

            Tato funkce vytváří ve třídě nové atributy.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        training_set = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
        testing_set = torchvision.datasets.MNIST(path, train=False, transform=transform, download=True)

        training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
        testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=4, shuffle=False)

        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.data_transform = transform

    def _train_epoch(self) -> List[float]:
        """ Tato funkce trénuje model ve třídě Network na datech, které poskytuje 'training_loader'. """
        losses = []

        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if i % 100 == 99:
                losses.append(loss.item())

            if i % 1000 == 99:
                print(f"{i / 150:.2f} % done. Last loss: {loss.item()}")

        return losses

    def test_network(self) -> List:
        """
            Testování sítě.
        """
        losses = []

        with torch.no_grad():
            for i, data in enumerate(self.testing_loader):
                inputs, labels = data
                output = self.model(inputs)
                loss = self.loss_fn(output, labels).item()

                if i % 100 == 99:
                    losses.append(loss)

        return losses

    def train_network(self, epochs=5, sched_step_size=2) -> Dict:
        """
            Trénuje síť po dobu specifikovaných epoch. Vrací tuple tvaru:
            '{epocha: průběhu chybové funkce během trénování}'
        """
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step_size, gamma=0.1)
        losses_per_epoch = {}

        for epoch in range(1, epochs + 1):
            print(f"EPOCH: {epoch}")
            self.model.train(True)
            training_losses = self._train_epoch()
            self.model.eval()
            losses_per_epoch[epoch] = training_losses
            scheduler.step()

        return losses_per_epoch

    def mnist_digit_classify(self, input_path: str) -> None:
        """ Vyhodnotí daný obrázek na síti. """
        img = Image.open(input_path).convert('L')
        img_tensor = self.data_transform(img)

        return max(self.model(img_tensor))

    def save_as_h5(self, path: str) -> None:
        """ Uloží parametry modelu ve formátu hdf5. """
        model_state = self.model.state_dict()

        with h5py.File(path, "w") as hdf5_file:
            for layer, state in model_state.items():
                hdf5_file.create_dataset(layer, data=state)

    def load_from_h5(self, path: str) -> None:
        """ Načte parametry sítě z hdf5 souboru. """
        model_state = h5py.File(path, "r")
        new_model_state = {}

        for layer, state in model_state.items():
            new_model_state[layer] = torch.Tensor(array(state))

        self.model.load_state_dict(new_model_state)
        self.model.eval()
