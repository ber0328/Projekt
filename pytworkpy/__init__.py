"""
    Toto je pouze malá knihovna, jejíž jedinný cíl je do sebe zabalit jednoduchou neuronovou síť schopnou rozpoznávat ručně psaná čísla. 
    Knihovna obsahuje dva soubory:
        network.py -- soubor zodpovědný za práci s neuronovou sítí
        visual.py -- obstarává vizualizaci dat (Např. vývoj chybové funkce při trénování, vykreslení parametrů sítě0

    Demonstrace funkčností jsou v souboru "examples.ipynb".

    Uživatel využije tyto třídy:
        Network -- třída zabalující a o něco usnadňující práci s konkrétní neuronovou sítí.
        Visual -- třída obsahující nástroje pro grafické znázorňění neuronové sítě.
"""

from .network import Network, SimpleMNISTClassifier
from .visual import Visual

__all__: list[str] = ["Network", "SimpleMNISTClassifier", "Visual"]