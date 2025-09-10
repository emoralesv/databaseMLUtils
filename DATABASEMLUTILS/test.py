
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from databaseMLUtils.transforms import Transformer



transforms = Transformer()
transforms.print()


# Usage
I = Image.open("databases/ToBRF.png")
It = transforms.apply_transforms(['RGB', 'VEG', 'LBP', 'NGRDI', 'GRRI','MGRVI','VDVI','VARI'], I)

transforms.showImages(It,cols = 4)


