
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from databaseMLUtils.transforms import Transformer
from databaseMLUtils.converter import convert_xml_to_Classification


transforms = Transformer()
transforms.print()


# Usage
I = Image.open("databases/ToBRF.png")
It = transforms.apply_transforms(['RGB', 'VEG', 'LBP', 'NGRDI', 'GRRI','MGRVI','VDVI','VARI'], I)



transforms.showImages(It,ncols = 4)


