
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


#convert_xml_to_Classification(r"C:\Users\Berries\Documents\Repos\databaseMLUtils\DATABASEMLUTILS\tomate_rugoso_merge-1/train",
                              # r"C:\Users\Berries\Documents\Repos\databaseMLUtils\DATABASEMLUTILS\views", 
                               #List_transforms_ids=["RGB","LBP"], test=True)


from databaseMLUtils.reporting import make_dataset_report


out_dir  = r"C:\Users\Berries\Documents\Repos\databaseMLUtils\DATABASEMLUTILS\report"
make_dataset_report(
    data=r"C:\Users\Berries\Documents\Repos\databaseMLUtils\DATABASEMLUTILS\views\RGB",
    name="Tomate Rugoso Clasificación",
    url=out_dir,
    out="report",
)




