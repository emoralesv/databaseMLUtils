
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

if True:
    views = ["RGB", "VEG", "LBP", "NGRDI", "GRRI","MGRVI","VDVI","VARI","ENTROPY","LBP"]
    convert_xml_to_Classification(r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\tomate_rugoso\train",
                                r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\views",
                                List_transforms_ids=views, test=False)
                               
from databaseMLUtils.reporting import make_dataset_report


make_dataset_report(
    data=r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\views\RGB",
    name="Tomate Rugoso",
    url="Rugose",
    out=r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\report",
)




