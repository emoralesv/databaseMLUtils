from roboflow import Roboflow
rf = Roboflow(api_key="tOj9s7Mvp12RVGME97Nj")
project = rf.workspace("greenhouse-hnq7q").project("tomate_rugoso_merge-zmhjr")
version = project.version(1)
dataset = version.download("voc")
