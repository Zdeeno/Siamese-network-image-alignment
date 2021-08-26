# Alignment

Align images from different seasons using deep learning

## Get datasets

[Testing](https://datasets.chronorobotics.tk/s/QUeUFeUen0942t9)

[Training](https://datasets.chronorobotics.tk/s/aVD7YOTvtOirYhU)

Unzip them etc

## Install

Do `pip3 install torchvision kornia einops`

I changed the parset_nordland (second path) to my dataset.
Create a path in the local folder called results_siam

## Training

Then train the model with python train_siam.py

## Evaluate

In evaluate_model.py, change the model path and the model number (training iterations).
Also in parser_grief, set the dataset path.
The results appear in results folder, along with the errors.csv.

The plots.py can be adjusted with the errors.csv to get the cumulative plots.
