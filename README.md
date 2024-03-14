# AgingGAN

Project attempting the aging and reverse aging of faces with generative adversarial networks (GANs), and corresponding interpretation.

[Data found here](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped?resource=download) 



TO RUN CYCLEGAN, AS CLONED HERE:

- download the data from above link
- run "split_data.py" (splits the data according to age groups, moves the data into the appropriate cyclegan subdirectory)
- train a network within the cyclegan directory using "!python train.py --dataroot ./datasets/young2old_cropped --name young2old_cropped --model cycle_gan --display_id -1" (change around arguments to match your data and model name, along with other parameters)