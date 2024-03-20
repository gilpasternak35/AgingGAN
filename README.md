# AgingGAN

Project attempting the aging and reverse aging of faces with generative adversarial networks (GANs), and corresponding interpretation.

[Data found here](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped?resource=download) 

To run the CycleGAN model website, download streamlit and then run:
streamlit run website2.py -- --dataroot test_img --gpu_ids -1 --name young2old_cropped_sa --model cycle_gan --netG self_attention
for the Self-attention cycleGAN model and:
streamlit run website2.py -- --dataroot test_img --gpu_ids -1 --name young2old_cropped --model cycle_gan --epoch 40
for the base cycleGAN model.