import streamlit as st
from PIL import Image
import numpy as np

from util import util
#import numpy as np

from data import create_dataset
from models import create_model
from options.test_options import TestOptions

import torchvision.transforms as T

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()
opt.dataroot = 'test_img'

# Function to process the image
def get_processed(image):
    # Placeholder function, replace this with your actual image processing logic
    # Here, we are simply converting the image to grayscale
    image.save('test_img/testA/test_img.jpg')
    image.save('test_img/testB/test_img.jpg')
    dataset = create_dataset(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()
        out = visuals['fake_B']
        print(util.tensor2im(out).shape)
    return util.tensor2im(out)

def main():
    st.title("Multiple Image Processing Application")
    
    uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_images:
        col_count = 3
        row_count = (len(uploaded_images) + col_count - 1) // col_count
        
        for i, uploaded_image in enumerate(uploaded_images):
            st.sidebar.image(uploaded_image, width=100, caption=f"Image {i+1}")
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button('Process Images'):
            for uploaded_image in uploaded_images:
                original_image = Image.open(uploaded_image)
                st.image(original_image, caption='Original Image', width=150)
                
                processed_image = get_processed(original_image)
                st.image(processed_image, caption='Processed Image', width=150)

if __name__ == '__main__':
    main()