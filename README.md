**0. Main Environments**
- pytorch==2.8.0
- cuda==11.8
- python==3.8

**1. Prepare the dataset.**
- After downloading the datasets, you are supposed to put them into './data/' and , and the file format reference is as follows.

- './data/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
run python predict.py
