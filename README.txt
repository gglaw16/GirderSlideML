This is branched from auto 

It uses masks for both positive and negatives.
All data is saved in girder.

1: train118.py
   This uses the error maps associated with images to sample a pdf.
   Load a new image after every epoch,
2: predict.py
   Process files create a prediction map and update the pdf.



