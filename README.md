
# Image proccessing number 1 Assignment:
### This assignment is the 1th assignment in image proccessing and computer vision course.
in this assignment we gonna do the next things: 
1) read and display image.
2) convert image from RGB to YIQ color spaces and also from YIQ to RGB.
3) we gonna do histogram eqalization on images.
4) also, we gonna do image quantization (reducing the number of colors used in digital image).
5) we gonna implement gamma correction function on images.

## the files for this project:
* ex1_main -> main file provided with the assignment.
* ex1_utils -> contains the functions for doing the things I've mentioned above (for example: image quantization..).
* gamma -> class that contains the gamma correction function.
* "bac_con.png" , "beach.jpg" , "dark.jpg", "water_bear.png"  -> images that comes with the assignment.
* "test1Img.jpg" , "test2Img.jpg"  -> images that i was added.

## the functions :
* transformRGB2YIQ  -> tranform image from RGB color space to YIQ color space using numpy array for display the image.
* transformYIQ2RGB -> tranform image from YIQ color space to RGB color space using numpy array for display the image.
* hsitogramEqualize -> function that perform histogram equalization (enhance the contrast of an image by redistributing the intensity values of the image pixels) of a given grayscale or RGB image.
* quantizeImage -> function that reducing the number of colors used in digital image.
* gammaDisplay -> function that perforn gamma correction with a given γ.

## Requirements && System preferences :
* the system used to make this project is windows 11 
* python version is: 3.8.8, and i was using pycharm workspace.
* libraries used : 
   - open cv
   - matplotlib
   - time
   - warnings
   - sys
   - sklearn
   - typing
