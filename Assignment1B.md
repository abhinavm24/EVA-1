## Assignment 1B

### Kernel

Mathematically, Kernel is a matrix of numbers(integers or floats) known as weights. The actual 
value of weights in the kernel are derived through the *learning algorithm* or *learned* via 
the back propagation algorithm.

Number of kernels typically start with 32 and continued with its multiples to avoid memory wastage. Kernels in the initial layers in a CNN detect gross features. Subsequent layers detect more minute and abstract features which usualy exist in many of the larger features of the previous layers.

Hence Kernel can be called **feature extractor**. When convolved over source image, it extracts features to form the output image.




### Channel

Channel is a set of similar features in an image. 

Eg - color red in the image can form a channel. In case of audio recording of vocal with guitar, piano etc., only the piano sound can be a channel.





### Why we mostly use 3x3 kernel

3x3 refers to the kernel size which is the field of view of the convolution, ie, the number of pixels of the input image being looked upon at a time. This filter is preferred over larger ones due to its computational efficiency with lesser number of parameters for the same receptive field.

3x3 kernel used twice is equivalent to a 5x5 kernel having global receptive field of 4.

In the former, parameters are 9+9=18x whereas in the latter, it is 25x



#### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

199x199 input when convolved with 3x3 produces 197x197. 

So each 3x3 convolution reduces dimension by 2.

To reach 1x1, number of layers needed = (199-1)/2 = 99

**We need to perform 3x3 convolution 99 times to reach 1x1 output.**