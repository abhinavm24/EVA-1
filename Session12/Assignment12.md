## Assignment 12a

#### How to improve training speed of Resnet18 on the CIFAR10 dataset on single GPU keeping accuracy at 94%.

**Journey from - 341s to 26s**

**Step1)** 

Repetition of batch norm-ReLU group in initial block removed. Learning rate at epoch 15 made smoother by removing the spike. **Reduction of 18s to 323s.**
Data augmentation - standard data augmentation of padding, clipping and randomly flipping left-right used. Reducing the number of dataloader processes is a further 15s.

Change in calling out to random number generators to select specific data augmentations - reduction of 7s.

Performing all data augmentations in the main process instead of in child process - reduction of 4s

**Training speed after Step1 - 297s**

**Step 2)** 

Increasing batch size from 128 to 512.

Training completes in 256s and with one minor adjustment to the learning rate – increasing it by 10%.

With low learning rate, full (augmented) dataset leads to a lower test loss . Then why was the learning rate increased?

**Training speed after Step2 - 256s**

**Step 3)**

On getting a rough timing profile of current setup (by selectively removing parts of the model and running the remainder), we find that **batch norm computations take a big chunk of time**.

This is solved by converting batch norm weights back to single precision and other weights continue to be in half precision. This problem may be specific to Pytorch 0.4, not tensorflow.

Training speed -186s

Further GPU code optimisation - forward and backward computations perform transposes before and after each convolution, accounting for a significant proportion of the overall run time. This can be made faster by triggering CuDNN libraries which expect data in NHWC format[batch, channels, height, width]. This is not supported currently in Pytorch (which has NCHW format or channel first).

**Speed 161s** - cutting training to 30 epochs by increasing the lr rate from 0-0.4 in 8 epochs instead of 15. 

Additional regularisation of cutout with random 8×8 square increases accuracy to 94.3% in 35 epochs.

With batch_size=768 and epochs=30, **speed is 154s**



**Step 4)**

Orig arch :

![Orig_arch](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/Artboard-1-5.svg)



15 different model architectures were tried using residual blocks, extra blocks of conv-bn-relu, downsampling blocks.

The final model is **Residual L1+L3** where residual branches are added after the first and third layers:, gives 94% accuracy in **24 epochs and training time of 79s**. 

![residualL1L3](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/residualL1L3.svg)



**Step 5)** **Hyper parameter tuning** 

Batch size, learning rate etc.

Weight decay in the presence of batch normalisation acts as a *stable control mechanism* on the effective step size. If gradient updates get too small, weight decay shrinks the weights and boosts gradient step sizes until equilibrium is restored.



**Step 6) Weight decay and learning rate**

A new technique of **Layer-wise Adaptive Rate Scaling** is mentioned which talks of stability in optimal learning rates across architectures. SGD with weight decay provides a useful type of adaptive scaling for the different layers so that each layer receive the same step size in scale invariant units and that this renders manual tuning of learning rates per layer unnecessary.



**Step 7) Batch Normalisation**

**Adv :**

- it stabilises optimisation allowing much higher learning rates and faster training
- it injects noise (through the batch statistics) improving regularisation
- it reduces sensitivity to weight initialisation
- it interacts with weight decay to control the learning rate dynamics

**Disadv :**

- it’s slow (although node fusion can help - whats this??)
- it’s different at training and test time and therefore fragile
- it’s ineffective for small batches and various layer types

In the absence of batch norm, the standard initialisation scheme for deep networks leads to ‘bad’ configurations in which the network effectively computes a constant function.

Gradient of the mean channel outputs are much larger for networks without active batch norm. Here, changes to the distribution of outputs in earlier layers, can propagate to changes in distribution at later layers. In other words, ***internal* covariate shift** propagates to *external* shift at the output layer in absence of batch norm.



**Step 8) Misc. tricks**

Conv-Pool-BN-ReLU instead of Conv-BN-ReLU-Pool

Label Smoothing

CeLU activation

**Ghost Batch Norm** - Batch norm seems to work best with batch size of around 32. But here batch size is 512 to reduce training time. Solution is to apply batch norm separately to subsets of a training batch. This technique, known as ‘ghost’ batch norm, is usually used in a distributed setting but is just as useful when using large batches on a single node. This gives healthy boost where test accuracy is achieved in 18 epochs with training time of 46s.

**Test time augmentation** - In 13 epochs test accuracy 94.6%

At training time, we present the network with a single version of each image – potentially subject to random flipping as data augmentation technique so that different versions are presented on different training epochs. Changed the network by splitting into two identical branches, one of which sees the flipped image, and then merging at the end. Through this lens, the original training can be viewed as a stochastic training procedure for a weight-tied, two branch network in which a single branch is ‘dropped-out’ for each training example.

Finally removing cutout gave **test accuracy 94.1% in 10 epochs of 26s**



## Assignment 12b

###  Tutorial 2: 94% accuracy on Cifar10 in less than 2 min.

While the former was the theory and analysis, this was the actual implementation using Fenwicks library.

Load the dataset and store in Cloud Storage in Tensorflow binary file format TFRecord which is efficient to store and copy.

Standard scaling for every color channel: subtract by mean, and divide by standard deviation 

Data augmentation - pad 4 pixels to 40×40, crop back to 32×32, and randomly flip left and right.

Build the CNN model and train for **24 epochs** on TPU, which is faster than GPU, but initialization is slower. Each epoch takes around 2.5 seconds, with 24 epochs taking ~60s

Optimiser used is with **Stochastic Gradient Descent with Nesterov momentum 0.9, with a slanted triangular learning rate schedule**.

Accuracy on test set after training for 24 epochs in 1 min, was 94%.

