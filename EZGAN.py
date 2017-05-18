    # This is an illustration of a very simple generative adversarial network, built with TensorFlow. It generates images that look like handwritten digits from the MNIST dataset.\n",
    # The code here is written for TensorFlow v0.12, but can be made to run on earlier versions with some quick changes—in particular, replacing `tf.global_variable_initializer()` with `tf.initialize_all_variables()`. This script sends very helpful output to TensorBoard; to make it work with TensorBoard v0.11 and earlier, replace `tf.summary.scalar()` and `tf.summary.image()` with `tf.scalar_summary()` and `tf.image_summary()`, respectively.\n",
    #"- This is a Deep Convolutional GAN\n",
    #"- The generator is a deconvolutional net\n",
    #"- The discriminator is a convolutional net\n",
    #"- Watch a GAN train https://www.youtube.com/watch?v=fN3egtFdA7s\n",

    import tensorflow as tf #machine learning
    import numpy as np #matrix math\n",
    import datetime #logging the time for model checkpoints and training\n",
    import matplotlib.pyplot as plt #visualize results matplotlib

    #Step 1 - Collect dataset\n",
    #MNIST - handwritten character digits ~50K training and validation images + labels, 10K testing\n",
    from tensorflow.examples.tutorials.mnist import input_data
    # will ensure that the correct data has been downloaded to your \n",
    #local training folder and then unpack that data to return a dictionary of DataSet instances.\n",
    mnist = input_data.read_data_sets(\"MNIST_data/\")
   
    # Here's the discriminator network. It takes `x_image` and returns a real/fake classification. As you'll see below, we can either feed `x_image` through a placeholder, or from another tensor—for instance, the output of the generator.\n",
    #This network structure is taken directly from TensorFlow's [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/) tutorial.\n",

    
    def discriminator(x_image, reuse=False):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        # First convolutional and pool layers\n",
        # These search for 32 different 5 x 5 pixel features\n",
        #We’ll start off by passing the image through a convolutional layer. \n",
        #First, we create our weight and bias variables through tf.get_variable. \n",
        #Our first weight matrix (or filter) will be of size 5x5 and will have a output depth of 32. \n",
        #It will be randomly initialized from a normal distribution.\n",
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))\n",
        #tf.constant_init generates tensors with constant values.\n",
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))\n",
        #tf.nn.conv2d() is the Tensorflow’s function for a common convolution.\n",
        #It takes in 4 arguments. The first is the input volume (our 28 x 28 x 1 image in this case). \n",
        #The next argument is the filter/weight matrix. Finally, you can also change the stride and \n",
        #padding of the convolution. Those two values affect the dimensions of the output volume.\n",
        #\"SAME\" tries to pad evenly left and right, but if the amount of columns to be added is odd, \n",
        #it will add the extra column to the right,\n",
        #strides = [batch, height, width, channels]\n",
        d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')\n",
        #add the bias\n",
        d1 = d1 + d_b1\n",
        #squash with nonlinearity (ReLU)\n",
        d1 = tf.nn.relu(d1)\n",
        ##An average pooling layer performs down-sampling by dividing the input into \n",
        #rectangular pooling regions and computing the average of each region. \n",
        #It returns the averages for the pooling regions.\n",
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')\n",
    
        #As with any convolutional neural network, this module is repeated, \n",
        # Second convolutional and pool layers\n",
        # These search for 64 different 5 x 5 pixel features\n",
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))\n",
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))\n",
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')\n",
        d2 = d2 + d_b2\n",
        d2 = tf.nn.relu(d2)\n",
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')\n",
    
         #and then followed by a series of fully connected layers. \n",
        # First fully connected layer\n",
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))\n",
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))\n",
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])\n",
        d3 = tf.matmul(d3, d_w3)\n",
        d3 = d3 + d_b3\n",
        d3 = tf.nn.relu(d3)\n",
    
        #The last fully-connected layer holds the output, such as the class scores.\n",
        # Second fully connected layer\n",
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))\n",
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))\n",
    
        #At the end of the network, we do a final matrix multiply and \n",
        #return the activation value. \n",
        #For those of you comfortable with CNNs, this is just a simple binary classifier. Nothing fancy.\n",
        # Final layer\n",
        d4 = tf.matmul(d3, d_w4) + d_b4\n",
        # d4 dimensions: batch_size x 1\n",

        return d4
   
  }
  {

      

    #And here's the generator. When it's called, it starts by creating a batch of random noise from the latente space $z$, then passes it through a handful of convolutions to produce a 28 x 28 image.\n",

    # This structure is borrowed [from Tim O'Shea](http://www.kdnuggets.com/2016/07/mnist-generative-adversarial-model-keras.html)."
    #You can think of the generator as being a kind of reverse ConvNet. With CNNs, the goal is to \n",
    #transform a 2 or 3 dimensional matrix of pixel values into a single probability. A generator, \n",
    #however, seeks to take a d-dimensional noise vector and upsample it to become a 28 x 28 image. \n",
    #ReLUs are then used to stabilize the outputs of each layer.\n",
    #example of CNN blocks http://cs231n.github.io/convolutional-networks/#fc\n",

    #it takes random inputs, and eventually mapping them down to a [1,28,28] pixel to match the MNIST data shape.  \n",
    #Be begin by generating a dense 14×14 set of values, and then run through a handful of filters of\n",
    #varying sizes and numbers of channels\n",
    #weight matrices get progressively smaller\n",

    def generator(batch_size, z_dim):
        z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
        #first deconv block\n",
        g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features\n",
        g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])
    
        # Generate 25 features
        g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])
    
        # Final convolution with one output channel
        g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + g_b4
        g4 = tf.sigmoid(g4)

        # No batch normalization at the final layer, but we do add\n",
        # a sigmoid activator to make the generated images crisper.\n",
        # Dimensions of g4: batch_size x 28 x 28 x 1\n",
    
        return g4
   

    #"Here we set up our losses and optimizers.\n",

    #"- The upside-down capital delta symbol denotse the gradient of the generator\n",
    #"- m is the number of samples\n",
    #"- Sigma notation tells you to sum up the function evaluated at particular points determined by the little numbers on top and below the big sigma. It is used to add a series of numbers.\n",


    #The gradient ascent expression for the discriminator. The first term corresponds to optimizing the probability that the real data (x) is rated highly. The second term corresponds to optimizing the probability that the generated data G(z) is rated poorly. Notice we apply the gradient to the discriminator, not the generator.\n",

    #"Gradient methods generally work better optimizing log⁡p(x) than p(x) because the gradient of log⁡p(x) is generally more well-scaled. That is, it has a size that consistently and helpfully reflects the objective function's geometry, making it easier to select an appropriate step size and get to the optimum in fewer steps. The computer uses a limited digit floating point representation of fractions, multiplying so many probabilities is guaranteed to be very very close to zero. With log, we don't have this issue.\n",

    #The generator is then optimized in order to increase the probability of the generated data being rated highly.\n",


    #The gradient descent expression for the generator. The term corresponds to optimizing the probability that the generated data G(z) is rated highly. Notice we apply the gradient to the generator network, not the discriminator.\n",

    #"By alternating gradient optimization between the two networks using these expressions on new batches of real and generated data each time, the GAN will slowly converge to producing data that is as realistic as the network is capable of modeling. "


    sess = tf.Session()\n",

    batch_size = 50\n",
    z_dimensions = 100\n",
    
    x_placeholder = tf.placeholder(\"float\", shape = [None,28,28,1], name='x_placeholder')\n",
    # x_placeholder is for feeding input images to the discriminator\n",

    #One of the trickiest parts about understanding GANs is that the loss function is a little bit more complex than that\n",
    #of a traditional CNN classifiers (For those, a simple MSE or Hinge Loss would do the trick). \n",
    #If you think back to the introduction, a GAN can be thought of as a zero sum minimax game. \n",
    #The generator is constantly improving to produce more and more realistic images, while the discriminator is \n",
    #trying to get better and better at distinguishing between real and generated images.\n",
    #This means that we need to formulate loss functions that affect both networks. \n",
    #Let’s take a look at the inputs and outputs of our networks.\n",

    Gz = generator(batch_size, z_dimensions)\n",
    # Gz holds the generated images\n",
    #g(z)\n",
    
    Dx = discriminator(x_placeholder)\n",
    # Dx hold the discriminator's prediction probabilities\n",
    # for real MNIST images\n",
    #d(x)\n",

    Dg = discriminator(Gz, reuse=True)\n",
    # Dg holds discriminator prediction probabilities for generated images\n",
    #d(g(z))\n",

    #So, let’s first think about what we want out of our networks. We want the generator network to create \n",
    #images that will fool the discriminator. The generator wants the discriminator to output a 1 (positive example).\n",
    #Therefore, we want to compute the loss between the Dg and label of 1. This can be done through \n",
    #the tf.nn.sigmoid_cross_entropy_with_logits function. This means that the cross entropy loss will \n",
    #be taken between the two arguments. The \"with_logits\" component means that the function will operate \n",
    #on unscaled values. Basically, this means that instead of using a softmax function to squish the output\n",
    #activations to probability values from 0 to 1, we simply return the unscaled value of the matrix multiplication.\n",
    #Take a look at the last line of our discriminator. There's no softmax or sigmoid layer at the end.\n",
    #The reduce mean function just takes the mean value of all of the components in the matrixx returned \n",
    #by the cross entropy function. This is just a way of reducing the loss to a single scalar value, \n",
    #instead of a vector or matrix.\n",
    #https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks\n",
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))\n",
    
    #Now, let’s think about the discriminator’s point of view. Its goal is to just get the correct labels \n",
    #(output 1 for each MNIST digit and 0 for the generated ones). We’d like to compute the loss between Dx \n",
    #and the correct label of 1 as well as the loss between Dg and the correct label of 0.\n",
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
    d_loss = d_loss_real + d_loss_fake
    
    tvars = tf.trainable_variables()\n",
    
    d_vars = [var for var in tvars if 'd_' in var.name]\n",
    g_vars = [var for var in tvars if 'g_' in var.name]\n",
    
    # Train the discriminator\n",
    # Increasing from 0.001 in GitHub version\n",
    with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:\n",
        #Next, we specify our two optimizers. In today’s era of deep learning, Adam seems to be the\n",
        #best SGD optimizer as it utilizes adaptive learning rates and momentum. \n",
        #We call Adam's minimize function and also specify the variables that we want it to update.\n",
        d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)\n",
        d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)\n",
    
        # Train the generator\n",
        # Decreasing from 0.004 in GitHub version\n",
        g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)"
   

    #Outputs a Summary protocol buffer containing a single scalar value.\n",
    tf.summary.scalar('Generator_loss', g_loss)\n",
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)\n",
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)\n",
    
    d_real_count_ph = tf.placeholder(tf.float32)\n",
    d_fake_count_ph = tf.placeholder(tf.float32)\n",
    "g_count_ph = tf.placeholder(tf.float32)\n",
    "\n",
    "tf.summary.scalar('d_real_count', d_real_count_ph)\n",
    "tf.summary.scalar('d_fake_count', d_fake_count_ph)\n",
    "tf.summary.scalar('g_count', g_count_ph)\n",
    "\n",
    "# Sanity check to see how the discriminator evaluates\n",
    "# generated and real MNIST images\n",
    "d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))\n",
    "d_on_real = tf.reduce_mean(discriminator(x_placeholder))\n",
    "\n",
    "tf.summary.scalar('d_on_generated_eval', d_on_generated)\n",
    "tf.summary.scalar('d_on_real_eval', d_on_real)\n",
    "\n",
    "images_for_tensorboard = generator(batch_size, z_dimensions)\n",
    "tf.summary.image('Generated_images', images_for_tensorboard, 10)\n",
    "merged = tf.summary.merge_all()\n",
    "logdir = \"tensorboard/gan/\"\n",
    "writer = tf.summary.FileWriter(logdir, sess.graph)\n",
    "print(logdir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We want to eventually reach a point where the discriminator correctly classifies nearly all real MNIST images as MNIST images, and classifies generated images as MNIST images about 50% of the time. There are several failure modes that we need to avoid:\n",
    "* **Discriminator losses approach zero**: this leaves practically no gradients for the generator's optimizer.\n",
    "* **Discriminator losses rise unbounded on generated images**: similarly, this leaves practically no gradient for the discriminator to improve, and the generator's training stalls, too, since the gradients it's reading suggest that it has achieved perfect performance.\n",
    "* **Divergent discriminator accuracy**: the discriminator learns a shortcut by either classifying everything as real or everything as generated. You can detect this by checking the discriminator's losses on generated images against the discriminator's losses on real images.\n",
    "\n",
    "To stay balanced between these, we use a controller in the training loop that runs each of the three training operations depending on their losses. Qualitatively speaking, the most rapid improvements in output come when the generator and discriminator are evenly matched; the controller avoids running a training operation when its network shows signs of overpowering the others.\n",
    "\n",
    "Here's our training loop. You'll need a writable directory in your current working directory called `tensorboard` for TensorBoard logs, and another one called `models` to store the five most recent checkpoints.\n",
    "\n",
    "Recognizable results should begin to appear before 10,000 cycles, and will improve after that. On a fast GPU machine, you can make it to 10,000 cycles in less than 10 minutes. It could take around 10 times as long to run on a desktop CPU. There are lots of random numbers involved, so you'll get different results every time you run this. In particular, it's likely to stall for upwards of 2,000 cycles at a time early on, but it should recover on its own."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TRAINING STEP 0 AT 2017-04-25 18:40:06.746826\n",
      "Discriminator classification [-0.00066344]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAGQtJREFUeJztnXmQlOW1xp/DsM6MCCMwoCKL7JuDTECFqCgiUsqSGJYE\nHasERKPRaFUupYmXVPiDUFdTVnIrVbgUhCBKXAIhBmMIQkBEBsIqO4Jsw4AgzADKdu4f09z0Nbzn\njDNDd3vf51c1NT399Ol+55t+5uvu855zRFVBCImPWuleACEkPdD8hEQKzU9IpND8hEQKzU9IpND8\nhEQKzU9IpND8hEQKzU9IpNRO5YPl5ORoo0aNqhy/f//+oNa5c+cq3y8AHDx40NTr1asX1Jo2bWrG\nHj9+3NRzcnJM/fz586Z+8uTJoJadnV3lWAAoLy83deu4AEDdunWD2qlTp8zYs2fPmnqTJk1MvaSk\nJKjVrm0/9fPy8kzdW1utWvZ51dLPnTtnxp4+fTqoHT16FOXl5WLeQYJqmV9EBgF4AUAWgJdUdYp1\n+0aNGuGhhx4K6llZWebjPfvss0Ft5syZZqyIfTyef/55U2/Tpk1QmzBhghn797//3dS/9a1vmfoX\nX3xh6qtXrw5q1113nRm7Zs0aU1+2bJmpt23b1tRbtWoV1NauXWvGHjp0yNTHjh1r6lOnTg1qnrnv\nu+8+Uy8tLTV1759ubm5uUPvss8/M2AMHDgS15557zoxNpsov+0UkC8B/A7gLQBcAo0WkS1XvjxCS\nWqrznr83gO2qulNVTwN4DcDQmlkWIeRSUx3zXwVgT9LPexPX/R9EZLyIFItI8YkTJ6rxcISQmuSS\nf9qvqtNUtVBVC70PtgghqaM65t8HoGXSz1cnriOEfAOojvlXAmgvIm1EpC6AUQDm1cyyCCGXmiqn\n+lT1rIg8CuBdVKT6XlHVjVZMrVq1zBTHoEGDzMe85pprgtrOnTvN2O7du5u6l7Kycu3333+/Gdu/\nf39Tv/3220191apVVda9dNmjjz5q6uvWrTP1yy+/3NTnzp0b1KqbTlu4cKGpT5w4MajNnz/fjD18\n+LCp9+nTx9S9vR1PPfVUUBs/frwZO3jw4KD20ksvmbHJVCvPr6rvAHinOvdBCEkP3N5LSKTQ/IRE\nCs1PSKTQ/IRECs1PSKTQ/IREiqRyYs+1116rv/zlL4O6VwbZoUOHoLZo0SIz9vrrrzd1L1/9l7/8\nJah5teF79+41da8mvmvXrqb+5ZdfBjWvz4FXLrx7925T90qCrf0TnTp1MmO941pWVmbqVj2/18eg\nSxe7QLVnz56mvnTpUlP/61//GtSaN29uxlpre/LJJ7F9+/ZK1fPzzE9IpND8hEQKzU9IpND8hEQK\nzU9IpND8hERKSlt3nzx50kwNeek2q+PqW2+9ZcZ6LcSs7ryAnfLasWOHGfv555+b+uuvv27qjz/+\nuKk//PDDQc1LM/7617829QEDBpi61Z0XsLv/Hj161IwdNWqUqXtpRqursZdOW7Fihak3bNjQ1L32\n21br7/r165uxXiv3ysIzPyGRQvMTEik0PyGRQvMTEik0PyGRQvMTEik0PyGRktKS3mbNmumIESOC\nuqUBdl7Ya73tlVh6k1GtPQZeaaq3B2HcuHGmvnLlSlPv1atXUPP2Tnh//02bNpn6vHn2qIZu3boF\nNa+s1it1vueee0zd2gfgHVPvvr3ny/r1603dOi7eHgRrj8DYsWOxefNmlvQSQsLQ/IRECs1PSKTQ\n/IRECs1PSKTQ/IRECs1PSKRUq55fRHYBKANwDsBZVS20bl+/fn0zJ+6Nkz527FhQu/XWW83Y/Px8\nU/daWFu6V9N+5swZU/dy6d64Z2uUtTe6vHHjxqbuHdc6deqYutXLwOux4I0u/+CDD0y9SZMmVX5s\nrx7fGhcPAO+//76pW62/N240J92jffv2Qc37eyRTE808+quqPcycEJJx8GU/IZFSXfMrgL+JyCoR\nGV8TCyKEpIbqvuzvp6r7RKQZgPdEZLOqLkm+QeKfwnjAf39JCEkd1Trzq+q+xPdSAG8D6H2R20xT\n1UJVLczNza3OwxFCapAqm19EckTksguXAQwEsKGmFkYIubRU52V/PoC3ReTC/byqqgtqZFWEkEtO\nlc2vqjsBXPd1Ys6cOWP2ke/evbsZb9Xsv/3222asVQMNAH369DF1K2eclZVlxnqjx7367ZdfftnU\nrdrygQMHmrFej3jvrdq6detM3fqcp3///masl2tv1qyZqb/44otB7d577zVjrT0lgD1iG/Dz/E88\n8URQ88Z/L168OKh5+xeSYaqPkEih+QmJFJqfkEih+QmJFJqfkEih+QmJlJSO6BYRM7XktTu+++67\ng5qXerFiAWDWrFmmPmzYsKDWqFEjM9ZLaXmlzI899pipz5w5M6h98sknZuyQIUNMvV27dqZutTQH\n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\nRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11f43fda0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Discriminator classification [-0.00072882]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11f3d40f0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Discriminator classification [-0.00070926]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nkUSp+UUSldWtu/fu3Uun5I499thK33fsCG62hBIA2rZtS3O2vTY7bhkAOnbsSPPY1M7GjV9cV/V5\n7Bjunj170tpGjRrRvHHjxjRft24dzdm247Glq7HlyGy6DADmz58fzGJTfbGlzs8//zzNY0ud2ZTc\nlVdeSWtvueWWYFZSUkJry9Mrv0ii1PwiiVLziyRKzS+SKDW/SKLU/CKJUvOLJCqr8/ylpaVYuXJl\nMJ88eTKtf++994LZ1KlTaS1bBgkAq1atovl1110XzJYuXUpr586dS/PYXPuSJUtovmXLlmA2dOjQ\nYAYAEydOpPmtt95K8/POO4/mxx13XDBr0KABrWXLgQGgV69eNL/77ruDWewagebNm1cpnzZtGs0H\nDAiviI8dF//jH/84mM2aNYvWlqdXfpFEqflFEqXmF0mUml8kUWp+kUSp+UUSpeYXSVRW5/lr1apF\n14effvrptJ5tcT1v3jxaGzu6+B//+AfN77zzzmDGtvUGgCZNmtA8djT5wIFfPCT581544YVgdvvt\nt9Pa2PHiTz75JM2/+c1v0pw9ryNGjKC1999/P81j22OzbcUHDx5Ma0888USaf+tb36L5mjVraL51\n69ZgFrv2YsqUKcGsuo/oFpH/h9T8IolS84skSs0vkig1v0ii1PwiiVLziyQqOs9vZg8DOBfAWnfv\nkrltNICrAHy2afvN7s43MkfZGmh25PP7779P69u3bx/MYvvy33XXXTQfMmQIzWfMmBHMYvPNsfX8\nsbny2Of2i1/8IpjF9gKI7WMwZ84cmp900kk0b9iwYTAbP348rY0dg/3RRx/R/IwzzghmN9xwA61d\nvnw5zc866yya33TTTTQfOXJkMOvevTutfemll4JZde/b/wiAA11lcre7d8/8ija+iNQs0eZ395kA\n+JExIvK1U5Wf+a8xs4Vm9rCZNau2EYlIVlS2+f8EoAOA7gBWAwj+QG1mw8ys0MwKv8rPIyJycFWq\n+d292N33uft+AA8C6EH+7jh3L3D3gtgCFxHJnko1v5m1LPfH7wFYVD3DEZFsqchU398A9ANwuJmt\nBHArgH5m1h2AAygCMPwgjlFEDoJo87v7JQe4mU/QBpSUlODFF18M5meeeSatZ2ean3LKKbS2d+/e\nNH/88cdp3rVr12C2fv16Wsv2rgeA2rX5P0NsH/cjjzwymLHnGwA6depE89hZ8a+88grNzSyYxfbt\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11f6c2160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "saved to models/pretrained_gan.ckpt-0\n"
     ]
    }
   ],
   "source": [
    "saver = tf.train.Saver()\n",
    "\n",
    "sess.run(tf.global_variables_initializer())\n",
    "\n",
    "#During every iteration, there will be two updates being made, one to the discriminator and one to the generator. \n",
    "#For the generator update, we’ll feed in a random z vector to the generator and pass that output to the discriminator\n",
    "#to obtain a probability score (this is the Dg variable we specified earlier).\n",
    "#As we remember from our loss function, the cross entropy loss gets minimized, \n",
    "#and only the generator’s weights and biases get updated.\n",
    "#We'll do the same for the discriminator update. We’ll be taking a batch of images \n",
    "#from the mnist variable we created way at the beginning of our program.\n",
    "#These will serve as the positive examples, while the images in the previous section are the negative ones.\n",
    "\n",
    "gLoss = 0\n",
    "dLossFake, dLossReal = 1, 1\n",
    "d_real_count, d_fake_count, g_count = 0, 0, 0\n",
    "for i in range(50000):\n",
    "    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])\n",
    "    if dLossFake > 0.6:\n",
    "        # Train discriminator on generated images\n",
    "        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],\n",
    "                                                    {x_placeholder: real_image_batch})\n",
    "        d_fake_count += 1\n",
    "\n",
    "    if gLoss > 0.5:\n",
    "        # Train the generator\n",
    "        _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],\n",
    "                                                    {x_placeholder: real_image_batch})\n",
    "        g_count += 1\n",
    "\n",
    "    if dLossReal > 0.45:\n",
    "        # If the discriminator classifies real images as fake,\n",
    "        # train discriminator on real values\n",
    "        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],\n",
    "                                                    {x_placeholder: real_image_batch})\n",
    "        d_real_count += 1\n",
    "\n",
    "    if i % 10 == 0:\n",
    "        real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])\n",
    "        summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,\n",
    "                                    d_fake_count_ph: d_fake_count, g_count_ph: g_count})\n",
    "        writer.add_summary(summary, i)\n",
    "        d_real_count, d_fake_count, g_count = 0, 0, 0\n",
    "\n",
    "    if i % 1000 == 0:\n",
    "        # Periodically display a sample image in the notebook\n",
    "        # (These are also being sent to TensorBoard every 10 iterations)\n",
    "        images = sess.run(generator(3, z_dimensions))\n",
    "        d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})\n",
    "        print(\"TRAINING STEP\", i, \"AT\", datetime.datetime.now())\n",
    "        for j in range(3):\n",
    "            print(\"Discriminator classification\", d_result[j])\n",
    "            im = images[j, :, :, 0]\n",
    "            plt.imshow(im.reshape([28, 28]), cmap='Greys')\n",
    "            plt.show()\n",
    "\n",
    "    if i % 5000 == 0:\n",
    "        save_path = saver.save(sess, \"models/pretrained_gan.ckpt\", global_step=i)\n",
    "        print(\"saved to %s\" % save_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let's see some of the images produced by the generator. (The generator has also been sending its images to TensorBoard regularly; click the \"images\" tab in TensorBoard to see them as this runs.)\n",
    "\n",
    "And, as a sanity check, let's look at some real MNIST images and make sure that the discriminator correctly classifies them as real MINST images."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "test_images = sess.run(generator(10, 100))\n",
    "test_eval = sess.run(discriminator(x_placeholder), {x_placeholder: test_images})\n",
    "\n",
    "real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])\n",
    "real_eval = sess.run(discriminator(x_placeholder), {x_placeholder: real_images})\n",
    "\n",
    "# Show discriminator's probabilities for the generated images,\n",
    "# and display the images\n",
    "for i in range(10):\n",
    "    print(test_eval[i])\n",
    "    plt.imshow(test_images[i, :, :, 0], cmap='Greys')\n",
    "    plt.show()\n",
    "\n",
    "# Now do the same for real MNIST images\n",
    "for i in range(10):\n",
    "    print(real_eval[i])\n",
    "    plt.imshow(real_images[i, :, :, 0], cmap='Greys')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One note that I’d like to make is that GANs are notoriously difficult to train. \n",
    "Without the right hyperparameters, network architecture, and training procedure, \n",
    "there is a high chance that either the generator or discriminator will overpower the other. \n",
    "A common case of this is the situation where the generator is able to find a flaw in the discriminator \n",
    "by repeatedly outputting an image that fits the data distribution the discriminator is looking for, \n",
    "but is nowhere close to being a readable MNIST digit. The generator has collapsed onto a single point, \n",
    "and therefore we won’t output a variety of digits. There are also cases where the discriminator becomes\n",
    "too powerful and is able to easily make the distinction between real and fake images.\n",
    "The mathematical intuition behind this phenomenon lies in that GANs are typically trained using gradient \n",
    "descent techniques that are designed to find the minimum value of a cost function, rather than to find \n",
    "the Nash equilibrium of a game. When used to seek for a Nash equilibrium, these algorithms may fail to \n",
    "converge. Further research into game theory and stable optimization techniques may result in GANs that \n",
    "are as easy to train as ConvNets!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
