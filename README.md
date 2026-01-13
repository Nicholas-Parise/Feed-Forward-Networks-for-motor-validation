# Feed forward neural networks and their applications in motor validation

Investigating the classification performance
of a multi layer feed forward neural network trained
on motor acceleration FFT data. Several experiments were
conducted to determine how different factors influenced generalisation
and accuracy. These experiments included: dataset
size, number of hidden nodes, use of multiple hidden layers,
dynamic learning rates, momentum, and softmax at the output
layer. A stratified three fold training method was used to make
sure test and training sets had the correct proportions of good and
bad motors. To measure the performance, the NN was randomly
initialised and ran with certain hyper parameters, this data was
then averaged, ranked and compared. Results show that larger
hidden layers generally reduced the amount of training needed,
but caused test accuracy to plummet. Experiments were then run
to see if a five layer NN would increase classification, experiments
were run with and without softmax. While they were slightly
better than the three layer NN, it was not marginally better.
Overall the model had classification rates of 74.

![Error Rate](graphs/150_Samples-(25)_37_Hidden_Nodes_(Momentum).png)
