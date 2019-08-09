In this project I look forward to training a writer.

First I take take a document and calculate the number of different characters in there.
Divide the text into sections of characters and one-hot encode all the characters in all the sections.
Label the sections with the characters that follow the last character in the section.
And pass this labeled data to my model.

The model has an output layer and two hidden lstm layer with the state at time step n updated as follows:
Sn = U * Sn-1 + (1 - U) * [ input * Wi + R * (Output * Wo) + B ]
where input is the current input, output is the output from the previous time step, U and R are scalars between 0 and 1,
and Wi and Wo are weights and B is the bias.

After the final time step, which is equal to the length of a section, we pass the output of the lstm layer into the
output layer, which has a softmax activation function.
Then compute the loss function which is the cross entropy loss function and take the mean accross all the batches
, since we are training in batches.
Then pass this mean to the gradient descent optimizer to update the weights and biases accordingly.

Questions:
I want to know if everything is right for me to carry out an experiment with the model or there's still moore work to be done?
If so then what?
Also general feedback and recomendations.
