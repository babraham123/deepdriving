model.fit_generator(generator(), samples_per_epoch=50, nb_epoch=10)

def generator(features, labels, batch_size):

 # Create empty arrays to contain batch of features and labels#

 batch_features = np.zeros((batch_size, 64, 64, 3))
 batch_labels = np.zeros((batch_size,1))

 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choice(len(features),1)
     batch_features[i] = some_processing(features[index])
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels