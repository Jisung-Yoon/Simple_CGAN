from model import *
import tensorflow as tf
import numpy as np

LABEL_SIZE = 10
LATENT_SIZE = 100
SEED = 111  # seed for shuffle data

TRAINING_EPOCHS = 100
BATCH_SIZE = 100
NAME = 'GAN_MNIST'


def main():
    images, labels = load_mnist_datasets(SEED)
    sess = tf.Session()
    model = CGAN(sess, latent_size=LATENT_SIZE, name=NAME)
    test_labels_for_test = np.tile(np.identity(10), (10, 1))
    total_batch = int(len(images) / BATCH_SIZE)

    for epoch in range(TRAINING_EPOCHS):
        for idx in range(total_batch):
            batch_images, batch_labels = \
                images[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE], labels[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            latents = np.random.normal(size=(BATCH_SIZE, LATENT_SIZE))
            model.train(latents, batch_images, batch_labels)

        if epoch % 10 == 0:
            # if you want to generate images using same latent variables,
            # please takes out below lines to outside of loop
            test_latents = np.random.normal(size=(BATCH_SIZE, LATENT_SIZE))
            generated_images = model.generating_images(test_latents, test_labels_for_test)
            reshaped_and_save_images(generated_images, model.result_path, epoch)

    print('Learning Finished')


if __name__ == '__main__':
    main()