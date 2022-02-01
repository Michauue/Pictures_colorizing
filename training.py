from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Concatenate, Conv2DTranspose, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt
import os
import time
import tensorflow as tf


tf.config.run_functions_eagerly(True)

START_TIME = time.time()
BATCH_SIZE = 10
IMG_SIZE = 128
DATASET_SPLIT = 800
NUM_OF_EPOCHS = 3
NUM_OF_SECONDS = 13680
PATH = os.path.dirname(__file__)
MASTER_DIR = PATH + '\\training_images'


def load_dataset():
    x = []
    y = []
    for image_file in os.listdir( MASTER_DIR )[ 0 : DATASET_SPLIT ]:
        rgb_image = Image.open( os.path.join( MASTER_DIR , image_file ) ).resize( ( IMG_SIZE , IMG_SIZE ) )
        rgb_img_array = (np.asarray( rgb_image ) ) / 255
        gray_image = rgb_image.convert('L')
        gray_img_array = ( np.asarray( gray_image ).reshape( ( IMG_SIZE , IMG_SIZE , 1 ) ) ) / 255
        x.append( gray_img_array )
        y.append( rgb_img_array )

    dataset = tf.data.Dataset.from_tensor_slices((x , y))
    dataset = dataset.batch( BATCH_SIZE )
    return dataset

def get_generator_model():
    inputs = Input( shape=(IMG_SIZE , IMG_SIZE , 1 ))

    conv1 = Conv2D( 16 , kernel_size=(5 , 5) , strides=1 )( inputs )
    conv1 = LeakyReLU()( conv1 )
    conv1 = Conv2D( 32 , kernel_size=(3 , 3) , strides=1)( conv1 )
    conv1 = LeakyReLU()( conv1 )
    conv1 = Conv2D( 32 , kernel_size=(3 , 3) , strides=1)( conv1 )
    conv1 = LeakyReLU()( conv1 )

    conv2 = Conv2D( 32 , kernel_size=(5 , 5) , strides=1)( conv1 )
    conv2 = LeakyReLU()( conv2 )
    conv2 = Conv2D( 64 , kernel_size=(3 , 3) , strides=1 )( conv2 )
    conv2 = LeakyReLU()( conv2 )
    conv2 = Conv2D( 64 , kernel_size=(3 , 3) , strides=1 )( conv2 )
    conv2 = LeakyReLU()( conv2 )

    conv3 = Conv2D( 64 , kernel_size=(5 , 5) , strides=1 )( conv2 )
    conv3 = LeakyReLU()( conv3 )
    conv3 = Conv2D( 128 , kernel_size=(3 , 3) , strides=1 )( conv3 )
    conv3 = LeakyReLU()( conv3 )
    conv3 = Conv2D( 128 , kernel_size=(3 , 3) , strides=1 )( conv3 )
    conv3 = LeakyReLU()( conv3 )

    bottleneck = Conv2D( 128 , kernel_size=(3 , 3) , strides=1 , activation='tanh' , padding='same' )( conv3 )

    concat_1 = Concatenate()( [ bottleneck , conv3 ] )
    conv_up_3 = Conv2DTranspose( 128 , kernel_size=(3 , 3) , strides=1 , activation='relu' )( concat_1 )
    conv_up_3 = Conv2DTranspose( 128 , kernel_size=(3 , 3) , strides=1 , activation='relu' )( conv_up_3 )
    conv_up_3 = Conv2DTranspose( 64 , kernel_size=(5 , 5) , strides=1 , activation='relu' )( conv_up_3 )

    concat_2 = Concatenate()( [ conv_up_3 , conv2 ] )
    conv_up_2 = Conv2DTranspose( 64 , kernel_size=(3 , 3) , strides=1 , activation='relu' )( concat_2 )
    conv_up_2 = Conv2DTranspose( 64 , kernel_size=(3 , 3) , strides=1 , activation='relu' )( conv_up_2 )
    conv_up_2 = Conv2DTranspose( 32 , kernel_size=(5 , 5) , strides=1 , activation='relu' )( conv_up_2 )

    concat_3 = Concatenate()( [ conv_up_2 , conv1 ] )
    conv_up_1 = Conv2DTranspose( 32 , kernel_size=(3 , 3) , strides=1 , activation='relu')( concat_3 )
    conv_up_1 = Conv2DTranspose( 32 , kernel_size=(3 , 3) , strides=1 , activation='relu')( conv_up_1 )
    conv_up_1 = Conv2DTranspose( 3 , kernel_size=(5 , 5) , strides=1 , activation='relu')( conv_up_1 )

    model = tf.keras.models.Model( inputs , conv_up_1 )
    return model


def get_discriminator_model():
    layers = [
        Conv2D( 32 , kernel_size=(7 , 7) , strides=1 , activation='relu' , input_shape=(IMG_SIZE , IMG_SIZE , 3)),
        Conv2D( 32 , kernel_size=(7, 7) , strides=1, activation='relu'),
        MaxPooling2D(),
        Conv2D( 64 , kernel_size=(5 , 5) , strides=1, activation='relu'),
        Conv2D( 64 , kernel_size=(5 , 5) , strides=1, activation='relu'),
        MaxPooling2D(),
        Conv2D( 128 , kernel_size=(3 , 3) , strides=1, activation='relu'),
        Conv2D( 128 , kernel_size=(3 , 3) , strides=1, activation='relu'),
        MaxPooling2D(),
        Conv2D( 256 , kernel_size=(3 , 3) , strides=1, activation='relu'),
        Conv2D( 256 , kernel_size=(3 , 3) , strides=1, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense( 512, activation='relu')  ,
        Dense( 128 , activation='relu') ,
        Dense( 16 , activation='relu') ,
        Dense( 1 , activation='sigmoid') 
    ]
    model = tf.keras.models.Sequential( layers )
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape , maxval=0.1), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape , maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output , real_y):
    real_y = tf.cast( real_y , 'float32')
    return mse( fake_output , real_y )


@tf.function
def train_step( input_x , real_y ):
   
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator( input_x, training=True)
        real_output = discriminator( real_y, training=True)
        generated_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss( generated_images , real_y )
        disc_loss = discriminator_loss( real_output, generated_output )
        
        losses["D"].append(disc_loss.numpy())
        losses["G"].append(gen_loss.numpy())
    
    # Obliczanie gradientów
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optymalizacja
    opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def plot_loss(losses):
    g_loss = []
    d_loss = []
    for i in losses['D']:
        d_loss.append(i)
    for i in losses['G']:
        g_loss.append(i)

    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PATH + 'plots\\plot.jpg')


dataset = load_dataset()
opt = tf.keras.optimizers.Adam( 0.0005 )

generator = get_generator_model()
discriminator = get_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

generator.compile(optimizer=opt, loss=generator_loss, metrics=['accuracy'])
discriminator.compile(optimizer=opt, loss=discriminator_loss, metrics=['accuracy'])


checkpoint_dir = PATH + '\\training_checkpoints\\training_checkpoints4h'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=opt,
                                 discriminator_optimizer=opt,
                                 generator=generator,
                                 discriminator=discriminator)


losses = {"D":[], "G":[]}

# for e in range( NUM_OF_EPOCHS ):
#     i = 0
#     print("Running epoch : ", e )
#     for (x ,y) in dataset:
#         print("     Batch: " + str(i))
#         i+=1
#         train_step(x , y)
#     if (e + 1) % 20 == 0:
#         checkpoint.save(file_prefix = checkpoint_prefix)

e = 0
while time.time()-START_TIME < NUM_OF_SECONDS:
    i = 0
    print("Running epoch : ", e )
    for ( x , y ) in dataset:
        print("Batch: " + str(i))
        train_step( x , y )
        i+= 1
    if (e + 1) % 50 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    e+=1


checkpoint.save(file_prefix = checkpoint_prefix)
plot_loss(losses)
