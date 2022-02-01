import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import numpy as np
import os
import tensorflow as tf

IMG_SIZE = 128
PATH = os.path.dirname(__file__)

def get_generator_model():

    inputs = tf.keras.layers.Input( shape=( IMG_SIZE , IMG_SIZE , 1 ) )

    conv1 = tf.keras.layers.Conv2D( 16 , kernel_size=( 5 , 5 ) , strides=1 )( inputs )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )

    conv2 = tf.keras.layers.Conv2D( 32 , kernel_size=( 5 , 5 ) , strides=1)( conv1 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )

    conv3 = tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1 )( conv2 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )

    bottleneck = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='tanh' , padding='same' )( conv3 )

    concat_1 = tf.keras.layers.Concatenate()( [ bottleneck , conv3 ] )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_1 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_3 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_3 )

    concat_2 = tf.keras.layers.Concatenate()( [ conv_up_3 , conv2 ] )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_2 )

    concat_3 = tf.keras.layers.Concatenate()( [ conv_up_2 , conv1 ] )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( concat_3 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( conv_up_1 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 3 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu')( conv_up_1 )

    model = tf.keras.models.Model( inputs , conv_up_1 )
    return model


def get_discriminator_model():
    layers = [
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7 , 7 ) , strides=1 , activation='relu' , input_shape=( IMG_SIZE , IMG_SIZE , 3 ) ),
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7, 7 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense( 512, activation='relu'  )  ,
        tf.keras.layers.Dense( 128 , activation='relu' ) ,
        tf.keras.layers.Dense( 16 , activation='relu' ) ,
        tf.keras.layers.Dense( 1 , activation='sigmoid' ) 
    ]
    model = tf.keras.models.Sequential( layers )
    return model



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform( shape=real_output.shape , maxval=0.1 ) , real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform( shape=fake_output.shape , maxval=0.1  ) , fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output , real_y):
    real_y = tf.cast( real_y , 'float32' )
    return mse( fake_output , real_y )


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

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def adjust_brightnes(img,w):
    positions = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i][j])):
                if w == 'l':
                    if img[i][j][k] >=0.85:
                        img[i][j][k] *= 0.90
                        positions.append([i,j,k])
                elif w =='h':
                    for x in positions:
                        img[x[0]][x[1]][x[2]] *= 1.05
    return img


def quality_check(original_image, new_rgb_img):
    out_arr = np.subtract(original_image, new_rgb_img)
    er = 0
    for i in range(len(out_arr)):
        for j in range(len(out_arr[i])):
            for k in range(len(out_arr[i][j])):
                if out_arr[i][j][k] > 30 or out_arr[i][j][k] < -30:
                    er += 1
    er /=196608
    er = round(er,3)
    print(er)
    error.configure(text='Poprawność: ' + str(round(1 - er,3)*100) + '%')
    error.image = 'Poprawność: ' + str(round(1 - er,3)*100)+ '%'



def colorize(gray, rgb, original_size):
    gray = adjust_brightnes(gray,'l')
    generated_image = generator.predict(gray)[0,:,:,:]
    generated_image = adjust_brightnes(generated_image,'h')
    
    image = Image.fromarray((generated_image * 255).astype( 'uint8' ))
    image = image.resize(original_size)
    quality_check(rgb, image)
    global img_r_holder
    img_r_holder = image
    image.save(PATH + "\\results\\result.jpg")
    new_rgb_img = ImageTk.PhotoImage(file=PATH + "\\results\\result.jpg")
    img_r.configure(image=new_rgb_img)
    img_r.image = new_rgb_img

def load_file():
    filename = fd.askopenfilename()
    original_rgb_image = Image.open(filename)
    original_size = original_rgb_image.size
    rgb_image = original_rgb_image.resize((IMG_SIZE,IMG_SIZE))
    gray_image = rgb_image.convert('L')
    gray_img_array = ( np.asarray( gray_image ).reshape( ( 1, IMG_SIZE , IMG_SIZE , 1 ) ) ) / 255


    new_img = ImageTk.PhotoImage(file=filename)
    img_l.configure(image=new_img)
    img_l.image = new_img
    colorize(gray_img_array, original_rgb_image, original_size)


def save_file():
    filename = fd.asksaveasfilename()
    global img_r_holder
    img_r_holder.save(filename)


root = tk.Tk()
root.geometry("1300x700")
root.title('Kolorowanie')
root.resizable(0, 0)


title = tk.Label(root, text="Kolorowanie obrazów", width=65, font=("Arial",25))
title.grid(column=0, row=0, columnspan=2, padx=5, pady=5)

error = tk.Label(root, text="Poprawność : ", width=65, font=("Arial",25))
error.grid(column=0, row=1, columnspan=2, padx=5, pady=5)

left_button = tk.Button(root, text ="Załaduj plik i pokoloruj", command = load_file)
left_button.grid(column=0, row=2,  padx=5, pady=5)

temp_img_l = ImageTk.PhotoImage(file="Colorization\\bin\\blank.jpg")
img_l = tk.Label(root, image=temp_img_l)
img_l.grid(column=0, row=3,  padx=5, pady=5)

right_button = tk.Button(root, text ="Pobierz pokolorowany plik", command = save_file)
right_button.grid(column=1, row=2,  padx=5, pady=5)

global img_r_holder
img_r_holder = None
temp_img_r = ImageTk.PhotoImage(file="Colorization\\bin\\blank.jpg")
img_r = tk.Label(root, image=temp_img_r)
img_r.grid(column=1, row=3,  padx=5, pady=5)



root.mainloop()
