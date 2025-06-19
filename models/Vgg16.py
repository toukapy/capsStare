from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model

def vgg16_model():
    img_input = Input(shape=(224, 224, 3), name='img_input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # base_output = MaxPooling2D((2, 2), strides = (2, 2), name = 'block5_pool')(x)

    # base_model =Model(inputs = img_input, outputs = base_output)
    # base_model.trainable=False

    # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                        WEIGHTS_PATH_NO_TOP,
    #                        cache_subdir='models')

    # base_model.load_weights(weights_path)


    output = Flatten(name='flatten')(x)
    # output = Dense(4096, kernel_initializer='normal',activation='relu',name='fc1')(output)
    # output = Dropout(0.5)(output)
    # output = Dense(4096, kernel_initializer='normal', name='fc2')(output)
    # output=BatchNormalization()(output)
    # output=Activation('relu')(output)
    # output = Dropout(0.5)(output)
    predict = Dense(2, kernel_initializer='normal', activation='linear', name='predict')(output)

    # Create your own model
    mymodel = Model(inputs=img_input, outputs=predict)

    return mymodel