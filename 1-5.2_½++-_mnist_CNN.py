from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import util1 as u

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


adj_size = 0   # 指定要將數字調整為多少像素, 設為 0 表示不調整
fns = ''        # 檔名附加訊息

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

u.showImgs(x_train, y_train, 0, 10)

if adj_size > 0:
    fns = '_S' + str(adj_size)
    print(f'調整 MNIST 圖片的數字大小改為 {adj_size} 像素, 並置中')
    import lab_mnist_util as u
    for i in range(len(x_train)):
        if i % 1000 == 0: print(i,end=',')
        x_train[i] = u.img_best(x_train[i], size=adj_size, vdif=1, hdif=1)
    for i in range(len(x_test)):
        if i % 1000 == 0: print(i,end=',')
        x_test[i] = u.img_best(x_test[i], size=adj_size, vdif=1, hdif=1)
    u.showImgs(x_train, y_train, 0, 10)
    print(f'\n調整完成, 開始預處理及訓練模型...')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 預設理
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('對測試資料集的準確率：', score[1])

#程 將模型存檔
model.save('模型_CNN' + fns + '_new.h5')   #← 將模型以指定的檔名存檔
