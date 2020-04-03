from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2          # 匯入 OpenCV 影像處理套件
import glob         # 匯入內建的檔案與資料夾查詢套件

# 載入 MNIST 及 3 種自製手寫圖片
# pre_adjust: 是否做前置影像調整
def load_dataset(imgs_fd, pre_adjust=True):
    imgsets = [None]
    labsets = [None]
    (_, _), (imgsets[0], labsets[0]) = mnist.load_data() # 固定要載入 MNIST 的測試資料
    for fd in imgs_fd[1:]:   # 走訪每個指定的圖檔資料夾 (跳過第 0 個 MNIST)
        imgs, labs = read_from_fd(fd, pre_adjust)   # 由資料夾中載入圖檔並做適當影像處理
        imgsets.append(imgs)
        labsets.append(labs)
    return imgsets, labsets

# 由 path 載入圖檔, 並以子資料夾名為標籤 (0~9的資料夾)
# pre_adjust: 是否做前置影像調整
def read_from_fd(path, pre_adjust=True):
    test_images=[]
    test_labels=[]
    for label in range(10):
        files = glob.glob(path + "/" + str(label) + "/*.png" )
        for file in files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #轉灰階
            img = cv2.bitwise_not(img)       # 反白：變成黑底白字
            img = cv2.resize(img, (28, 28))  # 重設大小為 28x28

            if pre_adjust: img = img_pre_adjust(img)  # 影像前置調整

            test_images.append(img)
            test_labels.append(label)
    return (np.array(test_images), np.array(test_labels))  # 轉為 numpy 陣列


#將單一影像做去背、稍微模糊、及修正亮度 (黑底白字的字太暗)
def img_pre_adjust(img):
    _, img = cv2.threshold(img,  40,   0, cv2.THRESH_TOZERO) #去背：深灰變黑色 (低於30的變0)
    img = cv2.blur(img,(2, 2))  # 平均模糊
    img = img_fix_dark(img)  #修正反白後的圖片太暗
    return img

#修正影像可能太暗的問題 (黑底白字的字太暗)
def img_fix_dark(img):
    imax = np.max(img)
    idx = img >= imax
    while np.sum(idx) < 30:  #希望全白至少要有 30 個像素
        imax -= 1
        if imax <= 30: break
        idx = img >= imax
    img[idx] = 255   # 調成全白
    img[(~idx) & (img>30)] += (255-imax)  # 其他也跟著調高, 但 <=20視為雜訊不調
    return img

# 資料預設理, mode: 'D'-處理成 DNN 適用, 'C'-處理成 CNN 適用
def pre_proc(imgs, labs, model):
    if imgs != np.ndarray: imgs = np.array(imgs)
    if len(model.input_shape) == 2:
        x = imgs.reshape((len(imgs), 784))  # for DNN
    else:
        x = imgs.reshape((len(imgs), 28, 28, 1))  # for CNN
    x = x.astype('float32') / 255
    y  = to_categorical(labs)
    return x, y


# 傳回圖片 (上,下,左,右) 的空行數 (像素數)
def img_getBorder(img):
    top = bot = lef = rig = -1
    rr = img.shape[0]
    cc = img.shape[1]
    for r in range(rr):
        for c in range(cc):
            if(img[r, c] != 0):
                top = r
                break
        if top != -1: break
    for r in range(rr-1, 0, -1):
        for c in range(cc):
            if(img[r, c] != 0):
                bot = rr-1-r
                break
        if bot != -1: break
    for c in range(cc):
        for r in range(rr):
            if(img[r, c] != 0):
                lef = c
                break
        if lef != -1: break
    for c in range(cc-1, 0, -1):
        for r in range(rr):
            if(img[r, c] != 0):
                rig = cc-1-c
                break
        if rig != -1: break
    return (top, bot, lef, rig)

# 將圖片顯示出來, 每行 10 張圖, 最多 240 張圖
# 參數依序為：圖片資料集,標籤資料集,開始顯示的索引,顯示的圖片數量,預測的答案資料集
# 若 num 為 0 則顯示由 start 開始的全部圖片
def showImgs(imgs, labs, start=0, num=0, predicts=[], by0123=True):
    max_num = len(imgs)-start
    if max_num > 240: max_num = 240  # 最多只顯示 240 張圖
    if num <= 0 or num > max_num: num = max_num
    plt.gcf().set_size_inches(16, 52 if len(predicts) else 40)
    idx_list = get_idxs(imgs, labs, start, num, by0123)
    for i in range(num):
        ax = plt.subplot(24, 10, 1+i)
        idx = idx_list[i]
        ax.imshow(imgs[idx], cmap='gray_r',   #反白顯示 (白底黑字)
                  norm=plt.Normalize(0.0, 255.0))  #指定灰階的範圍
        if len(predicts):
            title = 'label = ' + str(labs[idx])
            if labs[idx] == predicts[idx]:
                title += '\npredi = ' + str(predicts[idx])
            else:
                title += '\npre● = ' + str(predicts[idx])
            ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]) # X, Y 軸不顯示刻度
    plt.show()

def get_idxs(imgs, labs, start, num, by0123):
    if by0123 == False:
        return range(start, num)
    size = len(imgs)
    idx_list = []
    idx_read = [0 for i in range(10)]
    for i in range(size):
        n = i % 10
        for j in range(idx_read[n], size):
            if labs[j] == n:
                idx_list.append(j)
                idx_read[n] = j + 1
                break
        else:
            idx_read[n] = size
    if start + num > len(idx_list):
        return idx_list[start: ]
    else:
        return idx_list[start: num]

#######################################################

# 移動數字位置
# up:   向上 (正值) 或向下 (負值) 移動多少點
# left: 向左 (正值) 或向右 (負值) 移動多少點
def img_shift(img, up=0, left=0):
    if up > 0:     # 上移
        img = img[up:]          # 切上補下面
        img = cv2.copyMakeBorder(img,0,up,0,0,cv2.BORDER_CONSTANT,value=0)
    elif up < 0:   # 下移
        img = img[:up]         # 切下補上面
        img = cv2.copyMakeBorder(img,-up,0,0,0,cv2.BORDER_CONSTANT,value=0)
    if left > 0:   # 左移
        img = img[:, left:]    # 切左補右邊
        img = cv2.copyMakeBorder(img,0,0,0,left,cv2.BORDER_CONSTANT,value=0)
    elif left < 0: # 右移
        img = img[:, :left]   # 切右補左邊
        img = cv2.copyMakeBorder(img,0,0,-left,0,cv2.BORDER_CONSTANT,value=0)
    return img

# 將影像放大或縮小
# add: 放大(正值) 或縮小 (負值) 多少點
def img_bigger(img, add=0):
    if add > 0: #bigger
        img = cv2.resize(img, (28+add, 28+add))
        u = add // 2
        d = add - u
        img = img[u:-d, u:-d]
    elif add < 0:  # Shrink
        img = cv2.resize(img, (28+add, 28+add))
        d = (-add) // 2
        u = (-add) - d
        img = cv2.copyMakeBorder(img,u,d,u,d,cv2.BORDER_CONSTANT,value=0)
    return img

# 銳利化
# mode: 1-粗銳利化, 2-細銳利化
# times: 重複做幾次
def img_sharper(img, mode=1, times=1):
    if mode==1:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    else:
        kernel = np.array([[-0.125,-0.125,-0.125,-0.125,-0.125],
                           [-0.125,  0.25,  0.25,  0.25,-0.125],
                           [-0.125,  0.25,     1,  0.25,-0.125],
                           [-0.125,  0.25,  0.25,  0.25,-0.125],
                           [-0.125,-0.125,-0.125,-0.125,-0.125]])
    for i in range(times):
        img = cv2.filter2D(img, -1, kernel=kernel)
    return img

# 將線條加粗或變細
# add: 加粗 (正值) 或變細 (負值) 多少點
def img_thicker(img, add=1):
    if(add>0):   # 變粗
        kernel = np.ones((add, add), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    elif(add<0): #變細
        kernel = np.ones((-add, -add), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    return img

# 將線條加粗或變細
# add: 加粗 (正值) 或變細 (負值) 多少點
def img_threshold(img, th=220, low=0, high=-1):
    idx = img <= th
    if low >=0: img[idx] = low
    if high >= 0: img[~idx] = high
    return img

# 將線條加深或變淺
# add: 加深 (正值) 或變淺 (負值) 多少灰階值
def img_darker(img, add=2):
    if add > 0:
        idx = img >= (255-add)
        img[idx] = 255
        img[(~idx) & (img>20)] += add  # <=20視為雜訊, 不加 add
    elif add < 0:
        add  = -add
        idx = img <= add
        img[idx] = 0
        img[~idx] -= add
    return img

# 調整全部影像為固定大小, 置中
# size: 長及寬都調為 size 像素
# vdif,hdif 為垂直(上下),水平(左右)允許差異的像點數
#    例如 vdif=1 則上留白與下留白只允許差 1 像素, 超過 1 即進行調整
def img_best(img, size=20, vdif=1, hdif=1):
    if size <= 0: return img

    # 1.先將影像上下左右置中
    img = img_center(img, vdif, hdif)

    # 2.調到 上+下高度 == 8
    a, b, c, d = img_getBorder(img)
    border = 28-size
    if a+b < border:  # 空太少, 圖太大, 要縮小
        img = img_bigger(img, -(border-(a+b)+2))
    elif a+b > border and c+d > border: # 空太大, 圖太小, 要放大
        img = img_bigger(img, min(a+b-border, c+d-border))

    # 3.再次將影像上下左右置中
    img = img_center(img, vdif, hdif)

    return img

# 將影像中的數字置中
# 參數請見上一函式的說明
def img_center(img, vdif=1, hdif=1):
    h = 1 if hdif>1 else 0
    v = 1 if vdif>1 else 0

    a, b, c, d = img_getBorder(img)
    #1.左右置中
    if c-d > hdif:
        img = img_shift(img, up=0, left=(c-d-h)//2)
    elif d-c > hdif:
        img = img_shift(img, up=0, left= -((d-c-h)//2))
#        img = adjImg(img, 'l', -((d-c-h)//2))
    #2.上下置中
    if a-b >= vdif:
#        img = adjImg(img, 'u', (a-b-v)//2)
        img = img_shift(img, up=(a-b-v)//2, left=0)
    elif b-a >= vdif:
#        img = adjImg(img, 'u', -((b-a-v)//2))
        img = img_shift(img, up=-((b-a-v)//2), left=0)
    return img



