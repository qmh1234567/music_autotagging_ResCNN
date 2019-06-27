#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2019/06/21 10:25:17
@Author  :   Four0Eight
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import os
import sys
import time
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from progress.bar import Bar
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
# here put local import lib
from models import music_crnn,ResCNN
import constants as c
import matplotlib.pyplot as plt

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.7


    

def createDatasets():
    # load annotation csv
    df = pd.read_csv(c.ANNOTA_PATH, delimiter='\t')
    mp3_paths = df['mp3_path'].values
    labels = df[c.TAGS].values
   
    # split dataset
    train_paths,val_paths, test_paths= [],[],[]
    train_y,val_y,test_y = [],[],[]
    for index,x in enumerate(df['mp3_path']):
        directory = x.split('/')[0]
        part = int(directory,16)
        if part in range(12):
            train_paths.append(x)
            train_y.append(labels[index])
        elif part is 12:
            val_paths.append(x)
            val_y.append(labels[index])
        elif part in range(13,16):
            test_paths.append(x)
            test_y.append(labels[index])     

    train_dataset = (train_paths, train_y)
    val_dataset = (val_paths, val_y)
    test_dataset = (test_paths, test_y)

    return train_dataset, val_dataset, test_dataset


def loadAllData(dataset):
    (paths, labels) = dataset
    length = len(labels)

    x, y = [], []
    print("Starting loading data...")
    bar = Bar('Processing', max=length, fill='#', suffix='%(percent)d%%')
    for i in range(length):
        bar.next()
        try:
            feat_path = os.path.splitext(paths[i])[0]+'.npy'
            feat_path = os.path.join(c.SAVE_DIR, feat_path)

            # convert (1, 96, 1440) to (96, 1440, 1)
            x_array = np.squeeze(np.load(feat_path))
            x_array = np.expand_dims(x_array, 2)

            x.append(x_array)
            y.append(labels[i])
        except Exception as e:
            print(e)
    bar.finish()
    return (np.array(x), np.array(y))


def dataLoader(dataset):
    (paths, labels) = dataset
    length = len(labels)

    while True:
        # shuffle train data
        shuffle_inx = np.arange(0, length)
        shuffle_inx = np.random.choice(shuffle_inx, size=length, replace=False)
        paths = np.array(paths)[shuffle_inx]
        labels = np.array(labels)[shuffle_inx]
        idx = 0
        while idx < len(labels) - c.BATCH_SIZE:
            x, y = [], []
            while len(x) < c.BATCH_SIZE:
                idx += 1
                try:
                    feat_path = os.path.splitext(paths[idx])[0]+'.npy'
                    feat_path = os.path.join(c.SAVE_DIR, feat_path)
                    x_array = np.squeeze(np.load(feat_path))
                    x_array = np.expand_dims(x_array, 2)
                    x.append(x_array)
                    y.append(labels[idx])
                except Exception as e:
                    print(e)

            yield (np.array(x), np.array(y))


def main(mode):
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    # model = music_crnn((96, 1366, 1), len(c.TAGS))
    model = ResCNN((96,1366,1),len(c.TAGS))
    print(model.summary())
    # exit()
    
    if mode == 'train':
        train_dataset, val_dataset, _ = createDatasets()

        print(f"#SIZE of train dataset: {len(train_dataset[1])}")
        print(f"#SIZE of validation dataset: {len(val_dataset[1])}")
        print("Starting training...")
        optimizer = keras.optimizers.SGD(lr=0.1,momentum=0.9, nesterov=True, decay=1e-6)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpointer_best = ModelCheckpoint(f'{c.CHECKPOINT_DIR}/best.h5', monitor='val_loss', save_best_only=True)
        
        model.fit_generator(dataLoader(train_dataset), epochs=20,
                            steps_per_epoch=len(train_dataset[1]) // c.BATCH_SIZE,
                            validation_data=loadAllData(val_dataset),
                            validation_steps=len(val_dataset[1]) // c.BATCH_SIZE,
                            callbacks=[early_stopping, checkpointer_best])
    else:
        _, __, test_dataset = createDatasets()
        # model.load_weights(f'{c.CHECKPOINT_DIR}/best.h5')
        model.load_weights(f'{c.CHECKPOINT_DIR}/save/epoch_20_score_0.8770.h5')
        start = time.time()
        print(f"#SIZE of test dataset: {len(test_dataset[1])}")
        print("Starting load test dataset....")
        (x, y_true) = loadAllData(test_dataset)
        print(f"Completed, time usage: {time.time()-start}s")

        start = time.time()
        print("Starting predict....")
        y_pre = model.predict(x)
        print(f"Completed, time usage: {time.time()-start}s")

        rocauc = metrics.roc_auc_score(y_true, y_pre)
        print(f'=> Test scores: ROC-AUC={rocauc:.6f}')
        # caculate metrics
        evaluate(model,y_pre,y_true,c.TAGS)
        
def evaluate(model,y_pre,y_true,classes):
    # metrics
    rocauc = metrics.roc_auc_score(y_true,y_pre)
    prauc = metrics.average_precision_score(y_true,y_pre,average='macro')
    y_pred = (y_pre > 0.5).astype(np.float32)
    acc = metrics.accuracy_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true,y_pred,average='samples')
   
    # accuracy
    class_accs = []
    cls_rocaucs = []
    if classes is not None:
        print(f"\n=> Individual scores of {len(classes)} classes")
        for i,cls in enumerate(classes):
            cls_rocauc = metrics.roc_auc_score(y_true[:,i],y_pre[:,i])
            cls_prauc = metrics.average_precision_score(y_true[:,i],y_pre[:,i])
            cls_acc = metrics.accuracy_score(y_true[:,i],y_pred[:,i])
            cls_f1 = metrics.f1_score(y_true[:,i],y_pred[:,i])
            print(f'[{i:2} {cls:30}] rocauc={cls_rocauc:.4f} prauc = {cls_prauc:.4f} acc={cls_acc:.4f} f1={cls_f1:.4f}')
            class_accs.append(cls_acc)
            cls_rocaucs.append(cls_rocauc)
            print()

    np.save('rescnn_spec_accs.npy',np.array(class_accs))

    print(f'=> Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')

    return rocauc,prauc,acc,f1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
        print('Usage: python run.py [run_type]\n',
              '[run_type]: train | test')
        exit()
    mode = sys.argv[1]
    main(str(mode))
    # model = music_crnn((96, 1366, 1), len(c.TAGS))
    # print(model.summary())

