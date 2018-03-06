from common.datasets import source_target_dataset
import cv2 as cv
from common.models.transformers import ResNetImageTransformer
from chainer import serializers
import chainer
import os
import shutil
import numpy as np

def main():
    source_dataset_path = "../cowc_selwyn" #specify image dir directly
    output_dir = "experiment_data/NTT_fake_GT_L1_l10"
    transformer_path = "experiment_data/selwyn2tokyo_GT_L1_l10/gen_g200000.npz"
    gpu = 0

    data_usage = "train"

    dirs =[]
    dirs += [output_dir]
    output_dir_image = os.path.join(output_dir,data_usage)
    dirs += [output_dir_image]
    output_dir_cmp = os.path.join(output_dir_image,"cmp")
    dirs += [output_dir_cmp]
    output_dir_cmp_bbox = os.path.join(output_dir_image, "cmp_bbox")
    dirs += [output_dir_cmp_bbox]
    output_dir_list = os.path.join(output_dir, "list")
    dirs += [output_dir_list]
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)

    if os.path.isfile(os.path.join(output_dir,'../list',data_usage+'.txt')):
        shutil.copyfile(os.path.join(output_dir,'../list',data_usage+'.txt'),os.path.join(output_dir_list,data_usage+'.txt'))
        make_list = False
    else:
        make_list = True

    transformer = ResNetImageTransformer()
    serializers.load_npz(transformer_path,transformer)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        transformer.to_gpu()

    source_dataset = source_target_dataset(source_dataset_path,source_dataset_path)

    num_files = source_dataset.len_A

    for i in range(num_files):
        imgA, imgA_gt_map, imgB, bboxes, labels = source_dataset.get_example(i)
        bboxes = bboxes.astype(np.int32)
        img_original = ((imgA * 0.5 + 0.5) * 255).transpose(1,2,0).astype(np.uint8)
        img_original = cv.cvtColor(img_original, cv.COLOR_RGB2BGR)
        if gpu >= 0: imgA = chainer.cuda.to_gpu(imgA[None])
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            img_target = transformer(imgA)
        img_target = img_target.data
        if gpu >= 0: img_target = chainer.cuda.to_cpu(img_target)
        b, c, h, w = img_target.shape
        img_target = img_target.reshape(b*c,h,w)
        img_target = ((img_target * 0.5 + 0.5)*255).transpose(1,2,0).astype(np.uint8)
        img_target = cv.cvtColor(img_target, cv.COLOR_RGB2BGR)
        filename_ = source_dataset.get_filename_A(i)
        dir, file_ = os.path.split(filename_)
        file, ext = os.path.splitext(file_)
        shutil.copyfile(os.path.join(dir,file+".txt"),os.path.join(output_dir_image,file+".txt"))
        cv.imwrite(os.path.join(output_dir_image,file+ext),img_target)
        img_cmp = np.empty((h,w*2,c),dtype=np.uint8)
        img_cmp[:,0:w,:] = img_original
        img_cmp[:, w:, :] = img_target
        cv.imwrite(os.path.join(output_dir_cmp, file + ext), img_cmp)
        for bbox in bboxes:
            cv.rectangle(img_original, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0,0,255))
            cv.rectangle(img_target, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255))
        img_cmp[:, 0:w, :] = img_original
        img_cmp[:, w:, :] = img_target
        cv.imwrite(os.path.join(output_dir_cmp_bbox, file + ext), img_cmp)
        if make_list:
            with open(os.path.join(output_dir_list,data_usage+".txt"),mode='a') as f:
                print(file,file=f)
        if i % 100 == 0:
            print('{0} / {1} files finished.'.format(i,num_files))
    print('{0} / {1} files finished.'.format(num_files, num_files))

if __name__ == '__main__':
    main()