from skimage.io import imread, imshow, show
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from glob import glob
import pandas as pd

Psnr = []
Ssim = []
index = 1

df = pd.DataFrame(columns=['fake_path', 'real_path', 'Subject Name', 'SSIM', 'PSNR'])
print(df)
fake_files_path = '../cyclegan/results/T1_FLAIR/test_200/images/make_FLAIR/'
fake_files = glob(fake_files_path + '*_fake.png')
real_files_path = 'real/files/path'
real_files = glob(fake_files_path + '*_real.png')

for i in range(len(fake_files)-1):
    # 예측
    print(fake_files[i])
    im2 = imread(fake_files[i], as_gray=True)
    # 원본
    subject_name = fake_files[i].split('\\')[-1].replace('T1','FLAIR').replace('_fake.png','')
    print(subject_name)
    real_file = real_files_path + '\\' + subject_name + '.png'
    im1 = imread(real_file, as_gray=True)
    print(im1.shape)
    print(im2.shape)
    # 계산
    Ssim.append(ssim(im1, im2, multichannel=True))
    print('SSIM : ' + str(ssim(im1, im2, multichannel=True)))
    Psnr.append(psnr(im1, im2))
    print('PSNR : ' + str(psnr(im1, im2)))

    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(Psnr), 4),
            "SSIM: %.4f" % round(np.mean(Ssim), 4)
        )
    index += 1

df.to_csv('test.csv')

print("FINAL",
    "PSNR: %.4f" % round(np.mean(Psnr), 4),
    "SSIM: %.4f" % round(np.mean(Ssim), 4)
)
