from skimage.io import imread, imshow, show
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import pandas as pd

Psnr = []
Ssim = []
index = 1

df = pd.DataFrame(columns=['fake_path', 'real_path', 'Subject Name', 'SSIM', 'PSNR'])
print(df)
fake_files_path = 'fake/flair/file/path'
real_files_path = 'real/files/path'


for i in range(len(fake_files)-1):
    # 예측
    print(fake_files[i])
    im2 = imread(fake_files[i], as_gray=True)
    # 원본
    real_file = real_files_path + '.png'
    im1 = imread(real_file, as_gray=True)
    # 계산
    Ssim.append(ssim(im1, im2, multichannel=True))
    print('SSIM : ' + str(ssim(im1, im2, multichannel=True)))
    Psnr.append(psnr(im1, im2))
    print('PSNR : ' + str(psnr(im1, im2)))

    if np.mod(index, 50) == 0:
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
