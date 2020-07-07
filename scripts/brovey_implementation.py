import skimage.io
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from pycuda.elementwise import ElementwiseKernel
def step_1(matrix_color, matrix_suma):
    matrix_1 = gpuarray.if_positive(matrix_suma,(3*matrix_color)/matrix_suma,matrix_suma)
    return matrix_1
def step_2(matrix_1, matrix_image_pan):
    matrix_2 = linalg.multiply(matrix_1, matrix_image_pan)
    return matrix_2
def step_3(matrix_1):
    mat_max = np.amax(matrix_1.get())
    mat_min = np.amin(matrix_1.get())
    return mat_max, mat_min
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *z",
        "z[i] = ((x[i]-a)*255)/(b-a)",
        "linear_combination")
def step_4(matrix_1, matrix_color, mat_max, mat_min):
    lin_comb(mat_min, matrix_1, mat_max, matrix_color)
    return matrix_color
multispectral = skimage.io.imread('multispectral.tiff', plugin='tifffile')
panchromatic = skimage.io.imread('panchromatic.tiff', plugin='tifffile')
multispectral = multispectral.astype(np.float32)
r = multispectral[:,:,0].astype(np.float32)
g = multispectral[:,:,1].astype(np.float32)
b = multispectral[:,:,2].astype(np.float32)
panchromatic = panchromatic.astype(np.float32)
msuma = r+g+b
r_gpu = gpuarray.to_gpu(r)
g_gpu = gpuarray.to_gpu(g)
b_gpu = gpuarray.to_gpu(b)
panchromatic_gpu = gpuarray.to_gpu(panchromatic)
msuma_gpu = gpuarray.to_gpu(msuma)
linalg.init()
m11_gpu = step_1(r_gpu, msuma_gpu)
m22_gpu = step_2(m11_gpu, panchromatic_gpu)
m33_gpu = step_1(b_gpu, msuma_gpu)
m44_gpu = step_2(m33_gpu, panchromatic_gpu)
m55_gpu = step_1(g_gpu, msuma_gpu)
m66_gpu = step_2(m55_gpu, panchromatic_gpu)
Amax_host, Amin_host = step_3(m22_gpu)
rr_gpu = gpuarray.empty_like(r_gpu)
step_4(m22_gpu, rr_gpu, Amax_host, Amin_host)
Amax_host, Amin_host = step_3(m66_gpu)
gg_gpu = gpuarray.empty_like(g_gpu)
step_4(m66_gpu, gg_gpu, Amax_host, Amin_host)
Amax_host, Amin_host = step_3(m44_gpu)
bb_gpu = gpuarray.empty_like(b_gpu)
step_4(m44_gpu, bb_gpu, Amax_host, Amin_host)
ggg_host = gg_gpu.get().astype(np.uint8)
rrr_host = rr_gpu.get().astype(np.uint8)
bbb_host = bb_gpu.get().astype(np.uint8)
fusioned_image = np.stack((rrr_host, ggg_host, bbb_host),axis=2)
skimage.io.imsave('broveygpu_image.tif',fusioned_image, plugin='tifffile')
