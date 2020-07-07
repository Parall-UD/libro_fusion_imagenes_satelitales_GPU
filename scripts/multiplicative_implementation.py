import skimage.io
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
from pycuda.elementwise import ElementwiseKernel
def step_1(color_matrix, image_matrix):
    matrix_sal = linalg.multiply(color_matrix, image_matrix)
    return matrix_sal
def step_2(matrix_1):
    mat_max = np.amax(matrix_1)
    mat_min = np.amin(matrix_1)
    return mat_max, mat_min
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *z",
        "z[i] = ((x[i]-a)*255)/(b-a)",
        "linear_combination")
def step_3(matrix_1, matrix_color, mat_max, mat_min):
    lin_comb(mat_min, matrix_1, mat_max, matrix_color)
    return matrix_color.get()
multispectral = skimage.io.imread('multispectral.tiff', plugin='tifffile')
panchromatic = skimage.io.imread('panchromatic.tiff', plugin='tifffile')
multispectral = multispectral.astype(np.float32)
r = multispectral[:,:,0].astype(np.float32)
g = multispectral[:,:,1].astype(np.float32)
b = multispectral[:,:,2].astype(np.float32)
panchromatic = panchromatic.astype(np.float32)
r_gpu = gpuarray.to_gpu(r)
g_gpu = gpuarray.to_gpu(g)
b_gpu = gpuarray.to_gpu(b)
panchromatic_gpu = gpuarray.to_gpu(panchromatic)
linalg.init()
m33_gpu = step_1(r_gpu, panchromatic_gpu)
m44_gpu = step_1(g_gpu, panchromatic_gpu)
m55_gpu = step_1(b_gpu, panchromatic_gpu)
Amax, Amin = step_2(m33_gpu.get())
br_gpu = gpuarray.empty_like(r_gpu)
br_host = step_3(m33_gpu, br_gpu, Amax, Amin)
Amax, Amin = step_2(m44_gpu.get())
bg_gpu = gpuarray.empty_like(g_gpu)
bg_host = step_3(m44_gpu, bg_gpu, Amax, Amin)
Amax, Amin = step_2(m55_gpu.get())
bb_gpu = gpuarray.empty_like(b_gpu)
bb_host = step_3(m55_gpu, bb_gpu, Amax, Amin)
brr = br_host.astype(np.uint8)
bgg = bg_host.astype(np.uint8)
bbb = bb_host.astype(np.uint8)
fusioned_image = np.stack((brr, bgg, bbb),axis=2)
skimage.io.imsave('multiplicativegpu_image.tif',fusioned_image, plugin='tifffile')
