import skimage.io
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
import cupy as cp
from cupyx.scipy.ndimage import filters
adjustment_values = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)
multispectral = skimage.io.imread('multispectral.tiff', plugin='tifffile')
panchromatic = skimage.io.imread('panchromatic.tiff', plugin='tifffile')
hsv = rgb2hsv(multispectral)
val = hsv[:,:,2]
sat = hsv[:,:,1]
mat = hsv[:,:,0]
pani = hist_match(panchromatic,val)
s = np.array([[1/256, 1/64, 3/128, 1/64, 1/256],[1/64, 1/16, 3/32, 1/16, 1/64],[3/128, 3/32, 9/64, 3/32, 3/128],[1/64, 1/16, 3/32, 1/16, 1/64],[1/256, 1/64, 3/128, 1/64, 1/256]])
s_gpu = cp.array(s)
p_gpu = cp.array(pani)
I1_gpu = filters.correlate(p_gpu, s_gpu, mode='constant')
s1 = np.array([[1/256, 0, 1/64, 0, 3/128, 0, 1/64, 0, 1/256],[0, 0, 0, 0, 0, 0, 0, 0, 0],[1/64, 0, 1/16, 0, 3/32, 0, 1/16, 0, 1/64],[0, 0, 0, 0, 0, 0, 0, 0, 0],[3/128, 0, 3/32, 0, 9/64, 0, 3/32, 0, 3/128],[0, 0, 0, 0, 0, 0, 0, 0, 0],[1/64, 0, 1/16, 0, 3/32, 0, 1/16, 0, 1/64], [0, 0, 0, 0, 0, 0, 0, 0, 0],[1/256, 0, 1/64, 0, 3/128, 0, 1/64, 0, 1/256]])
s1_gpu = cp.array(s1)
I2_gpu = filters.correlate(I1_gpu, s1_gpu, mode='constant')
W1=(pani-I1_gpu.get())
W1_gpu = gpuarray.to_gpu(W1)
W1_gpu_new = gpuarray.empty_like(W1_gpu)
adjustment_values(W1_gpu,W1_gpu_new)
W1 = W1_gpu_new.get().astype(np.uint8)
W2=(I1_gpu.get()-I2_gpu.get())
W2_gpu = gpuarray.to_gpu(W2)
W2_gpu_new = gpuarray.empty_like(W2_gpu)
adjustment_values(W2_gpu,W2_gpu_new)
W2 = W2_gpu_new.get().astype(np.uint8)
nint=(panchromatic+W1+W2).astype(np.uint8)
n_hsv = np.stack((mat, sat, nint),axis=2)
fusioned_image = hsv2rgb(n_hsv).astype(np.uint8)
skimage.io.imsave('atrousgpu_image.tif',fusioned_image, plugin='tifffile')
