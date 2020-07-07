import skimage.io
import numpy as np
from numpy import linalg as la
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda import compiler
import skcuda.misc as misc
from pycuda.elementwise import ElementwiseKernel
kernel_var_cov = """
#include <stdio.h>
__global__ void CovarianceKernel(float *R, float *G, float *B,float *D)
{
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    __shared__ float prueba_salida;
    if (threadIdx.x == 0) prueba_salida = 0;
    float valor_temp = 0;
    float salida_temp[9];
    __syncthreads();
    const int size = 3;
    float arreglo[size];
    arreglo[0] = R[ty * %(BLOCK_SIZE)s + tx];
    arreglo[1] = G[ty * %(BLOCK_SIZE)s + tx];
    arreglo[2] = B[ty * %(BLOCK_SIZE)s + tx];
    __syncthreads();
    for(int k = 0; k < 3; k++){
        for(int h = 0; h < 3; h++){
            valor_temp = arreglo[k]*arreglo[h];
            salida_temp[k*3+h] = valor_temp;
            valor_temp = 0;
        }
    }
    __syncthreads();
   for (int i = 0; i < 9; ++i){
    atomicAdd(&prueba_salida,salida_temp[i]);
    __syncthreads();
    D[i] += prueba_salida;
    __syncthreads();
    prueba_salida = 0.0;
    __syncthreads();
   }
}
"""
kernel_componentes_principales_original = """
#include <stdio.h>
__global__ void componentesPrincipalesOriginal(float *R, float *G, float *B, float *Q, float *S1, float *S2, float *S3)
{
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const int size = 3;
    float salida_temp [size];
    float valor_temp = 0.0;
    float arreglo[size];
    arreglo[0] = R[ty * %(BLOCK_SIZE)s + tx];
    arreglo[1] = G[ty * %(BLOCK_SIZE)s + tx];
    arreglo[2] = B[ty * %(BLOCK_SIZE)s + tx];
    __syncthreads();
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            valor_temp += (Q[i*3+j] * arreglo[j]);
        }
        salida_temp[i] = valor_temp;
        valor_temp = 0.0;
    }
    __syncthreads();
    S1[ty * %(BLOCK_SIZE)s + tx] = salida_temp[0];
    __syncthreads();
    S2[ty * %(BLOCK_SIZE)s + tx] = (-1.0)*salida_temp[1];
    __syncthreads();
    S3[ty * %(BLOCK_SIZE)s + tx] = salida_temp[2];
    __syncthreads();
}
"""
kernel_componentes_principales_pancromatica = """
#include <stdio.h>
__global__ void componentesPrincipalesPancromatica(float *R, float *G, float *B, float *E, float *S1, float *S2, float *S3)
{
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const int size = 3;
    float salida_temp [size];
    float valor_temp = 0.0;
    float arreglo[size];
    arreglo[0] = R[ty * %(BLOCK_SIZE)s + tx];
    arreglo[1] = G[ty * %(BLOCK_SIZE)s + tx];
    arreglo[2] = B[ty * %(BLOCK_SIZE)s + tx];
    __syncthreads();
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            valor_temp += (E[i*3+j] * arreglo[j]);
        }
        salida_temp[i] = valor_temp;
        valor_temp = 0.0;
    }
    __syncthreads();
    S1[ty * %(BLOCK_SIZE)s + tx] = salida_temp[0];
    __syncthreads();
    S2[ty * %(BLOCK_SIZE)s + tx] = salida_temp[1];
    __syncthreads();
    S3[ty * %(BLOCK_SIZE)s + tx] = salida_temp[2];
    __syncthreads();
}
"""
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))
def varianza_cov( R_s, G_s, B_s):
    kernel_code = kernel_var_cov % { 'BLOCK_SIZE': BLOCK_SIZE }
    mod = compiler.SourceModule(kernel_code)
    covariance_kernel = mod.get_function("CovarianceKernel")
    salida_gpu = gpuarray.zeros((3, 3), np.float32)
    Rs_gpu = gpuarray.to_gpu(R_s)
    Gs_gpu = gpuarray.to_gpu(G_s)
    Bs_gpu = gpuarray.to_gpu(B_s)
    for i in range(len(R_s)):
        covariance_kernel(Rs_gpu[i], Gs_gpu[i], Bs_gpu[i],salida_gpu,block = (32, 32, 1)
    return salida_gpu.get()
def stack_values(list_cp, array_split, size, block_size):
    block_size = block_size
    valor_inicial = 0
    valor_final = 0
    list_cp_nueva = []
    factor_div = (size//block_size)
    factor_ite = len(array_split)//factor_div
    for i in range(factor_ite):
        valor_final = valor_final + factor_div
        list_cp_nueva.append(np.hstack(list_cp[valor_inicial:valor_final]))
        valor_inicial = valor_inicial + factor_div
    cp_final = np.vstack(list_cp_nueva)
    return cp_final
def componentes_principales_original(r_s , g_s, b_s, q, size, block_size):
    cp1_temp, cp2_temp,cp3_temp  = []
    size = size
    block_size = block_size
    kernel_code = kernel_componentes_principales_original % { 'BLOCK_SIZE': BLOCK_SIZE }
    mod = compiler.SourceModule(kernel_code)
    kernel = mod.get_function("componentesPrincipalesOriginal")
    s1_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    s2_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    s3_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    q_gpu = gpuarray.to_gpu(q)
    Rs_gpu_t = gpuarray.to_gpu(r_s)
    Gs_gpu_t = gpuarray.to_gpu(g_s)
    Bs_gpu_t = gpuarray.to_gpu(b_s)
    for i in range(len(r_s)):
        kernel(Rs_gpu_t[i], Gs_gpu_t[i], Bs_gpu_t[i], q_gpu,s1_gpu, s2_gpu, s3_gpu, block = (block_size, block_size, 1))
        cp1_temp.append(s1_gpu.get())
        cp2_temp.append(s2_gpu.get())
        cp3_temp.append(s3_gpu.get())
    cp1 = stack_values(cp1_temp, r_s, size, block_size)
    cp2 = stack_values(cp2_temp, r_s, size, block_size)
    cp3 = stack_values(cp3_temp, r_s, size, block_size)
    return cp1, cp2, cp3
def componentes_principales_panchromartic(r_s , g_s, b_s, q, size, block_size):
    block_size = block_size
    nb1_temp, nb2_temp, nb3_temp = []
    size = size
    kernel_code = kernel_componentes_principales_pancromatica % { 'BLOCK_SIZE': BLOCK_SIZE }
    mod = compiler.SourceModule(kernel_code)
    kernel = mod.get_function("componentesPrincipalesPancromatica")
    s1_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    s2_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    s3_gpu = gpuarray.zeros((block_size,block_size),np.float32)
    Rs_gpu_t = gpuarray.to_gpu(r_s)
    Gs_gpu_t = gpuarray.to_gpu(g_s)
    Bs_gpu_t = gpuarray.to_gpu(b_s)
    q_gpu = gpuarray.to_gpu(q)
    for i in range(len(r_s)):
        kernel(Rs_gpu_t[i], Gs_gpu_t[i], Bs_gpu_t[i], q_gpu, s1_gpu, s2_gpu, s3_gpu, block = (block_size, block_size, 1))
        nb1_temp.append(s1_gpu.get())
        nb2_temp.append(s2_gpu.get())
        nb3_temp.append(s3_gpu.get())
    nb1 = stack_values(nb1_temp, g_s, size, block_size)
    nb2 = stack_values(nb2_temp, g_s, size, block_size)
    nb3 = stack_values(nb3_temp, g_s, size, block_size)
    return nb1, nb2, nb3
substract = ElementwiseKernel(
        "float *x, float y, float *z",
        "z[i] = x[i]-y",
        "substract_value")
negative_adjustment = ElementwiseKernel(
        "float *x, float *z",
        "if(x[i] < 0){z[i] = 0.0;}else{z[i] = x[i];}",
        "adjust_value")
def successive_powers(ortogonal_matrix):
    size_mat_ort = len(ortogonal_matrix)
    s = np.zeros((size_mat_ort,1))
    B = np.zeros((size_mat_ort,size_mat_ort))
    for i in range(1, (size_mat_ort+1)):
        B=la.matrix_power(ortogonal_matrix,i)
        s[i-1]=np.trace(B)
    return s
def polynomial_coefficients(polynomial_trace, ortogonal_matrix):
    n_interations = len(ortogonal_matrix)
    polynomial = np.zeros((n_interations))
    polynomial[0] = -polynomial_trace[0]
    for i in range(1,n_interations):
        polynomial[i]=-polynomial_trace[i]/(i+1)
        for j in range(i):
            polynomial[i]=polynomial[i]-(polynomial[j]*(polynomial_trace[(i-j)-1])/(i+1))
    return polynomial
def eigenvectors_norm(mat_eigenvalues, ortogonal_matrix, mat_eigenvectors):
    n = len(mat_eigenvalues)
    V = np.zeros((n,n))
    S = np.zeros((n,1))
    for i in range(n):
        B= ortogonal_matrix[1:n,1:n]-mat_eigenvalues[i,i]*np.eye(n-1)
        temp_s=la.lstsq(B,mat_eigenvectors,rcond=-1)[0].transpose()
        S=np.insert(temp_s,0,1);
        V[0:n,i]=S/la.norm(S)
    return V, V.transpose()
multispectral = skimage.io.imread('multispectral.tiff', plugin='tifffile')
panchromatic = skimage.io.imread('panchromatic.tiff', plugin='tifffile')
size_rgb = multispectral.shape
BLOCK_SIZE = 32
n_bands = size_rgb[2]
m_host = multispectral.astype(np.float32)
r_host = m_host[:,:,0].astype(np.float32)
g_host = m_host[:,:,1].astype(np.float32)
b_host = m_host[:,:,2].astype(np.float32)
panchromatic_host = panchromatic.astype(np.float32)
r_gpu = gpuarray.to_gpu(r_host)
g_gpu = gpuarray.to_gpu(g_host)
b_gpu = gpuarray.to_gpu(b_host)
p_gpu = gpuarray.to_gpu(panchromatic_host)
mean_r_gpu = misc.mean(r_gpu)
mean_g_gpu = misc.mean(g_gpu)
mean_b_gpu = misc.mean(b_gpu)
r_gpu_subs = gpuarray.zeros_like(r_gpu,np.float32)
g_gpu_subs = gpuarray.zeros_like(g_gpu,np.float32)
b_gpu_subs = gpuarray.zeros_like(b_gpu,np.float32)
substract( r_gpu, mean_r_gpu.get(), r_gpu_subs)
substract( g_gpu, mean_g_gpu.get(), g_gpu_subs)
substract( b_gpu, mean_b_gpu.get(), b_gpu_subs)
r_subs_split = split(r_gpu_subs.get(),BLOCK_SIZE,BLOCK_SIZE)
g_subs_split = split(g_gpu_subs.get(),BLOCK_SIZE,BLOCK_SIZE)
b_subs_split = split(b_gpu_subs.get(),BLOCK_SIZE,BLOCK_SIZE)
mat_var_cov = varianza_cov(r_subs_split,g_subs_split,b_subs_split)
coefficient = 1.0/((size_rgb[0]*size_rgb[1])-1)
ortogonal_matrix = mat_var_cov*coefficient
polynomial_trace = successive_powers(ortogonal_matrix)
characteristic_polynomial = polynomial_coefficients(polynomial_trace, ortogonal_matrix)
characteristic_polynomial_roots = np.roots(np.insert(characteristic_polynomial,0,1))
eigenvalues_mat = np.diag(characteristic_polynomial_roots)
eigenvectors_mat = -1*ortogonal_matrix[1:n_bands,0]
mat_ortogonal_base, q_matrix = eigenvectors_norm(eigenvalues_mat, ortogonal_matrix, eigenvectors_mat)
q_matrix_list = q_matrix.tolist()
q_matrix_cpu = np.array(q_matrix_list).astype(np.float32)
w1 = q_matrix_cpu[0,:]
w2 = (-1)*q_matrix_cpu[1,:]
w3 = q_matrix_cpu[2,:]
eigenvectors = np.array((w1,w2,w3))
inv_eigenvectors = la.inv(eigenvectors)
inv_list = inv_eigenvectors.tolist()
inv_eigenvector_cpu = np.array(inv_list).astype(np.float32)
r_subs_split_cp = split(r_host,BLOCK_SIZE,BLOCK_SIZE)
g_subs_split_cp = split(g_host,BLOCK_SIZE,BLOCK_SIZE)
b_subs_split_cp = split(b_host,BLOCK_SIZE,BLOCK_SIZE)
pc_1,pc_2,pc_3 = componentes_principales_original(r_subs_split_cp,g_subs_split_cp,b_subs_split_cp,q_matrix_cpu,r_host.shape[0], BLOCK_SIZE)
p_subs_split_nb = split(panchromatic_host,BLOCK_SIZE,BLOCK_SIZE)
pc_2_subs_split_nb = split(pc_2,BLOCK_SIZE,BLOCK_SIZE)
pc_3_subs_split_nb = split(pc_3,BLOCK_SIZE,BLOCK_SIZE)
nb1,nb2,nb3 = componentes_principales_panchromartic(p_subs_split_nb,pc_2_subs_split_nb,pc_3_subs_split_nb,inv_eigenvector_cpu,r_host.shape[0], BLOCK_SIZE)
nb11 = nb1.astype(np.float32)
nb22 = nb2.astype(np.float32)
nb33 = nb3.astype(np.float32)
nb11_gpu = gpuarray.to_gpu(nb11)
nb22_gpu = gpuarray.to_gpu(nb22)
nb33_gpu = gpuarray.to_gpu(nb33)
nb111_gpu = gpuarray.empty_like(nb11_gpu)
nb222_gpu = gpuarray.empty_like(nb22_gpu)
nb333_gpu = gpuarray.empty_like(nb33_gpu)
negative_adjustment(nb11_gpu,nb111_gpu)
negative_adjustment(nb22_gpu,nb222_gpu)
negative_adjustment(nb33_gpu,nb333_gpu)
nb111_cpu = nb111_gpu.get().astype(np.uint8)
nb222_cpu = nb222_gpu.get().astype(np.uint8)
nb333_cpu = nb333_gpu.get().astype(np.uint8)
fusioned_image=np.stack((nb111_cpu,nb222_cpu,nb333_cpu),axis=2)
skimage.io.imsave('pcagpu_image.tif',fusioned_image, plugin='tifffile')
