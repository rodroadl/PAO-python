'''
https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html
'''
import os
import cv2
import numpy as np
from scipy import signal, optimize, sparse
from math import sqrt, pi

def pao_dir(dir_path):
    fnames = os.listdir(dir_path)
    for i in range(len(fnames)):
        fnames[i] = os.path.join(dir_path, fnames[i])
    return fnames

def pao_imread(fname, must_fail=False):
    if not must_fail: must_fail = 0
    else: raise Exception("ERROR: Must fail is true when reading file {}".format(fname))

    img = cv2.imread(fname)

    if img.shape[2] == 1: img = np.tile(img, (1,1,3))

    return img

def pao_compute_kappa_imgs(img_fnames, gamma=None, shift=None):
    if not gamma: gamma = 1
    img = pao_imread(img_fnames[0])
    img_avg = np.zeros(img.shape, np.float64)
    img2_avg = np.zeros(img.shape, np.float64)

    for i in range(len(img_fnames)):
        img = pao_imread(img_fnames[i]) ** gamma

        if shift: img = np.maximum(0, img-shift)

        # NOTE: sus
        img_avg += img
        img2_avg += img**2

    img_avg = img_avg / len(img_fnames)
    img2_avg = img2_avg / len(img_fnames)

    img_kappa = img_avg**2 / np.maximum(realmin('double'), img2_avg)

    return img_kappa, img_avg, img2_avg

def pao_alpha_initial_estimate(kappa, mask=None):
    KMAG_THRESH = .005 #original value: .005
    # Estimate values for f
    f = np.zeros((kappa.shape[2], 1)) # 3x1
    kappa_max = np.zeros((kappa.shape[2], 1)) #3x1
    
    for i in range(max(f.shape)):
        # We should not consider the kappa value around depth discontinuities,
        # since our model does not explain these regions
        kdx, kdy = pao_gradient(kappa[:,:,i], 3)
        kmag = np.sqrt(kdx**2 + kdy**2)
        sel = kmag < KMAG_THRESH
        kappa_aux = kappa[:,:,i].copy()
        kappa_aux = np.reshape(kappa_aux[sel], (-1,1), order='F')
        if 0 in kappa_aux.shape: kappa_max[i] = 0.999
        else: kappa_max[i] = np.minimum(0.999, np.max(kappa_aux, axis=0))
        f[i] = (sqrt(3.)*np.sqrt(kappa_max[i] * (1-kappa_max[i])) + 3*kappa_max[i]-3)/ (6*pi*(1-kappa_max[i]))

    kappa_max[:] = 0.75
    kappa *= kappa_max[0]
    f = (sqrt(3.)*np.sqrt(kappa_max * (1-kappa_max)) + 3*kappa_max - 3) / (6*pi*(1-kappa_max))

    # Compute mapping kappa -> alpha
    n_samples = 1000
    kappa2alpha = np.zeros((n_samples, max(f.shape))) # 1000x3

    for i in range(max(f.shape)):
        a = np.array([np.linspace(0, pi/2, n_samples)]) # a.shape: 1x1000
        r_of_a = pao_compute_kappa(f[i], a) # r_of_a.shape: 1x1000x1
        r_of_a = np.squeeze(r_of_a,) # r_of_a.shape: 1000
        r_of_a[0] = 0
        r = np.linspace(0, kappa_max[i], n_samples) #r.shape: 1000x1
        r = np.squeeze(r)
        a = np.squeeze(a)
        for j in range(max(r.shape)):
            idx = np.nonzero(r_of_a <= r[j])[0][-1]
            kappa2alpha[j, i] = a[idx]
            
    
    # Now compute the alpha's
    alpha = np.zeros((kappa.shape[0], kappa.shape[1], max(f.shape))) # 
    for i in range(max(f.shape)):
        idxs = np.zeros(alpha[:,:,i].shape)
        idxs = np.round((n_samples-1) * kappa[:,:,i] / kappa_max[i]) + 1
        idxs = np.minimum(idxs, n_samples-1)
        tmp = kappa2alpha[:, i] # 1000x1
        idxs = np.reshape(idxs, (-1,1), order='F')
        idxs = np.squeeze(idxs)
        idxs = np.uint64(idxs)
        tmp1 = tmp[np.ix_(idxs)]
        tmp1 = np.reshape(tmp1, (kappa.shape[0],kappa.shape[1]), order='F')
        alpha[:,:,i] = tmp1 # 1000x1[500x365], 
    
    alpha = np.sum(alpha, axis=2)/3

    return f, alpha

def pao_compute_kappa(f, alpha):
    # equation (9) in the paper
    # Single bounce model for kappa

    kappa = np.zeros((alpha.shape[0], alpha.shape[1], max(f.shape)))

    for i in range(max(f.shape)):
        nom = 3 * (1 + 2*f[i]*pi)**2 * np.sin(alpha)**4
        den = 4 * (1 - np.cos(alpha)**3 + 3*f[i]*pi*(1 + f[i]*pi)*np.sin(alpha)**4) + 1e-4
        kappa[:,:,i] = nom / den

    return kappa

def pao_alpha2amboc(alpha):
    # Convert alpha image to ambient occlusion image
    amboc = np.sin(alpha)**2
    return amboc

def pao_compute_albedo(avg_img, alpha, f):
    albedo = np.zeros(avg_img.shape)
    if np.size(f) == avg_img.shape[2]:
        new_f = np.zeros(avg_img.shape)
        for i in range(np.size(f)):
            new_f[:,:,i] = f[0][i] if np.ndim(f) == 2 else f[i]
        f = new_f
    
    for i in range(f.shape[2]):
        albedo[:,:,i] = 2*avg_img[:,:,i] / ((1 + 2*f[:,:,i]*pi) * np.sin(alpha)**2)

    return albedo

def pao_alpha_nonlinear_opt(kappa, f0, alpha0, avg, mask=None):
    if not mask: mask = []
    n_pix = kappa.shape[0] * kappa.shape[1]
    n_vars = 3 + n_pix

    lb = np.zeros((n_vars))
    ub = np.zeros((n_vars))
    ub[:3] = 1
    ub[3:] = pi/2

    x0 = np.block([[np.reshape(f0, (1, -1)), np.reshape(alpha0, (1, -1))]])

    # Pixel weight
    sum_avg = np.sum(avg, axis=2)
    sum_avg_max = np.maximum(0.01, sum_avg)
    tmp1 = np.zeros(kappa.shape)
    for c in range(3):
        tmp1[:,:,c] = sum_avg_max.copy()

    kappa_w = avg / tmp1
    for i in range(kappa_w.shape[2]):
        aux = kappa_w[:,:,i]
        aux[mask == 0] = 0
        kappa_w[:,:,i] = aux
    
    kdx, kdy = pao_gradient(kappa, 3)
    kmag = np.sqrt(kdx**2 +kdy**2)
    kmag = np.max(kmag, axis=2)
    tmp2 = np.zeros(kappa.shape)
    for c in range(3):
        tmp2[:,:,c] = 1.1 - kmag / np.max(kmag[:])
    kappa_w = kappa_w * tmp2

    kappa_w[:] = 1
    if mask: kappa[mask == 0] = 0

    # Set options for non-linear optimization
    lsq_Method = 'trf' # trf: trust-region-reflective
    lsq_Display = 2 # 2: display progress during iterations
    lsq_JacobPattern = jacobian_sparsity_pattern(kappa.shape)
    # lsq_MaxIter = 10

    # Optimize objective function
    print('Running optimization')

    x0 = np.squeeze(x0)
    x_opt = optimize.least_squares(objective_function, 
                                   x0, 
                                   args=(kappa, kappa_w), 
                                   bounds=(lb,ub), 
                                   method=lsq_Method, 
                                   verbose=lsq_Display, 
                                   jac_sparsity=lsq_JacobPattern)
    f = x_opt.x[0:3]
    alpha = np.reshape(x_opt.x[3:], (kappa.shape[0], kappa.shape[1]), order='F').copy()

    return f, alpha

def objective_function(x, kappa, kappa_w):
    f = x[:3]
    alpha = np.reshape(x[3:], (kappa.shape[0], kappa.shape[1]))
    kappa_pred = pao_compute_kappa(f, alpha)
    diff_pred = (kappa_pred - kappa) * kappa_w
    diff_pred = diff_pred.flatten()
    return diff_pred

def jacobian_sparsity_pattern(imsize):
    n_f = imsize[2] # 3
    n_alpha = imsize[0] * imsize[1] # 182500

    f_ijs = np.zeros((n_alpha*n_f, 2))
    for i in range(n_f):
        idx = i*n_alpha + np.array(range(n_alpha))
        f_ijs[np.ix_(idx, [0])] = np.reshape(i*n_alpha + np.array(range(n_alpha)), (-1,1))
        f_ijs[np.ix_(idx, [1])] = i
    
    alpha_ijs = np.zeros((n_alpha*n_f, 2))
    for i in range(n_f):
        alpha_ijs[np.ix_(idx,[0])] = np.reshape(i*n_alpha + np.array(range(n_alpha)), (-1,1))
        alpha_ijs[np.ix_(idx,[1])] = np.reshape(n_f + np.array(range(n_alpha)), (-1,1))
    ijs = np.concatenate((f_ijs, alpha_ijs), axis=0) # 6x182500
    vals = np.ones((ijs.shape[0]))

    J = sparse.csr_matrix((vals, (ijs[:,0], ijs[:,1])), shape=(n_f*n_alpha, n_f+n_alpha))
    return J

def ctranspose(mat):
    '''
    computes the complex conjugate transpose of mat
    '''
    return np.conj(mat).T


def realmin(precision):
    '''
    https://www.mathworks.com/help/matlab/ref/realmin.html
    returns the smallest positive normalized floating-point number in IEEE double precision. This is equal to realmin for double precision
    '''
    return np.finfo(np.double).tiny # https://lists.archive.carbon60.com/python/python/767032

def pao_gradient(img, win_size):
    # GRADIENT Compute the gradient of the grayscale image img and return it in
    # img_dx and img_dy    
    KERN_B_3 = [0.223755, 0.552490, 0.223755]
    KERN_D_3 = [-0.453014, 0.0, 0.453014]

    KERN_B_5 = [0.036420, 0.248972, 0.42917, 0.248972, 0.036420]
    KERN_D_5 = [-0.108415, -0.280353, 0.0, 0.280353, 0.108415]

    if win_size == 3:
        kern_b = KERN_B_3
        kern_d = KERN_D_3
    elif win_size == 5:
        kern_b = KERN_B_5
        kern_d = KERN_D_5
    else:
        raise Exception('win_size must be either 3 or 5')
    
    imgdx = np.zeros(img.shape)
    imgdy = np.zeros(img.shape)

    if np.ndim(img) == 3:
        for c in range(img.shape[2]):
            # C = conv2(u,v,A) first convolves each column of A with the vector u, 
            # and then it convolves each row of the result with the vector v.
            imgdx[:,:,c] = signal.convolve2d(img[:,:,c], np.atleast_2d(kern_b).T, mode='same')
            imgdx[:,:,c] = signal.convolve2d(img[:,:,c], np.atleast_2d(kern_d), mode='same')
            imgdy[:,:,c] = signal.convolve2d(img[:,:,c], np.atleast_2d(kern_d).T, mode='same')
            imgdy[:,:,c] = signal.convolve2d(img[:,:,c], np.atleast_2d(kern_b), mode='same')
    else:
        imgdx[:,:] = signal.convolve2d(img[:,:], np.atleast_2d(kern_b).T, mode='same')
        imgdx[:,:] = signal.convolve2d(img[:,:], np.atleast_2d(kern_d), mode='same')
        imgdy[:,:] = signal.convolve2d(img[:,:], np.atleast_2d(kern_d).T, mode='same')
        imgdy[:,:] = signal.convolve2d(img[:,:], np.atleast_2d(kern_b), mode='same')
    return imgdx, imgdy