import util as U
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    ## Get image filenames
    img_fnames = U.pao_dir(args.dir)
    
    gamma = 1.0; # Gamma used for inverse tone mapping

    ## Compute image statistics
    kappa, img_avg, _ = U.pao_compute_kappa_imgs(img_fnames, gamma)

    ## 1st Estimate: f assumed 0
    
    # Compute albedo and alpha image
    f, alpha_1est = U.pao_alpha_initial_estimate( kappa )
    amboc_1est = U.pao_alpha2amboc(alpha_1est)

    ctransposed_f = U.ctranspose(f)
    # Albedo
    albedo_1est = U.pao_compute_albedo(img_avg, alpha_1est, ctransposed_f)

    ## 2nd Estimate: Non linear optimization corrects for f
    #  This stage only makes sense if there is an ambient light source in the
    #  scene.
        
    f, alpha_2est = U.pao_alpha_nonlinear_opt(kappa, f, alpha_1est, img_avg)
    ctransposed_f = U.ctranspose(f)
    # Ambient Occlusion
    amboc_2est = U.pao_alpha2amboc(alpha_2est)
        
    # Albedo
    albedo_2est = U.pao_compute_albedo(img_avg, alpha_2est, ctransposed_f)

    ## Estimate L for an image
            
    img = U.pao_imread(img_fnames[3])
        
    lum_1est = img / np.maximum(U.realmin('double'), albedo_1est)
    lum_2est = img /  np.maximum(U.realmin('double'), albedo_2est)

    amboc_1est = cv2.normalize(src=amboc_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    albedo_1est = cv2.normalize(src=albedo_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lum_1est = cv2.normalize(src=lum_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    amboc_2est = cv2.normalize(src=amboc_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    albedo_2est = cv2.normalize(src=albedo_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lum_2est = cv2.normalize(src=lum_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    amboc_1est = cv2.cvtColor(amboc_1est, cv2.COLOR_BGR2RGB)
    albedo_1est = cv2.cvtColor(albedo_1est,cv2.COLOR_BGR2RGB)
    lum_1est = cv2.cvtColor(lum_1est,cv2.COLOR_BGR2RGB)

    amboc_2est = cv2.cvtColor(amboc_2est,cv2.COLOR_BGR2RGB)
    albedo_2est = cv2.cvtColor(albedo_2est,cv2.COLOR_BGR2RGB)
    lum_2est = cv2.cvtColor(lum_2est,cv2.COLOR_BGR2RGB)

    ## Show results
    plt.subplot(242)
    plt.imshow(amboc_1est)
    plt.title('Ambient Occlusion 1est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(243)
    plt.imshow(albedo_1est)
    plt.title('Albedo 1est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(244)
    plt.imshow(lum_1est)
    plt.title('Illumination 1est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(246)
    plt.imshow(amboc_2est)
    plt.title('Ambient Occlusion 2est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(247)
    plt.imshow(albedo_2est)
    plt.title('Albedo 2est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(248)
    plt.imshow(lum_2est)
    plt.title('Illumination 2est')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(141)
    plt.imshow(img)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()
