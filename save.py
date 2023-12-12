import util as U
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    for sub_dir in os.listdir(args.input_dir):

        output_sub_dir = os.path.join(args.output_dir, sub_dir)
        if not os.path.exists(output_sub_dir): os.makedirs(output_sub_dir)

        ## Get image filenames
        img_fnames = U.pao_dir(os.path.join(args.input_dir, sub_dir))
        
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

        amboc_1st = cv2.normalize(src=amboc_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        albedo_1st = cv2.normalize(src=albedo_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        amboc_2nd = cv2.normalize(src=amboc_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        albedo_2nd = cv2.normalize(src=albedo_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imwrite(os.path.join(output_sub_dir, f'albedo_1st.jpg'), albedo_1st)
        cv2.imwrite(os.path.join(output_sub_dir, f'albedo_2nd.jpg'), albedo_2nd)
        cv2.imwrite(os.path.join(output_sub_dir, f'amboc_1st.jpg'), amboc_1st)
        cv2.imwrite(os.path.join(output_sub_dir, f'amboc_2nd.jpg'), amboc_2nd)

        ## Estimate L for an image
        for i in range(len(img_fnames)):            
            img = U.pao_imread(img_fnames[i])
                
            lum_1est = img / np.maximum(U.realmin('double'), albedo_1est)
            lum_2est = img /  np.maximum(U.realmin('double'), albedo_2est)

            lum_1est = cv2.normalize(src=lum_1est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            lum_2est = cv2.normalize(src=lum_2est, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            cv2.imwrite(os.path.join(output_sub_dir, f'light_{i}_1st.jpg'), lum_1est)
            cv2.imwrite(os.path.join(output_sub_dir, f'light_{i}_2nd.jpg'), lum_2est)

if __name__ == "__main__":
    main()
