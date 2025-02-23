import os

import gpytorch
import kornia as K
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from BO_functions import covariate_finder, score
from dataset_def import RotatedMNISTDataset, RotatedMNISTDataset_partial
from GP_def import ExactGPModel
from kernel_gen import generate_kernel_batched
from parse_model_args import ModelArgs
from training import hensman_BO_restart_training, VAE_BO_restart_training_updated
from VAE import ConvVAE_mod1, ConvVAE_partial

eps = 1e-6

if __name__ == "__main__":
    """
    Root file for running GP-VAE-BO.
    
    Run command: python VAE_GP_BO.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)
    original_image_pad = pd.read_csv(os.path.join(data_source_path, 'base_image.csv'), header=None).to_numpy()
    if iter_num:
        save_path = os.path.join(save_path, str(iter_num))
        results_path = os.path.join(results_path, str(iter_num))
        data_source_path = os.path.join(data_source_path, str(iter_num))
        os.mkdir(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    dataset = RotatedMNISTDataset_partial(csv_file_data, csv_file_label, mask_file, csv_file_label_mask, root_dir=data_source_path,
                                  transform=transforms.ToTensor())
    validation_dataset = RotatedMNISTDataset(csv_file_validation_data, csv_file_validation_label, validation_mask_file, root_dir=data_source_path,
                                             transform=transforms.ToTensor())

    print('Length of dataset:  {}'.format(len(dataset)))
    N = len(dataset)
    Q = 2
    target_np = np.genfromtxt(os.path.join(data_source_path, 'target_label.csv'), delimiter=',')
    target_label = torch.tensor(target_np).reshape(1, 3).to(device)
    print(target_label)
    # read in the files
    original_image_pad = np.reshape(original_image_pad, (52, 52))
    original_image_pad = original_image_pad[..., np.newaxis]
    original_image_pad_tensor = torch.tensor(original_image_pad, dtype=torch.float).reshape(1, 1, original_image_pad.shape[0],
                                                                                            original_image_pad.shape[1])
    original_image_pad_tensor = original_image_pad_tensor/255
    image_goal_tensor = K.geometry.transform.rotate(original_image_pad_tensor, torch.Tensor([target_np[2]]))
    image_goal_tensor = K.geometry.transform.translate(image_goal_tensor, torch.Tensor([[target_np[0], target_np[1]]]))
    image_goal_tensor = image_goal_tensor.to(device)

    new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(image_goal_tensor, original_image_pad_tensor)
    target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
    target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)

    Q = len(dataset[0]['label'])

    print('Using convolutional neural network')
    if GP_VAE == True:
        nnet_model = ConvVAE_partial(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed, X_dim=4).double().to(device)
    else:
        nnet_model = ConvVAE_mod1(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed).double().to(device)
    nnet_model = nnet_model.double().to(device)

    # set up Data Loader for GP initialisation
    setup_dataloader = DataLoader(dataset, batch_size=N, shuffle=True, num_workers=4)

    # Get values for GP initialisation:
    Z = torch.zeros(N, latent_dim, dtype=torch.double).to(device)
    train_x = torch.zeros(N, Q, dtype=torch.double).to(device)
    label_mask = torch.zeros(N, Q, dtype=torch.double).to(device)
    nnet_model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(setup_dataloader):
            # no mini-batching. Instead get a batch of dataset size
            label_id = sample_batched['idx']
            train_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)
            label_mask = sample_batched['label_mask'].double().to(device)

            if GP_VAE == True:
                mu, log_var,  mu_X, log_var_X = nnet_model.encode(data, train_x)
            else:
                mu, log_var = nnet_model.encode(data)
            Z[label_id] = nnet_model.sample_latent(mu, log_var)

    label_max = train_x.max(dim=0).values.to(device)
    label_min = train_x.min(dim=0).values.to(device)
    covar_module = []
    covar_module0 = []
    zt_list = []
    likelihoods = []
    gp_models = []
    adam_param_list = []

    likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
                                                          noise_constraint=gpytorch.constraints.GreaterThan(
                                                              1.000E-08)).to(device)

    if constrain_scales:
        likelihoods.noise = 1
        likelihoods.raw_noise.requires_grad = False

    covar_module0 = generate_kernel_batched(latent_dim, cat_kernel, bin_kernel, sqexp_kernel,
                                            cat_int_kernel, bin_int_kernel,
                                            covariate_missing_val)
    print(covar_module0)
    print(covar_module0.kernels)
    print(covar_module0.kernels[0].outputscale)
    gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                            covar_module0).to(device)

    covariates = train_x[label_mask.numpy()[:, -1]]
    zt_list = torch.zeros(latent_dim, M, train_x.shape[1], dtype=torch.double).to(device)
    for i in range(latent_dim):
        zt_list[i] = covariates[np.random.choice(covariates.shape[0], M, replace=False)].clone().detach()

    zt_list.requires_grad_(True)
    print(zt_list.shape)

    adam_param_list.append({'params': covar_module0.parameters()})
    adam_param_list.append({'params': zt_list})

    covar_module0.train().double()
    likelihoods.train().double()

    m = torch.randn(latent_dim, M, 1).double().to(device).detach()
    H = (torch.randn(latent_dim, M, M) / 10).double().to(device).detach()

    if natural_gradient:
        H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)

    if not natural_gradient:
        adam_param_list.append({'params': m})
        adam_param_list.append({'params': H})
        m.requires_grad_(True)
        H.requires_grad_(True)

    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)
    nnet_model.train()

    if memory_dbg:
        print("Max memory allocated during initialisation: {:.2f} MBs".format(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        torch.cuda.reset_max_memory_allocated(device)

    burn_in = 9
    
    if GP_VAE == True:
        print('Using GP prior')
        _ = hensman_BO_restart_training(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, num_dim, covar_module0, likelihoods,
                                        m, H, zt_list, weight, loss_function, image_goal_tensor, target_label,
                                        original_image_pad_tensor, save_path,
                                        csv_file_data, csv_file_label, mask_file, csv_file_label_mask, validation_dataset, vy_init, vy_fixed, constrain_scales,
                                        cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val,
                                        M, N, data_source_path, natural_gradient, natural_gradient_lr)

        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, \
        H, zt_list, cost_arr, Z_arr, validation_loss_arr = _[0], _[1], _[2], _[3], _[4], _[5], _[6], _[7], _[8], _[9], _[10]
    else:
        print('Using standard Gaussian prior')

        _ = VAE_BO_restart_training_updated(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim,
                                            weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path,
                                            csv_file_data, csv_file_label, mask_file, csv_file_label_mask, validation_dataset, num_dim, data_source_path, vy_init, vy_fixed)
        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, cost_arr,\
        Z_arr, validation_loss_arr = _[0], _[1], _[2], _[3], _[4], _[5], _[6], _[7]

    print('Saving')
    pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, cost_arr, Z_arr, validation_loss_arr],
                 os.path.join(save_path, 'diagnostics.pkl'))
