import gpytorch
import umap
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import os
from matplotlib import cm

from BO_functions import run_BO_step, covariate_finder, run_BO_step_simple, score
from VAE import ConvVAE_mod1, ConvVAE_partial
from dataset_def import RotatedMNISTDataset_BO, RotatedMNISTDataset_partial_BO
from elbo_functions import minibatch_sgd
import matplotlib.pyplot as plt
import time

from kernel_gen import generate_kernel_batched


def hensman_training_simple(nnet_model, epochs, dataset, optimiser, latent_dim, covar_module0,
                           likelihoods, m, H, zt_list, weight, loss_function,
                           natural_gradient=False, natural_gradient_lr=0.01):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)
    eps=1e-6

    batch_size = 30
    n_batches = (N + batch_size - 1) // (batch_size)

    # set up Data Loader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    if loss_function == 'mse':
        valid_best = np.Inf
    else:
        valid_best = np.Inf

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader):
            optimiser.zero_grad()

            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            print(data.shape)
            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
            kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, covariates, mu, log_var,
                                                     zt_list, N_batch, natural_gradient, eps, N)

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                        grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list


def hensman_BO_training(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, covar_module0, likelihoods, m, H, zt_list,
                        weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path, 
                        csv_file_data, csv_file_label, mask_file, validation_dataset, natural_gradient=False,
                        natural_gradient_lr=0.01, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = './data'
    MC_SAMPLES = 2048
    seed = 1
    batch_size = 25
#    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5

    label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    Z_arr = np.empty((0, latent_dim))
    nll_loss_best = np.Inf
    epoch_flag = 0
    best_cost = np.Inf

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=True, num_workers=4)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z =torch.tensor([]).double().to(device)
        covariates_list = torch.tensor([]).double().to(device)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)

            recon_batch, mu, log_var = nnet_model(data)
            z_samples = nnet_model.sample_latent(mu, log_var)
            train_z = torch.cat((train_z, mu.clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates.clone().detach()), dim=0)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
#            print(recon_batch.max())
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

#            net_loss = torch.tensor([0.0]).to(device)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
            kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, covariates, mu, log_var,
                                                     zt_list, N_batch, natural_gradient, eps, N)

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + weight * kld_loss
 #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

#            if (epoch > burn_in and epoch % 10 == 0) or (epoch <= burn_in):
            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                        grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 10 == 0:
            nnet_model.eval()
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_cost, 90, 10, -10))
            plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')


            dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(train_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_train' + str(epoch) +'.pdf'), bbox_inches='tight')
                plt.close('all')

            data_base_np= pd.read_csv('./data/dataset_3000_restrict/base_image.csv', header=None).to_numpy() # 100 x 2704
            mask_base_np = np.ones_like(data_np)
            label_base_np = np.array([[1, 0, 0, 0, 0]])
            dataset_BO = RotatedMNISTDataset_BO(data_base_np, mask_base_np, label_base_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')

                ax.imshow(np.reshape(recon_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')


                ax.imshow(np.reshape(train_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base_train' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')

            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
                        if epoch <= burn_in:
                            print("Saving the best models...")
                            torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_val.pth'))
                            torch.save(covar_module0.state_dict(), os.path.join(save_path, 'covar_module0_best_val.pth'))
                            torch.save(likelihoods.state_dict(), os.path.join(save_path, 'likelihoods_best_val.pth'))
                            torch.save(zt_list, os.path.join(save_path, 'zt_list_best_val.pth'))
                            torch.save(m, os.path.join(save_path, 'm_best_val.pth'))
                            torch.save(H, os.path.join(save_path, 'H_best_val.pth'))
                            torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer_best_val.pth'))
                            print('Best validation model saved')

        if epoch == (burn_in + 1):
            print("Loading best model...")
            nnet_model.load_state_dict(torch.load(os.path.join(save_path, 'vae_model_best_val.pth')))
            covar_module0.load_state_dict(torch.load(os.path.join(save_path, 'covar_module0_best_val.pth')))
            likelihoods.load_state_dict(torch.load(os.path.join(save_path, 'likelihoods_best_val.pth')))
            optimiser.load_state_dict(torch.load(os.path.join(save_path, 'optimizer_best_val.pth')))
            zt_list = torch.load(os.path.join(save_path, 'zt_list_best_val.pth'))
            m = torch.load(os.path.join(save_path, 'm_best_val.pth'))
            H = torch.load(os.path.join(save_path, 'H_best_val.pth'))
            print("Loaded best model...")

            nnet_model.eval()
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
#            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_cost, 90, 10, -10))
            plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(train_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_train' + str(epoch) +'.pdf'), bbox_inches='tight')
                plt.close('all')

            data_base_np= pd.read_csv('./data/dataset_3000_restrict/base_image.csv', header=None).to_numpy() # 100 x 2704
            mask_base_np = np.ones_like(data_np)
            label_base_np = np.array([[1, 0, 0, 0, 0]])
            dataset_BO = RotatedMNISTDataset_BO(data_base_np, mask_base_np, label_base_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')

                ax.imshow(np.reshape(recon_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')


                ax.imshow(np.reshape(train_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base_train' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([recon_arr],
                             os.path.join(save_path, 'diagnostics_plot.pkl'))
#                sys.exit()
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
                torch.save(covar_module0.state_dict(), os.path.join(save_path, 'covar_module0.pth'))
                torch.save(zt_list, os.path.join(save_path, 'zt_list.pth'))
                torch.save(m, os.path.join(save_path, 'm.pth'))
                torch.save(H, os.path.join(save_path, 'H.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
                print('Model saved')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()

                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[N_orig:, 0], embedding[N_orig:, 1], c=covariates_list_plot[N_orig:, 3].cpu().numpy(),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent.pkl'))
#            return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr
        nnet_model.train()
        if epoch > burn_in:
            print('Running BO...')
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            z_target = nnet_model.sample_latent(mu_target, log_var_target)
            new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 3].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                            state_dict, MC_SAMPLES, seed, -label_max[3], -label_min[3], epoch, save_path,
                                                            original_image_pad_tensor, target_label, z_target, N_orig)
            print('Candidate cost: %f'%(new_obj))
            print('New candidate obtained...')

            nnet_model.eval()
            # image goal tensor
            new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
            img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
            new_y = new_y.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            img_best_cost = -score(new_y * 255, image_goal_tensor * 255).unsqueeze(-1)
#            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
            print('Candidate replacement cost (mod): %f'%(img_best_cost))
            print('New covariates obtained...')
            cost_arr = np.append(cost_arr, img_best_cost.cpu().item())
            Z_arr = np.append(Z_arr, new_z.reshape((1, -1)).cpu(), axis=0)

            if img_best_cost < best_cost:
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_cost.pth'))
                best_cost = img_best_cost

            if epoch == (burn_in + 1):
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Optimised image.')
                plt.savefig(os.path.join(save_path, 'optimised_first_sample.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(img_best.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Corresponding image.')
                plt.savefig(os.path.join(save_path, 'corresponding_first_sample.pdf'), bbox_inches='tight')

            data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
            data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
            train_x_test = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
            train_x_test = np.append(np.array([1]), train_x_test)
            label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
            mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)

            if epoch == epochs:
                with torch.no_grad():
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                    ax.set_title('Optimised image.')
                    plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                    plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3]))
                    plt.savefig(os.path.join(save_path, 'optimised_best_corresponding.pdf'), bbox_inches='tight')

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 1], label='Rotation')
                    ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 2], label='Shift_x')
                    ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 3], label='Shift_y')
                    ax.plot(range(1, len(cost_arr)+1), cost_arr, 'r-', label='Cost')
                    ax.set_xlabel('BO steps')
                    ax.legend()
                    plt.savefig(os.path.join(save_path, 'label-trace.pdf'), bbox_inches='tight')

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1], c=covariates_list_plot[N_orig:-1, 3].cpu().numpy(),
                                               marker='v', linewidths=0)
                                s.set_clim([cmin, cmax])
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO.pdf'), bbox_inches='tight')
                    plt.close('all')

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s2 = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1],
                                                c=range(1, len(embedding[N_orig:-1, 0]) + 1),
                                                marker='v', linewidths=0, cmap=cm.coolwarm)
                                cb2 = fig.colorbar(s2)
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    cb2.set_label('BO Steps')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO_steps.pdf'), bbox_inches='tight')
                    plt.close('all')

                    pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                                 os.path.join(save_path, 'diagnostics_latent_BO.pkl'))

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 2):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_check.pdf'), bbox_inches='tight')
                    plt.close('all')

                    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_end.pth'))

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, Z_arr, validation_loss_arr

def kl_divergence_simple(z, mu, std, covariates_mask):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
#    kl = kl * (1 - covariates_mask)
    print(kl.shape)
    kl = kl.sum()
    return kl

def kl_divergence_pytorch(z, mu, std, covariates_mask):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    kl_loss = nn.KLDivLoss(reduction="batchmean")
    output = kl_loss(q, p)
    print(output.shape)
    return output

def init_model(latent_dim, num_dim, vy_init, vy_fixed, constrain_scales, cat_kernel, bin_kernel, sqexp_kernel,
               cat_int_kernel, bin_int_kernel, covariate_missing_val, train_x, M, N, natural_gradient):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet_model = ConvVAE_partial(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed, X_dim=2).double().to(device)
    covar_module = []
    covar_module0 = []
    zt_list = []
    likelihoods = []
    adam_param_list = []
    N = train_x.shape[0]
    
    likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
                                                          noise_constraint=gpytorch.constraints.GreaterThan(
                                                              1.000E-08)).to(device)
    if constrain_scales:
        likelihoods.noise = 1
        likelihoods.raw_noise.requires_grad = False

    covar_module0 = generate_kernel_batched(latent_dim, cat_kernel, bin_kernel, sqexp_kernel,
                                            cat_int_kernel, bin_int_kernel,
                                            covariate_missing_val)

    # initialise inducing points
    print(train_x.shape)
    zt_list = torch.zeros(latent_dim, M, train_x.shape[1], dtype=torch.double).to(device)
    for i in range(latent_dim):
        zt_list[i] = train_x[np.random.choice(train_x.shape[0], M, replace=False)].clone().detach()
        # zt_list[i]=torch.cat((train_x[20:60], train_x[10000:10040]), dim=0).clone().detach()
        # zt_list[i]=torch.cat((train_x[0:40], train_x[2000:2040]), dim=0).clone().detach()
    zt_list.requires_grad_(True)

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

    return nnet_model, optimiser, covar_module0, likelihoods, m, H, zt_list


def hensman_BO_restart_training(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, num_dim, covar_module0, likelihoods,
                                m, H, zt_list, weight, loss_function, image_goal_tensor, target_label,
                                original_image_pad_tensor, save_path,
                                csv_file_data, csv_file_label, mask_file, csv_file_label_mask, validation_dataset, vy_init, vy_fixed, constrain_scales,
                                cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val,
                                M, N, data_source_path, natural_gradient=False, natural_gradient_lr=0.01, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = data_source_path

    MC_SAMPLES = 2048
    seed = 1
    batch_size = 100
    #    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5
    label_mask_np = pd.read_csv(os.path.join(root_dir, csv_file_label_mask), header=None).to_numpy() # 100 x 5
    label_mask_np = label_mask_np[:, [1, 2, 3, 4]]  # 100 x 2

    covariates_np = label_np[:, [1 ,2, 3, 4]]
    train_label_means_np = np.sum(covariates_np, axis=0) / np.sum(label_mask_np, axis=0)
    covariates_mask_bool = np.greater(label_mask_np, 0)
    train_label_std_np = np.zeros(label_mask_np.shape[1])
    for col in range(0, label_mask_np.shape[1]):
        train_label_std_np[col] = np.std(covariates_np[covariates_mask_bool[:, col], col])

 #   label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
 #   label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    label_np = label_np[:, [1, 2, 3, 4]]

#    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
#    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    Z_arr = np.empty((0, latent_dim))
    nll_loss_best = np.Inf
    epoch_flag = 0
    best_cost = np.Inf
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    epoch = 1
    BO_count = 0
    start_time = time.perf_counter()
    while epoch <= epochs:
        start_epoch_time = time.perf_counter()
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        nll_X_loss_sum = 0
        kl_X_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_partial_BO(data_np, mask_np, label_np, label_mask_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z = torch.tensor([]).double().to(device)
        covariates_list = torch.tensor([]).double().to(device)
        train_x_stack = torch.tensor([]).double().to(device)
        nnet_model.train()
        covar_module0.train()
        likelihoods.train()

        train_label_means = torch.from_numpy(train_label_means_np)
        train_label_std = torch.from_numpy(train_label_std_np)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()

            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            label_mask = sample_batched['label_mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_norm = (covariates - train_label_means) / train_label_std
            noise_replace = torch.zeros_like(covariates_norm)


            covariates_norm = (covariates_norm * label_mask) + (noise_replace * (1 - label_mask))
#            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)
            train_x_stack = torch.cat((train_x_stack, train_x.clone().detach()), dim=0)
            recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, label_mask)
            X_tilde_norm = X_tilde * (1 - label_mask) + covariates_norm * label_mask
            X_tilde_denorm = (X_tilde * train_label_std) + train_label_means
            X_hat = X_tilde_denorm * (1 - label_mask) + covariates * label_mask

#            loss = nn.MSELoss(reduction='none')
#            se = torch.mul(loss(X_tilde_norm.view(-1, covariates.shape[1]), covariates_norm.view(-1, covariates.shape[1])), label_mask.view(-1, covariates.shape[1]))
#            mask_sum = torch.sum(label_mask.view(-1, covariates.shape[1]), dim=1)
#            mask_sum[mask_sum == 0] = 1
#            mse_X = torch.sum(torch.sum(se, dim=1) / mask_sum)

            [mse_X, nll_X] = nnet_model.cov_loss_function(X_tilde_norm, covariates_norm, label_mask)
            mse_X = torch.sum(mse_X)
            nll_X = torch.sum(nll_X)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            # sample X_hat from q_X
            std_X = torch.exp(log_var_X / 2)
#            q_X = torch.distributions.Normal(mu_X, std_X)
#            X_hat_sample = q_X.rsample()
#            print(std_X)
#            kl_X = kl_divergence_simple(X_tilde_norm, mu_X, std_X, label_mask)
            kl_X = -0.5 * torch.sum(1 + std_X - mu_X.pow(2) - std_X.exp())

            cost_mask = label_mask[:, -1].numpy()
            idx_cost_mask = np.argwhere(cost_mask).reshape(-1)
            train_z = torch.cat((train_z, mu[idx_cost_mask].clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates[idx_cost_mask].clone().detach()), dim=0)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
            kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, X_hat, mu, log_var,
                                                     zt_list, N_batch, natural_gradient, eps, N)

            mse_X = mse_X * N / N_batch
            nll_X = nll_X * N / N_batch
            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + nll_X + kld_loss + kl_X
            #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + mse_X + weight * (kld_loss + kl_X)

            #            if (epoch > burn_in and epoch % 10 == 0) or (epoch <= burn_in):
            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                        grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches
            nll_X_loss_sum += nll_X.item() / n_batches
            kl_X_loss_sum += kl_X.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - KL X loss: %.3f - NLL_X loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, kl_X_loss_sum, nll_X_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

#        print(covariates_list.shape)
#        print(covariates_list)
#        print(train_z.shape)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 10 == 0:
            nnet_model.eval()
#            target_label_mask = torch.ones_like(target_label)
#            target_label_norm = (target_label - train_label_means)/train_label_std
#            recon_batch_target, mu_target, log_var_target, mu_X_target, log_var_X_target, X_tilde_target = nnet_model(image_goal_tensor.double().to(device), target_label_norm, target_label_mask)
#            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            #            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
#            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
#            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)

                    covariates = train_x[:, 0:4]
                    label_mask = torch.ones_like(covariates).double().to(device)
                    covariates_norm = (covariates - train_label_means) / train_label_std
                    noise_replace = torch.zeros_like(covariates_norm)

                    covariates_norm = (covariates_norm * label_mask) + (noise_replace * (1 - label_mask))

                    recon_batch, mu, log_var, mu_X, log_var_X, X_tilde = nnet_model(data, covariates_norm, label_mask)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
                        if epoch <= burn_in:
                            print("Saving the best models...")
                            torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_val.pth'))
                            torch.save(covar_module0.state_dict(), os.path.join(save_path, 'covar_module0_best_val.pth'))
                            torch.save(likelihoods.state_dict(), os.path.join(save_path, 'likelihoods_best_val.pth'))
                            torch.save(zt_list, os.path.join(save_path, 'zt_list_best_val.pth'))
                            torch.save(m, os.path.join(save_path, 'm_best_val.pth'))
                            torch.save(H, os.path.join(save_path, 'H_best_val.pth'))
                            torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer_best_val.pth'))
                            print('Best validation model saved')

        if epoch == (burn_in + 1):
            print(covar_module0.kernels[0].outputscale)
            print(covar_module0.kernels[1].outputscale)
            print(covar_module0.kernels[2].outputscale)
            print(covar_module0.kernels[3].outputscale)
            print("Loading best model...")
            nnet_model.load_state_dict(torch.load(os.path.join(save_path, 'vae_model_best_val.pth')))
            covar_module0.load_state_dict(torch.load(os.path.join(save_path, 'covar_module0_best_val.pth')))
            likelihoods.load_state_dict(torch.load(os.path.join(save_path, 'likelihoods_best_val.pth')))
            optimiser.load_state_dict(torch.load(os.path.join(save_path, 'optimizer_best_val.pth')))
            zt_list = torch.load(os.path.join(save_path, 'zt_list_best_val.pth'))
            m = torch.load(os.path.join(save_path, 'm_best_val.pth'))
            H = torch.load(os.path.join(save_path, 'H_best_val.pth'))
            print("Loaded best model...")
        nnet_model.train()
        if epoch > burn_in:
            print('Running BO...')

            new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 1].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                            state_dict)
            print('Candidate cost: %f'%(new_obj))
            print('New candidate obtained...')
            BO_count = BO_count + 1
            nnet_model.eval()

            # image goal tensor
            new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
            img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
            new_y = new_y.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            img_best_cost = -score(new_y * 255, image_goal_tensor * 255).unsqueeze(-1)
            #            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
            print('Candidate replacement cost (mod): %f'%(img_best_cost))
            print('New covariates obtained...')
            cost_arr = np.append(cost_arr, img_best_cost.cpu().item())
            Z_arr = np.append(Z_arr, new_z.reshape((1, -1)).cpu(), axis=0)

            if img_best_cost < best_cost:
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_cost.pth'))
                best_cost = img_best_cost

            data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
            data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
            train_x_test = np.array([new_x_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
#            train_x_test = np.append(np.array([1]), train_x_test)
            label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
            mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)
            label_mask_np = np.append(label_mask_np, np.ones((1, 2)), axis=0)

 #           train_x_val = torch.tensor(np.array([new_x_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)).to(device)
#            train_x_val = train_x_val.reshape(1, -1)
#            train_x_stack = torch.cat((train_x_stack, train_x_val.clone().detach()), dim=0)

            if BO_count == 200:
                with torch.no_grad():
                    # fig, ax = plt.subplots(1, 1)
                    # ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                    # ax.set_title('Optimised image.')
                    # plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                    plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                print('Total run time: %0.4f' %(time.perf_counter() - start_time))
                return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, Z_arr, validation_loss_arr
#            if BO_count % 10 == 0:
#                epoch = 0
#                nnet_model, optimiser, covar_module0, likelihoods, m, H, zt_list = init_model(latent_dim, num_dim, vy_init, vy_fixed, constrain_scales, cat_kernel, bin_kernel, sqexp_kernel,
#                                                                                              cat_int_kernel, bin_int_kernel, covariate_missing_val, covariates_list, M, N, natural_gradient)
        epoch = epoch + 1
        print('Epoch run time: %0.4f' %(time.perf_counter() - start_epoch_time))
        print('BO count: %d' %(BO_count))


    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, Z_arr, validation_loss_arr


def VAE_BO_training_updated(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, likelihoods,
                            weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path,
                            csv_file_data, csv_file_label, mask_file, validation_dataset, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = './data'
    #    csv_file_data = 'dataset_300_new_target/train_data_masked.csv'
    #    mask_file = 'dataset_300_new_target/train_mask.csv'
    #    csv_file_label = 'dataset_300_new_target/train_labels.csv'
    MC_SAMPLES = 2048
    seed = 1
    batch_size = 25
    #    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5

    label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    Z_arr = np.empty((0, latent_dim))
    nll_loss_best = np.Inf
    epoch_flag = 0
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z =torch.tensor([]).double().to(device)
        covariates_list = torch.tensor([]).double().to(device)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)

            recon_batch, mu, log_var = nnet_model(data)
            z_samples = nnet_model.sample_latent(mu, log_var)
            train_z = torch.cat((train_z, mu.clone().detach().double()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates.clone().detach().double()), dim=0)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            #            print(recon_batch.max())
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            #            net_loss = torch.tensor([0.0]).to(device)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + weight * kld_loss
            #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

#            if (epoch > burn_in and epoch % 10 == 0) or (epoch <= burn_in):
            net_loss.backward()
            optimiser.step()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 10 == 0:
            if epoch % 100 == 0:
                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                #            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
                new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
                target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
                new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
                target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
                target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
                ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                             % (target_img_cost, 90, 10, -10))
                plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
                ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                             % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
                plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

                data_sample_test = torch.Tensor(data_np[2, :]).reshape(1, 52, 52).double()
                recon_batch_target, mu_target, log_var_target = nnet_model(data_sample_test.double().to(device))
                #            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
                new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(data_sample_test.cpu().numpy(), [52, 52]), cmap='gray')
                ax.set_title('Training image')
                plt.savefig(os.path.join(save_path, 'training_true_' + str(epoch) + '.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
                ax.set_title('Training image - recon')
                plt.savefig(os.path.join(save_path, 'training_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
                        if epoch <= burn_in:
                            print("Saving the best models...")
                            torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_val.pth'))
                            torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer_best_val.pth'))
                            print('Best validation model saved')
        #                  else:
        #                      epoch_flag = epoch_flag + 1
        #               if epoch_flag == 4:
        #                   print('No change in validation. Early stopping at epoch: %d'%(epoch))
        #                   epoch = burn_in

        if epoch == (burn_in + 1):
            print("Loading best model...")
            nnet_model.load_state_dict(torch.load(os.path.join(save_path, 'vae_model_best_val.pth')))
            optimiser.load_state_dict(torch.load(os.path.join(save_path, 'optimizer_best_val.pth')))
            print("Loaded best model...")

            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            #                new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_cost, 90, 10, -10))
            plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            recon_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions.pdf'), bbox_inches='tight')
                plt.close('all')
                pd.to_pickle([recon_arr],
                             os.path.join(save_path, 'diagnostics_plot.pkl'))
                #                sys.exit()
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
                print('Model saved')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target.double()), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()

                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[N_orig:, 0], embedding[N_orig:, 1], c=covariates_list_plot[N_orig:, 3].cpu().numpy(),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent.pkl'))
        #           return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr

        if epoch > burn_in:
            print('Running BO...')
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 3].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                            state_dict, MC_SAMPLES, seed, -label_max[3], -label_min[3], epoch, save_path,
                                                            original_image_pad_tensor, target_label, mu_target, N_orig)
            print('Candidate cost: %f' %(new_obj))
            print('New candidate obtained...')
            nnet_model.eval()
            # image goal tensor
            new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
            img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
            img_best_cost = -score(new_y * 255, image_goal_tensor * 255).unsqueeze(-1)
            #            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
            print('Candidate replacement cost: %f'%(img_best_cost))
            print('New covariates obtained...')
            cost_arr = np.append(cost_arr, img_best_cost.cpu().item())
            Z_arr = np.append(Z_arr, new_z.reshape((1, -1)).cpu(), axis=0)

            if epoch == (burn_in + 1):
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Optimised image.')
                plt.savefig(os.path.join(save_path, 'optimised_first_sample.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(img_best.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Corresponding image.')
                plt.savefig(os.path.join(save_path, 'corresponding_first_sample.pdf'), bbox_inches='tight')

            data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
            data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
            train_x_test = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
            train_x_test = np.append(np.array([1]), train_x_test)
            label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
            mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)

            # if epoch % 2 == 0:
            #     recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            #     new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #     target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            #     new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            #     target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #     target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            #     fig, ax = plt.subplots(1, 1)
            #     ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            #     ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
            #                  % (target_img_cost, 90, 10, -10))
            #     plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            #     fig, ax = plt.subplots(1, 1)
            #     ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            #     ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
            #                  % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            #     plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            if epoch == (epochs - 1):
                with torch.no_grad():
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                    ax.set_title('Optimised image.')
                    plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                    plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3]))
                    plt.savefig(os.path.join(save_path, 'optimised_best_corresponding.pdf'), bbox_inches='tight')

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target.double()), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1], c=covariates_list_plot[N_orig:-1, 3].cpu().numpy(),
                                               marker='v', linewidths=0)
                                s.set_clim([cmin, cmax])
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO.pdf'), bbox_inches='tight')
                    plt.close('all')

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s2 = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1],
                                                c=range(1, len(embedding[N_orig:-1, 0]) + 1),
                                                marker='v', linewidths=0, cmap=cm.coolwarm)
                                cb2 = fig.colorbar(s2)
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    cb2.set_label('BO Steps')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO_steps.pdf'), bbox_inches='tight')
                    plt.close('all')

                    pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                                 os.path.join(save_path, 'diagnostics_latent_BO.pkl'))

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target.double()), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 2):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_check.pdf'), bbox_inches='tight')
                    plt.close('all')

                    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_end.pth'))

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, cost_arr, Z_arr, validation_loss_arr


def init_VAE_model(latent_dim, num_dim, vy_init, vy_fixed):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet_model = ConvVAE_mod1(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed).double().to(device)
    adam_param_list = []
    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)

    return nnet_model, optimiser


def VAE_BO_restart_training_updated(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim,
                            weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path,
                            csv_file_data, csv_file_label, mask_file, csv_file_label_mask, validation_dataset, num_dim, data_source_path, vy_init, vy_fixed, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = data_source_path

    MC_SAMPLES = 2048
    seed = 1
    batch_size = 100
    #    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5
    label_mask_np = pd.read_csv(os.path.join(root_dir, csv_file_label_mask), header=None).to_numpy() # 100 x 5
    label_mask_np = label_mask_np[:, [2, 4]]  # 100 x 2

    covariates_np = label_np[:, [2, 4]]
    train_label_means_np = np.sum(covariates_np, axis=0) / np.sum(label_mask_np, axis=0)
    covariates_mask_bool = np.greater(label_mask_np, 0)
    train_label_std_np = np.zeros(label_mask_np.shape[1])
    for col in range(0, label_mask_np.shape[1]):
        train_label_std_np[col] = np.std(covariates_np[covariates_mask_bool[:, col], col])

    #   label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    #   label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    label_np = label_np[:, [2, 4]]

    #    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    #    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    Z_arr = np.empty((0, latent_dim))
    nll_loss_best = np.Inf
    epoch_flag = 0
    best_cost = np.Inf
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    epoch = 1
    BO_count = 0
    start_time = time.perf_counter()

    while epoch <= epochs:
        start_epoch_time = time.perf_counter()
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        nll_X_loss_sum = 0
        kl_X_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_partial_BO(data_np, mask_np, label_np, label_mask_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z = torch.tensor([]).double().to(device)
        covariates_list = torch.tensor([]).double().to(device)
        train_x_stack = torch.tensor([]).double().to(device)
        nnet_model.train()

        train_label_means = torch.from_numpy(train_label_means_np)
        train_label_std = torch.from_numpy(train_label_std_np)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()

            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            label_mask = sample_batched['label_mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:2]
            #            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)
            train_x_stack = torch.cat((train_x_stack, train_x.clone().detach()), dim=0)

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            cost_mask = label_mask[:, -1].numpy()
            idx_cost_mask = np.argwhere(cost_mask).reshape(-1)
            train_z = torch.cat((train_z, mu[idx_cost_mask].clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates[idx_cost_mask].clone().detach()), dim=0)

            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * (kld_loss)

            #            if (epoch > burn_in and epoch % 10 == 0) or (epoch <= burn_in):
            net_loss.backward()
            optimiser.step()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 2 == 0:
            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
                        if epoch <= burn_in:
                            print("Saving the best models...")
                            torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model_best_val.pth'))
                            torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer_best_val.pth'))
                            print('Best validation model saved')
        #                  else:
        #                      epoch_flag = epoch_flag + 1
        #               if epoch_flag == 4:
        #                   print('No change in validation. Early stopping at epoch: %d'%(epoch))
        #                   epoch = burn_in

        if epoch == (burn_in + 1):
            print("Loading best model...")
            nnet_model.load_state_dict(torch.load(os.path.join(save_path, 'vae_model_best_val.pth')))
            optimiser.load_state_dict(torch.load(os.path.join(save_path, 'optimizer_best_val.pth')))
            print("Loaded best model...")

            # with torch.no_grad():
            #     #                sys.exit()
            #     torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
            #     torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
            #     print('Model saved')
        #           return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr

        if epoch > burn_in:
            print('Running BO...')
            print(train_z.shape)
            print(covariates_list.shape)
            new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 1].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                            state_dict)
            print('Candidate cost: %f' %(new_obj))
            print('New candidate obtained...')
            BO_count = BO_count + 1
            nnet_model.eval()
            # image goal tensor
            new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
            img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
            new_y = new_y.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            img_best_cost = -score(new_y * 255, image_goal_tensor * 255).unsqueeze(-1)
            #            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
            print('Candidate replacement cost: %f'%(img_best_cost))
            print('New covariates obtained...')
            cost_arr = np.append(cost_arr, img_best_cost.cpu().item())
            Z_arr = np.append(Z_arr, new_z.reshape((1, -1)).cpu(), axis=0)

            data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
            data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
            train_x_test = np.array([new_x_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
            label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
            mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)
            label_mask_np = np.append(label_mask_np, np.ones((1, 2)), axis=0)

            if BO_count == 200:
                with torch.no_grad():
                    # fig, ax = plt.subplots(1, 1)
                    # ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                    # ax.set_title('Optimised image.')
                    # plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                    plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                print('Total run time: %0.4f' %(time.perf_counter() - start_time))
                return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, cost_arr, Z_arr, validation_loss_arr
            if BO_count % 10 == 0:
                epoch = 0
                nnet_model, optimiser = init_VAE_model(latent_dim, num_dim, vy_init, vy_fixed)
        epoch = epoch + 1
        print('Epoch run time: %0.4f' %(time.perf_counter() - start_epoch_time))
        print('BO count: %d' %(BO_count))
    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, cost_arr, Z_arr, validation_loss_arr


def hensman_BO_training_fixed(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, covar_module0, likelihoods, m, H, zt_list,
                              weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path, 
                              csv_file_data, csv_file_label, mask_file, validation_dataset, natural_gradient=False,
                              natural_gradient_lr=0.01, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = './data'
#    csv_file_data = 'dataset_3000/train_data_masked.csv'
#    mask_file = 'dataset_3000/train_mask.csv'
#    csv_file_label = 'dataset_3000/train_labels.csv'
    MC_SAMPLES = 2048
    seed = 1
    batch_size = 25
#    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5

    label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    nll_loss_best = np.Inf
    epoch_flag = 0
#    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_z =torch.tensor([]).to(device)
    covariates_list = torch.tensor([]).to(device)

    for epoch in range(1, burn_in + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=False, num_workers=4)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z =torch.tensor([]).double().to(device)
        covariates_list = torch.tensor([]).double().to(device)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)

            recon_batch, mu, log_var = nnet_model(data)
            z_samples = nnet_model.sample_latent(mu, log_var)
            train_z = torch.cat((train_z, mu.clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates.clone().detach()), dim=0)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
#            print(recon_batch.max())
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

#            net_loss = torch.tensor([0.0]).to(device)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
            kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, covariates, mu, log_var,
                                                     zt_list, N_batch, natural_gradient, eps, N)

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + weight * kld_loss
 #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                        grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 25 == 0:
            nnet_model.eval()
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
#            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_cost, 90, 10, -10))
            plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

#             data_sample_test = torch.Tensor(data_np[2, :]).reshape(1, 52, 52).double()
#             recon_batch_target, mu_target, log_var_target = nnet_model(data_sample_test.double().to(device))
# #            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#             new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()

#             fig, ax = plt.subplots(1, 1)
#             ax.imshow(np.reshape(data_sample_test.cpu().numpy(), [52, 52]), cmap='gray')
#             ax.set_title('Training image')
#             plt.savefig(os.path.join(save_path, 'training_true_' + str(epoch) + '.pdf'), bbox_inches='tight')

#             fig, ax = plt.subplots(1, 1)
#             ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
#             ax.set_title('Training image - recon')
#             plt.savefig(os.path.join(save_path, 'training_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')


            dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(train_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_train' + str(epoch) +'.pdf'), bbox_inches='tight')
                plt.close('all')

            data_base_np= pd.read_csv('./data/dataset_3000_restrict/base_image.csv', header=None).to_numpy() # 100 x 2704
            mask_base_np = np.ones_like(data_np)
            label_base_np = np.array([[1, 0, 0, 0, 0]])
            dataset_BO = RotatedMNISTDataset_BO(data_base_np, mask_base_np, label_base_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')

                ax.imshow(np.reshape(recon_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')


                ax.imshow(np.reshape(train_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base_train' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')

            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
  #                  else:
  #                      epoch_flag = epoch_flag + 1
 #               if epoch_flag == 4:
 #                   print('No change in validation. Early stopping at epoch: %d'%(epoch))
 #                   epoch = burn_in

        if epoch == burn_in:
            nnet_model.eval()
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
#            new_target = nnet_model.decode(mu_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_target = recon_batch_target.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
            new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
            target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_cost, 90, 10, -10))
            plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
            ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                         % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
            plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')
            
#            recon_arr = np.empty((0, 2704))
            # with torch.no_grad():
            #     for batch_idx, sample_batched in enumerate(dataloader_BO):
            #         data = sample_batched['digit'].double().to(device)
            #         recon_batch, mu, log_var = nnet_model(data)
            #         recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

            #     fig, ax = plt.subplots(10, 10)
            #     for ax_ in ax:
            #         for ax__ in ax_:
            #             ax__.set_xticks([])
            #             ax__.set_yticks([])
            #     plt.axis('off')

            #     for i in range(0, 10):
            #         for j in range(0, 10):
            #             idx = i * 10 + j
            #             ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
            #     plt.savefig(os.path.join(save_path, 'reconstructions.pdf'), bbox_inches='tight')
            #     plt.close('all')



            dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(train_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_train' + str(epoch) +'.pdf'), bbox_inches='tight')
                plt.close('all')

            data_base_np= pd.read_csv('./data/dataset_3000_restrict/base_image.csv', header=None).to_numpy() # 100 x 2704
            mask_base_np = np.ones_like(data_np)
            label_base_np = np.array([[1, 0, 0, 0, 0]])
            dataset_BO = RotatedMNISTDataset_BO(data_base_np, mask_base_np, label_base_np, transform=transforms.ToTensor())
            dataloader_BO = DataLoader(dataset_BO, batch_size=25, shuffle=False, num_workers=4)
            recon_arr = np.empty((0, 2704))
            train_arr = np.empty((0, 2704))

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)
                    train_arr = np.append(train_arr, data.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')

                ax.imshow(np.reshape(recon_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')
                
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')


                ax.imshow(np.reshape(train_arr[0], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions_base_train' + str(epoch) + '.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([recon_arr],
                             os.path.join(save_path, 'diagnostics_plot.pkl'))
#                sys.exit()
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
                torch.save(covar_module0.state_dict(), os.path.join(save_path, 'covar_module0.pth'))
                torch.save(zt_list, os.path.join(save_path, 'zt_list.pth'))
                torch.save(m, os.path.join(save_path, 'm.pth'))
                torch.save(H, os.path.join(save_path, 'H.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
                print('Model saved')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()

                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[N_orig:, 0], embedding[N_orig:, 1], c=covariates_list_plot[N_orig:, 3].cpu().numpy(),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent.pkl'))
#            return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr
        nnet_model.train()

    for epoch in range(burn_in + 1, epochs + 1):
        print('Running BO...')
        recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
        z_target = nnet_model.sample_latent(mu_target, log_var_target)
        new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 3].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                        state_dict, MC_SAMPLES, seed, -label_max[3], -label_min[3], epoch, save_path,
                                                        original_image_pad_tensor, target_label, z_target, N_orig)
        print('Candidate cost: %f'%(new_obj))
        print('New candidate obtained...')

        nnet_model.eval()
        # image goal tensor
        new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
        img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
#            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
        new_y = new_y.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        img_best_cost = -score(new_y * 255, image_goal_tensor * 255).unsqueeze(-1)
#            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
        print('Candidate replacement cost (mod): %f'%(img_best_cost))
        print('New covariates obtained...')
        cost_arr = np.append(cost_arr, img_best_cost.cpu().item())

        if epoch == (burn_in + 1):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
            ax.set_title('Optimised image.')
            plt.savefig(os.path.join(save_path, 'optimised_first_sample.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(img_best.cpu(), [52, 52]), cmap='gray')
            ax.set_title('Corresponding image.')
            plt.savefig(os.path.join(save_path, 'corresponding_first_sample.pdf'), bbox_inches='tight')

        data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
        data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
        train_x_test = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
        train_x_test = np.append(np.array([1]), train_x_test)
        label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
        mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)

        # if epoch % 25 == 0:
        #     recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
        #     new_target = nnet_model.decode(recon_batch_target).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        #     target_img_cost = -score(new_target * 255, image_goal_tensor * 255).unsqueeze(-1)
        #     new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value, target_img_best = covariate_finder(new_target, original_image_pad_tensor)
        #     target_img_best = target_img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        #     target_img_best_cost = -score(target_img_best * 255, image_goal_tensor * 255).unsqueeze(-1)

        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(np.reshape(new_target.cpu().numpy(), [52, 52]), cmap='gray')
        #     ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
        #                  % (target_img_cost, 120, 10, 10))
        #     plt.savefig(os.path.join(save_path, 'target_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(np.reshape(target_img_best.cpu().numpy(), [52, 52]), cmap='gray')
        #     ax.set_title('Target reconstuction. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
        #                  % (target_img_best_cost, new_target_rotation_value, new_target_x_shift_value, new_target_y_shift_value))
        #     plt.savefig(os.path.join(save_path, 'target_replaced_image_recon_' + str(epoch) + '.pdf'), bbox_inches='tight')

        if epoch == epochs:
            with torch.no_grad():
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Optimised image.')
                plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                lowest_cost_idx = np.argmin(cost_arr)
                best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                             % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                lowest_cost_idx = np.argmin(cost_arr)
                best_opt_img = data_np[N_orig + lowest_cost_idx, :]
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                             % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3]))
                plt.savefig(os.path.join(save_path, 'optimised_best_corresponding.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 1], label='Rotation')
                ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 2], label='Shift_x')
                ax.plot(range(1, len(cost_arr)+1), label_np[N_orig:, 3], label='Shift_y')
                ax.plot(range(1, len(cost_arr)+1), cost_arr, 'r-', label='Cost')
                ax.set_xlabel('BO steps')
                ax.legend()
                plt.savefig(os.path.join(save_path, 'label-trace.pdf'), bbox_inches='tight')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)

                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 3):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        if i == 1:
                            s = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1], c=covariates_list_plot[N_orig:-1, 3].cpu().numpy(),
                                           marker='v', linewidths=0)
                            s.set_clim([cmin, cmax])
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space_BO.pdf'), bbox_inches='tight')
                plt.close('all')

                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 3):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        if i == 1:
                            s2 = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1],
                                            c=range(1, len(embedding[N_orig:-1, 0]) + 1),
                                            marker='v', linewidths=0, cmap=cm.coolwarm)
                            cb2 = fig.colorbar(s2)
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                cb2.set_label('BO Steps')
                plt.savefig(os.path.join(save_path, 'latent_space_BO_steps.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent_BO.pkl'))

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 10, -10, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space_check.pdf'), bbox_inches='tight')
                plt.close('all')

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr



def hensman_BO_training_no_update(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim, covar_module0, likelihoods, m, H, zt_list,
                                  weight, loss_function, image_goal_tensor, original_image_pad_tensor, save_path, validation_dataset, natural_gradient=False,
                                  natural_gradient_lr=0.01, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)
    eps = 1e-6
    root_dir = './data'
    csv_file_data = 'dataset_100/train_data_masked.csv'
    mask_file = 'dataset_100/train_mask.csv'
    csv_file_label = 'dataset_100/train_labels.csv'
    MC_SAMPLES = 2048
    seed = 1
    batch_size = 25
    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5

    label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    nll_loss_best = np.Inf
    epoch_flag = 0
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_z =torch.tensor([]).to(device)
    covariates_list = torch.tensor([]).to(device)
    for epoch in range(1, burn_in + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=False, num_workers=4)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z =torch.tensor([]).to(device)
        covariates_list = torch.tensor([]).to(device)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)

            recon_batch, mu, log_var = nnet_model(data)
            z_samples = nnet_model.sample_latent(mu, log_var)
            train_z = torch.cat((train_z, mu.clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates.clone().detach()), dim=0)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            #            print(recon_batch.max())
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            #            net_loss = torch.tensor([0.0]).to(device)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))
            kld_loss, grad_m, grad_H = minibatch_sgd(covar_module0, likelihoods, latent_dim, m, PSD_H, covariates, mu, log_var,
                                                     zt_list, N_batch, natural_gradient, eps, N)

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr * (grad_H + grad_H.transpose(-1, -2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr * (
                        grad_m - 2 * torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 2 == 0:
            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
        #                  else:
        #                      epoch_flag = epoch_flag + 1
        #               if epoch_flag == 4:
        #                   print('No change in validation. Early stopping at epoch: %d'%(epoch))
        #                   epoch = burn_in

        if epoch == burn_in:
            recon_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions.pdf'), bbox_inches='tight')
                plt.close('all')
                pd.to_pickle([recon_arr],
                             os.path.join(save_path, 'diagnostics_plot.pkl'))
                #                sys.exit()
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
                torch.save(covar_module0.state_dict(), os.path.join(save_path, 'covar_module0.pth'))
                torch.save(zt_list, os.path.join(save_path, 'zt_list.pth'))
                torch.save(m, os.path.join(save_path, 'm.pth'))
                torch.save(H, os.path.join(save_path, 'H.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
                print('Model saved')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 4, 6, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()

                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:100, 0], embedding[0:100, 1], c=covariates_list_plot[0:100, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[100:, 0], embedding[100:, 1], c=covariates_list_plot[100:, 3].cpu().numpy(),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent.pkl'))

    for epoch in range(burn_in + 1, epochs + 1):
        nnet_model.eval()
        print('Running BO in epoch: %d'%(epoch))
        new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 3].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                        state_dict, MC_SAMPLES, seed, -label_max[3], -label_min[3], epoch, save_path)
        print('Candidate cost: %f'%(new_obj))
        print('New candidate obtained...')

        # image goal tensor
        new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
        img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
        img_best_cost = -score(img_best * 255, image_goal_tensor * 255).unsqueeze(-1)
        print('Candidate replacement cost: %f'%(img_best_cost))
        print('New covariates obtained...')
        cost_arr = np.append(cost_arr, img_best_cost.cpu().item())
        train_z = torch.cat((train_z, new_z), dim=0)
        new_labels_np = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()[0]])
        label_tensor = torch.tensor(new_labels_np).to(device)
        covariates_list = torch.cat((covariates_list, label_tensor.reshape(1, 4).to(device)), dim=0)

        if epoch == (burn_in + 1):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
            ax.set_title('Optimised image.')
            plt.savefig(os.path.join(save_path, 'optimised_first_sample.pdf'), bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.reshape(img_best.cpu(), [52, 52]), cmap='gray')
            ax.set_title('Corresponding image.')
            plt.savefig(os.path.join(save_path, 'corresponding_first_sample.pdf'), bbox_inches='tight')

        data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
        data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
        train_x_test = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()[0]])
        train_x_test = np.append(np.array([1]), train_x_test)
        label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
        mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)


        if epoch == (epochs - 1):
            with torch.no_grad():
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Optimised image.')
                plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                lowest_cost_idx = np.argmin(cost_arr)
                best_opt_img = data_optim_np[100 + lowest_cost_idx, :]
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                             % (cost_arr[lowest_cost_idx], label_np[100 + lowest_cost_idx, 1],
                                label_np[100 + lowest_cost_idx, 2], label_np[100 + lowest_cost_idx, 3]))
                plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                lowest_cost_idx = np.argmin(cost_arr)
                best_opt_img = data_np[100 + lowest_cost_idx, :]
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                             % (cost_arr[lowest_cost_idx], label_np[100 + lowest_cost_idx, 1],
                                label_np[100 + lowest_cost_idx, 2], label_np[100 + lowest_cost_idx, 3]))
                plt.savefig(os.path.join(save_path, 'optimised_best_corresponding.pdf'), bbox_inches='tight')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 4, 6, 0]).reshape(1, 4).double().to(device)), dim=0)

                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 3):
                    if cb is None:
                        s = ax.scatter(embedding[0:100, 0], embedding[0:100, 1], c=covariates_list_plot[0:100, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        if i == 1:
                            s = ax.scatter(embedding[100:-1, 0], embedding[100:-1, 1], c=covariates_list_plot[100:-1, 3].cpu().numpy(),
                                           marker='v', linewidths=0)
                            s.set_clim([cmin, cmax])
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space_BO.pdf'), bbox_inches='tight')
                plt.close('all')

                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 3):
                    if cb is None:
                        s = ax.scatter(embedding[0:100, 0], embedding[0:100, 1], c=covariates_list_plot[0:100, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        if i == 1:
                            s2 = ax.scatter(embedding[100:-1, 0], embedding[100:-1, 1],
                                            c=range(1, len(embedding[100:-1, 0]) + 1),
                                            marker='v', linewidths=0, cmap=cm.coolwarm)
                            cb2 = fig.colorbar(s2)
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                cb2.set_label('BO Steps')
                plt.savefig(os.path.join(save_path, 'latent_space_BO_steps.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent_BO.pkl'))

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([90, 4, 6, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()
                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:100, 0], embedding[0:100, 1], c=covariates_list_plot[0:100, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                       c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space_check.pdf'), bbox_inches='tight')
                plt.close('all')

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, zt_list, cost_arr, validation_loss_arr


def VAE_BO_training(nnet_model, epochs, burn_in, dataset, optimiser, latent_dim,
                    weight, loss_function, image_goal_tensor, target_label, original_image_pad_tensor, save_path,
                    validation_dataset, state_dict=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_orig = len(dataset)
    eps = 1e-6
    root_dir = './data'
    csv_file_data = 'dataset_3000/train_data_masked.csv'
    mask_file = 'dataset_3000/train_mask.csv'
    csv_file_label = 'dataset_3000/train_labels.csv'
    MC_SAMPLES = 2048
    seed = 1
    batch_size = 25
    #    n_batches = (N + batch_size - 1) // (batch_size)

    data_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    data_optim_np = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None).to_numpy() # 100 x 2704
    mask_np = pd.read_csv(os.path.join(root_dir, mask_file), header=None).to_numpy() # 100 x 2704
    label_np = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0).to_numpy() # 100 x 5

    label_max = torch.tensor(label_np[:, 1:].max(axis=0)).to(device)
    label_min = torch.tensor(label_np[:, 1:].min(axis=0)).to(device)

    cost_mean = torch.tensor(label_np[:, 4].mean()).to(device)
    cost_std = torch.tensor(label_np[:, 4].std()).to(device)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    validation_loss_arr = np.empty((0, 1))
    cost_arr = np.empty((0, 1))
    nll_loss_best = np.Inf
    epoch_flag = 0
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0

        dataset_BO = RotatedMNISTDataset_BO(data_np, mask_np, label_np, transform=transforms.ToTensor())
        dataloader_BO = DataLoader(dataset_BO, batch_size=batch_size, shuffle=False, num_workers=4)
        N = len(dataset_BO)
        print('Size of dataset: %d'%(N))
        n_batches = (N + batch_size - 1) // (batch_size)
        data_test = []
        train_x_test = []
        mask_test = []
        train_z =torch.tensor([]).to(device)
        covariates_list = torch.tensor([]).to(device)

        for batch_idx, sample_batched in enumerate(dataloader_BO):
            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]
            covariates = train_x[:, 0:4]
            covariates_gp_prior = (train_x[:, 0:4] - label_min)/(label_max - label_min)

            recon_batch, mu, log_var = nnet_model(data)
            z_samples = nnet_model.sample_latent(mu, log_var)
            train_z = torch.cat((train_z, mu.clone().detach()), dim=0)
            covariates_list = torch.cat((covariates_list, covariates.clone().detach()), dim=0)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            #            print(recon_batch.max())
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)
            #            net_loss = torch.tensor([0.0]).to(device)
            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            #               net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)

        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr, net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        # validation
        if epoch % 50 == 0:
            print('Performing validation')
            # set up Data Loader for training
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(validation_dataloader):
                    data = sample_batched['digit'].double().to(device)
                    train_x = sample_batched['label'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                    recon_validation_loss = torch.sum(recon_loss)
                    nll_validation_loss = torch.sum(nll)
                    validation_loss_arr = np.append(validation_loss_arr, nll_validation_loss.cpu().item())
                    if nll_validation_loss < nll_loss_best:
                        epoch_flag = 0
                        nll_loss_best = nll_validation_loss
                        print("Validation loss: %f"%(nll_validation_loss))
        #                  else:
        #                      epoch_flag = epoch_flag + 1
        #               if epoch_flag == 4:
        #                   print('No change in validation. Early stopping at epoch: %d'%(epoch))
        #                   epoch = burn_in

        if epoch == burn_in:
            recon_arr = np.empty((0, 2704))
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(dataloader_BO):
                    data = sample_batched['digit'].double().to(device)
                    recon_batch, mu, log_var = nnet_model(data)
                    recon_arr = np.append(recon_arr, recon_batch.reshape(-1, 2704).detach().cpu().numpy(), axis=0)

                fig, ax = plt.subplots(10, 10)
                for ax_ in ax:
                    for ax__ in ax_:
                        ax__.set_xticks([])
                        ax__.set_yticks([])
                plt.axis('off')

                for i in range(0, 10):
                    for j in range(0, 10):
                        idx = i * 10 + j
                        ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
                plt.savefig(os.path.join(save_path, 'reconstructions.pdf'), bbox_inches='tight')
                plt.close('all')
                pd.to_pickle([recon_arr],
                             os.path.join(save_path, 'diagnostics_plot.pkl'))
                #                sys.exit()
                torch.save(nnet_model.state_dict(), os.path.join(save_path, 'vae_model.pth'))
                torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimizer.pth'))
                print('Model saved')

                recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                train_z_plot = torch.cat((train_z, mu_target), dim=0)
                covariates_list_plot = torch.cat((covariates_list, torch.tensor([120, 10, 10, 0]).reshape(1, 4).double().to(device)), dim=0)
                fig, ax = plt.subplots(1, 1)
                cmin = covariates_list_plot[:, 3].min()
                cmax = covariates_list_plot[:, 3].max()
                cb = None
                embedding = np.array([])
                if latent_dim > 2:
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                elif latent_dim == 2:
                    embedding = train_z_plot.cpu().numpy()

                for i in range(0, 2):
                    if cb is None:
                        s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                       marker='o', linewidths=0)
                        s.set_clim([cmin, cmax])
                        cb = fig.colorbar(s)
                    else:
                        s = ax.scatter(embedding[N_orig:, 0], embedding[N_orig:, 1], c=covariates_list_plot[N_orig:, 3].cpu().numpy(),
                                       marker='X', linewidths=0)
                        s.set_clim([cmin, cmax])

                cb.set_label('Cost')
                plt.savefig(os.path.join(save_path, 'latent_space.pdf'), bbox_inches='tight')
                plt.close('all')

                pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                             os.path.join(save_path, 'diagnostics_latent.pkl'))

        if epoch > burn_in:
            print('Running BO...')
            recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
            new_z, new_obj, state_dict = run_BO_step_simple(train_z, -covariates_list[:, 3].reshape(-1, 1), nnet_model, image_goal_tensor,
                                                            state_dict, MC_SAMPLES, seed, -label_max[3], -label_min[3], epoch, save_path,
                                                            original_image_pad_tensor, target_label, mu_target, N_orig)
            print('Candidate cost: %f'%(new_obj))
            print('New candidate obtained...')

            nnet_model.eval()
            # image goal tensor
            new_y = nnet_model.decode(new_z).reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
            img_best = img_best.reshape(1, 1, original_image_pad_tensor.shape[2], original_image_pad_tensor.shape[3]).clone().detach()
            #            image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
            img_best_cost = -score(img_best * 255, image_goal_tensor * 255).unsqueeze(-1)
            #            img_best_cost = ((target_label - image_label)**2).sum().sqrt()
            print('Candidate replacement cost: %f'%(img_best_cost))
            print('New covariates obtained...')
            cost_arr = np.append(cost_arr, img_best_cost.cpu().item())

            if epoch == (burn_in + 1):
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Optimised image.')
                plt.savefig(os.path.join(save_path, 'optimised_first_sample.pdf'), bbox_inches='tight')

                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.reshape(img_best.cpu(), [52, 52]), cmap='gray')
                ax.set_title('Corresponding image.')
                plt.savefig(os.path.join(save_path, 'corresponding_first_sample.pdf'), bbox_inches='tight')

            data_np = np.append(data_np, (img_best*255).reshape((1, -1)).cpu(), axis=0)
            data_optim_np = np.append(data_optim_np, (new_y*255).reshape((1, -1)).cpu(), axis=0)
            train_x_test = np.array([new_rotation_value, new_x_shift_value, new_y_shift_value, img_best_cost.detach().cpu().numpy()], dtype=float)
            train_x_test = np.append(np.array([1]), train_x_test)
            label_np = np.append(label_np, train_x_test.reshape((1, -1)), axis=0)
            mask_np = np.append(mask_np, np.ones_like(new_y.reshape((1, -1)).cpu()), axis=0)

            if epoch == (epochs - 1):
                with torch.no_grad():
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(new_y.cpu(), [52, 52]), cmap='gray')
                    ax.set_title('Optimised image.')
                    plt.savefig(os.path.join(save_path, 'optimised.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_optim_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f. BO step: %d'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3], lowest_cost_idx+1))
                    plt.savefig(os.path.join(save_path, 'optimised_best.pdf'), bbox_inches='tight')

                    lowest_cost_idx = np.argmin(cost_arr)
                    best_opt_img = data_np[N_orig + lowest_cost_idx, :]
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(np.reshape(best_opt_img, [52, 52]), cmap='gray')
                    ax.set_title('Best optimised image. Cost: %f. Rotation: %f. Shift_x: %f. Shift_y: %f'
                                 % (cost_arr[lowest_cost_idx], label_np[N_orig + lowest_cost_idx, 1],
                                    label_np[N_orig + lowest_cost_idx, 2], label_np[N_orig + lowest_cost_idx, 3]))
                    plt.savefig(os.path.join(save_path, 'optimised_best_corresponding.pdf'), bbox_inches='tight')

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([120, 10, 10, 0]).reshape(1, 4).double().to(device)), dim=0)

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1], c=covariates_list_plot[N_orig:-1, 3].cpu().numpy(),
                                               marker='v', linewidths=0)
                                s.set_clim([cmin, cmax])
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO.pdf'), bbox_inches='tight')
                    plt.close('all')

                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 3):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            if i == 1:
                                s2 = ax.scatter(embedding[N_orig:-1, 0], embedding[N_orig:-1, 1],
                                                c=range(1, len(embedding[N_orig:-1, 0]) + 1),
                                                marker='v', linewidths=0, cmap=cm.coolwarm)
                                cb2 = fig.colorbar(s2)
                            else:
                                s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                               c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                               marker='X', linewidths=0)
                                s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    cb2.set_label('BO Steps')
                    plt.savefig(os.path.join(save_path, 'latent_space_BO_steps.pdf'), bbox_inches='tight')
                    plt.close('all')

                    pd.to_pickle([embedding, covariates_list_plot, train_z.cpu().numpy()],
                                 os.path.join(save_path, 'diagnostics_latent_BO.pkl'))

                    recon_batch_target, mu_target, log_var_target = nnet_model(image_goal_tensor.double().to(device))
                    train_z_plot = torch.cat((train_z, mu_target), dim=0)
                    covariates_list_plot = torch.cat((covariates_list, torch.tensor([120, 10, 10, 0]).reshape(1, 4).double().to(device)), dim=0)
                    fig, ax = plt.subplots(1, 1)
                    cmin = covariates_list_plot[:, 3].min()
                    cmax = covariates_list_plot[:, 3].max()
                    cb = None
                    embedding = np.array([])
                    if latent_dim > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(train_z_plot.cpu().numpy())
                    elif latent_dim == 2:
                        embedding = train_z_plot.cpu().numpy()
                    for i in range(0, 2):
                        if cb is None:
                            s = ax.scatter(embedding[0:N_orig, 0], embedding[0:N_orig, 1], c=covariates_list_plot[0:N_orig, 3].cpu().numpy(),
                                           marker='o', linewidths=0)
                            s.set_clim([cmin, cmax])
                            cb = fig.colorbar(s)
                        else:
                            s = ax.scatter(embedding[-1, 0].reshape((1,)), embedding[-1, 1].reshape((1,)),
                                           c=covariates_list_plot[-1, 3].cpu().numpy().reshape((1,)),
                                           marker='X', linewidths=0)
                            s.set_clim([cmin, cmax])

                    cb.set_label('Cost')
                    plt.savefig(os.path.join(save_path, 'latent_space_check.pdf'), bbox_inches='tight')
                    plt.close('all')

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, cost_arr, validation_loss_arr