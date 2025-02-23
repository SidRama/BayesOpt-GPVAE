from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import torch
import torch.nn.functional as F
import kornia as K
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module):
    def __init__(self, rotation, xval, yval, original_image_pad_tensor):
        super().__init__()

        self.original_image_pad_tensor = original_image_pad_tensor
        self.rotation = nn.Parameter(torch.Tensor([rotation]), requires_grad=True)
        self.translate = nn.Parameter(torch.Tensor([[xval, yval]]), requires_grad=True)

    def forward(self):
        """Implement function to be optimised.
        """
        img_tensor = K.geometry.transform.rotate(self.original_image_pad_tensor, self.rotation.cpu())
        img_tensor = K.geometry.transform.translate(img_tensor, self.translate.cpu())

        return img_tensor


def training_loop(model, optimizer, image_goal_tensor, n=1000):
    losses = []
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(n):
        preds = model().to(device)
        loss = F.mse_loss(preds.float(), image_goal_tensor.float()).sqrt()
#        print('====> Epoch: {} - Average loss: {:.4f}'.format(i, loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())

    return losses


def score(pred_image_tensor, image_goal_tensor):
    loss = -F.mse_loss(pred_image_tensor.float(), image_goal_tensor.float()).sqrt()
    return loss

def plot_GP_high_dim(model, train_z, train_obj, epoch, save_path):
    likelihood = model.likelihood
    latent_dim = train_z.shape[1]
    likelihood.eval()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_coord = train_z[-1, :].to(device)
    for i in range(0, latent_dim):
        min = train_z[:, i].min().detach().cpu().numpy()
        max = train_z[:, i].max().detach().cpu().numpy()
        x1 = torch.tensor(np.linspace(min - 1, max + 1, 300)).reshape(-1).to(device)
        coord_grid = target_coord.repeat(300, 1)
        coord_grid[:, i] = x1
        val = model(coord_grid)
        f_pred_mean = val.mean.detach().cpu().numpy()
        f_pred_var = val.variance.detach().cpu().numpy()
        lower, upper = val.confidence_region()

        fig, ax = plt.subplots(1, 1)
        ax.plot(x1.detach().cpu().numpy(), f_pred_mean, 'b')
        ax.fill_between(x1.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy(), alpha=0.5)
        ax.scatter(target_coord[i].detach().cpu().numpy(), train_obj[-1].detach().cpu().numpy(), marker='X', linewidths=0)

        ax.set_xlabel('Z - dim: %d' %(i))
        ax.set_ylabel('Cost')
        plt.savefig(os.path.join(save_path, 'GP_hline_latent_dim' + str(i) + '_' + str(epoch) + '.pdf'), bbox_inches='tight')


def plot_GP(model, train_z, train_obj, epoch, save_path, N_orig):
    likelihood = model.likelihood
    likelihood.eval()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = np.linspace(-0.1, 1.1, 300)
    y1 = np.linspace(-0.1, 1.1, 300)
    XX, YY = np.meshgrid(x1, y1)
    comb_array = torch.tensor(np.array(np.meshgrid(x1, y1)).T.reshape(-1, 2)).to(device)
    dataset = TensorDataset(comb_array)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    f_pred2 = np.array([0.0])
    f_pred2_var = np.array([0.0])
    for x_batch in loader:
        val = model(x_batch[0])
        f_pred2 = np.concatenate((f_pred2, val.mean.detach().cpu().numpy()), axis=0)
        f_pred2_var = np.concatenate((f_pred2_var, val.variance.detach().cpu().numpy()), axis=0)
    f_pred2 = f_pred2[1:]
    f_pred2_var = f_pred2_var[1:]

    Z = f_pred2.reshape(XX.shape).T
    Z_var = f_pred2_var.reshape(XX.shape).T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(train_z[0:N_orig, 0].cpu().detach().numpy(), train_z[0:N_orig, 1].cpu().detach().numpy(), train_obj[0:N_orig, 0].cpu().detach().numpy(), color='red', marker='X', alpha=0.4, linewidths=0)
    ax.scatter(train_z[N_orig:-2, 0].cpu().detach().numpy(), train_z[N_orig:-2, 1].cpu().detach().numpy(), train_obj[N_orig:-2, 0].cpu().detach().numpy(), color='red', marker='v', alpha=0.4, linewidths=0)
    ax.scatter(train_z[-2, 0].cpu().detach().numpy(), train_z[-2, 1].cpu().detach().numpy(), train_obj[-2, 0].cpu().detach().numpy(), color='purple', marker='v', alpha=0.4, linewidths=0)
    ax.scatter(train_z[-1, 0].cpu().detach().numpy(), train_z[-1, 1].cpu().detach().numpy(), train_obj[-1, 0].cpu().detach().numpy(), color='green', marker='*', alpha=0.4, linewidths=0)
    ax.plot_surface(XX, YY, Z, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'GP_' + str(epoch) + '.pdf'), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(train_z[0:N_orig, 0].cpu().detach().numpy(), train_z[0:N_orig, 1].cpu().detach().numpy(), train_obj[0:N_orig, 0].cpu().detach().numpy(), color='red', marker='X', alpha=0.4, linewidths=0)
    ax.scatter(train_z[N_orig:-2, 0].cpu().detach().numpy(), train_z[N_orig:-2, 1].cpu().detach().numpy(), train_obj[N_orig:-2, 0].cpu().detach().numpy(), color='red', marker='v', alpha=0.4, linewidths=0)
    ax.scatter(train_z[-2, 0].cpu().detach().numpy(), train_z[-2, 1].cpu().detach().numpy(), train_obj[-2, 0].cpu().detach().numpy(), color='purple', marker='v', alpha=0.4, linewidths=0)
    ax.scatter(train_z[-1, 0].cpu().detach().numpy(), train_z[-1, 1].cpu().detach().numpy(), train_obj[-1, 0].cpu().detach().numpy(), color='green', marker='*', alpha=0.4, linewidths=0)
    ax.plot_surface(XX, YY, Z, alpha=0.3)
    ax.elev = 1
    plt.savefig(os.path.join(save_path, 'GP_' + str(epoch) + 'rotate.pdf'), bbox_inches='tight')

    cmin = train_obj[:, 0].cpu().detach().numpy().min()
    cmax = train_obj[:, 0].cpu().detach().numpy().max()
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(XX, YY, Z)
    cb = fig.colorbar(cp)

    s1 = ax.scatter(train_z[0:N_orig, 0].cpu().detach().numpy(), train_z[0:N_orig, 1].cpu().detach().numpy(),
                    c=train_obj[0:N_orig, 0].cpu().detach().numpy(), alpha=0.5, marker='o', linewidths=0, cmap=cm.jet)
    s1.set_clim([cmin, cmax])
    cb2 = fig.colorbar(s1)
    s2 = ax.scatter(train_z[N_orig:-2, 0].cpu().detach().numpy(), train_z[N_orig:-2, 1].cpu().detach().numpy(),
                     c=train_obj[N_orig:-2, 0].cpu().detach().numpy(), alpha=0.5, marker='v', linewidths=0, cmap=cm.jet)
    s2.set_clim([cmin, cmax])
    s3 = ax.scatter(train_z[-2, 0].cpu().detach().numpy().reshape((1,)), train_z[-2, 1].cpu().detach().numpy().reshape((1,)),
                     c=train_obj[-2, 0].cpu().detach().numpy().reshape((1,)), alpha=0.5, marker='*', linewidths=0, cmap=cm.jet)
    s3.set_clim([cmin, cmax])
    s4 = ax.scatter(train_z[-1, 0].cpu().detach().numpy().reshape((1,)), train_z[-1, 1].cpu().detach().numpy().reshape((1,)),
                    c=train_obj[-1, 0].cpu().detach().numpy().reshape((1,)), alpha=0.5, marker='X', linewidths=0, cmap=cm.jet)
    s4.set_clim([cmin, cmax])
    cb.set_label('Cost - GP prediction')
    cb2.set_label('Cost - Data')
    plt.savefig(os.path.join(save_path, 'GP_contour_fill_' + str(epoch) + '.pdf'), bbox_inches='tight')

    cmin = train_obj[0:-1, 0].cpu().detach().numpy().min()
    cmax = train_obj[0:-1, 0].cpu().detach().numpy().max()
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(XX, YY, Z)
    cb = fig.colorbar(cp)

    cp1 = ax.scatter(train_z[0:N_orig, 0].cpu().detach().numpy(), train_z[0:N_orig, 1].cpu().detach().numpy(),
                     c=train_obj[0:N_orig, 0].cpu().detach().numpy(), alpha=0.5, marker='o', linewidths=0, cmap=cm.jet)
    cp1.set_clim([cmin, cmax])
    cb2 = fig.colorbar(cp1)
    cp2 = ax.scatter(train_z[N_orig:-2, 0].cpu().detach().numpy(), train_z[N_orig:-2, 1].cpu().detach().numpy(),
                     c=train_obj[N_orig:-2, 0].cpu().detach().numpy(), alpha=0.5, marker='v', linewidths=0, cmap=cm.jet)
    cp2.set_clim([cmin, cmax])
    cp3 = ax.scatter(train_z[-2, 0].cpu().detach().numpy().reshape((1,)), train_z[-2, 1].cpu().detach().numpy().reshape((1,)),
                     c=train_obj[-2, 0].cpu().detach().numpy().reshape((1,)), alpha=0.5, marker='*', linewidths=0, cmap=cm.jet)
    cp3.set_clim([cmin, cmax])
    cp4 = ax.scatter(train_z[-1, 0].cpu().detach().numpy().reshape((1,)), train_z[-1, 1].cpu().detach().numpy().reshape((1,)),
                    c='black', alpha=0.5, marker='X', linewidths=0)
#    cp4.set_clim([cmin, cmax])
    cb.set_label('Cost - GP prediction')
    cb2.set_label('Cost - Data')
    plt.savefig(os.path.join(save_path, 'GP_contour_line_' + str(epoch) + '.pdf'), bbox_inches='tight')

    cmin = train_obj[0:-1, 0].cpu().detach().numpy().min()
    cmax = train_obj[0:-1, 0].cpu().detach().numpy().max()
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(XX, YY, Z_var)
    cb = fig.colorbar(cp)

    cp1 = ax.scatter(train_z[0:N_orig, 0].cpu().detach().numpy(), train_z[0:N_orig, 1].cpu().detach().numpy(),
                     c=train_obj[0:N_orig, 0].cpu().detach().numpy(), alpha=0.5, marker='o', linewidths=0, cmap=cm.jet)
    cp1.set_clim([cmin, cmax])
    cb2 = fig.colorbar(cp1)
    cp2 = ax.scatter(train_z[N_orig:-2, 0].cpu().detach().numpy(), train_z[N_orig:-2, 1].cpu().detach().numpy(),
                     c=train_obj[N_orig:-2, 0].cpu().detach().numpy(), alpha=0.5, marker='v', linewidths=0, cmap=cm.jet)
    cp2.set_clim([cmin, cmax])
    cp3 = ax.scatter(train_z[-2, 0].cpu().detach().numpy().reshape((1,)), train_z[-2, 1].cpu().detach().numpy().reshape((1,)),
                     c=train_obj[-2, 0].cpu().detach().numpy().reshape((1,)), alpha=0.5, marker='*', linewidths=0, cmap=cm.jet)
    cp3.set_clim([cmin, cmax])
    cp4 = ax.scatter(train_z[-1, 0].cpu().detach().numpy().reshape((1,)), train_z[-1, 1].cpu().detach().numpy().reshape((1,)),
                     c='black', alpha=0.5, marker='X', linewidths=0)
    #    cp4.set_clim([cmin, cmax])
    cb.set_label('Cost - GP prediction (variance)')
    cb2.set_label('Cost - Data')
    plt.savefig(os.path.join(save_path, 'GP_contour_line_variance_' + str(epoch) + '.pdf'), bbox_inches='tight')

    # latent slice...
    train_z_np = train_z.cpu().detach().numpy()
    train_obj_np = train_obj.cpu().detach().numpy()
    xaxis_height = train_z_np[-1, 1]

    train_z_og = train_z_np[0:N_orig, :]
    train_obj_og = train_obj_np[0:N_orig, :]
    train_z_bo = train_z_np[N_orig:-1, :]
    train_obj_bo = train_obj_np[N_orig:-1, :]

    fig, ax = plt.subplots(1, 1)
    # original
    train_z_args = np.where((train_z_og[:, 1] >= (xaxis_height - 0.5)) & (train_z_og[:, 1] <= (xaxis_height + 0.5)))
    ax.scatter(train_z_og[train_z_args, 0], train_obj_og[train_z_args], alpha=0.5, marker='o', linewidths=0)

    # BO
    train_z_args = np.where((train_z_bo[:, 1] >= (xaxis_height - 0.5)) & (train_z_bo[:, 1] <= (xaxis_height + 0.5)))
    ax.scatter(train_z_bo[train_z_args, 0], train_obj_bo[train_z_args], alpha=0.5, marker='*', linewidths=0)

    # target
    ax.scatter(train_z_np[-1, 0], train_obj_np[-1], alpha=0.5, marker='X', linewidths=0)

    train_z_args = np.where((train_z_np[:, 1] >= (xaxis_height - 0.5)) & (train_z_np[:, 1] <= (xaxis_height + 0.5)))
    range_x1 = train_z_np[train_z_args, 0].min() - 1
    range_x2 = train_z_np[train_z_args, 0].max() + 1

    z1 = np.linspace(range_x1, range_x2, 50).reshape((-1, 1))
    z2 = np.ones((z1.shape[0], 1)) * xaxis_height
    z_test = np.concatenate((z1, z2), axis=1)
    z_test = np.concatenate((z_test, train_z_np[train_z_args, :].squeeze()), axis=0)
    z_test = z_test[z_test[:, 0].argsort()]
    val = model(torch.tensor(z_test).double().to(device))
    means = val.mean.detach().cpu().numpy()
    lower, upper = val.confidence_region()
    ax.plot(z_test[:, 0], means, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(z_test[:, 0], lower.detach().cpu().numpy(), upper.detach().cpu().numpy(), alpha=0.5)

    ax.set_xlabel('Position on h-line: %f ' % (xaxis_height))
    ax.set_ylabel('Cost')
    plt.savefig(os.path.join(save_path, 'GP_hline_' + str(epoch) + '.pdf'), bbox_inches='tight')

    print('GP plot generated.')


def get_fitted_model(train_z, train_obj, state_dict=None):
    # initialize and fit model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleTaskGP(train_X=train_z, train_Y=train_obj).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
    mll.to(train_z)
    fit_gpytorch_model(mll)
    print(model.covar_module.base_kernel.lengthscale)
    return model


def optimize_acqf_and_get_observation(acq_func, Q, nnet_model, image_goal_tensor, bounds):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""

    BATCH_SIZE = 1
    NUM_RESTARTS = 20
    RAW_SAMPLES = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(Q, dtype=torch.double, device=device),
            torch.ones(Q, dtype=torch.double, device=device),
        ]),
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # observe new values
    new_z = unnormalize(candidates.detach(), bounds=bounds)
#    print(image_goal_tensor.shape)
#    print(new_z.shape)
#    print(nnet_model.decode(new_z).shape)
    new_obj = score((nnet_model.decode(new_z)*255).reshape(1, 1, image_goal_tensor.shape[2], image_goal_tensor.shape[3]),
                    image_goal_tensor*255).unsqueeze(-1)  # add output dimension
    return new_z, new_obj


def run_BO_step(train_z, train_obj, nnet_model, image_goal_tensor, state_dict, MC_SAMPLES, seed, train_obj_min, train_obj_max,
                epoch, save_path, original_image_pad_tensor, target_label, mu_target, N_orig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = train_z.shape[1]
    state_dict = state_dict
    # min-max normalisation bounds
    bounds = torch.stack([torch.min(train_z, dim=0).values, torch.max(train_z, dim=0).values]).to(device)
    #    print(normalize(train_z, bounds=bounds).shape)
    #    print(standardize(train_obj).shape)
    # fit the model
    model = get_fitted_model(
        normalize(train_z, bounds=bounds),
        standardize(train_obj),
        state_dict=state_dict,
    ).to(device)

    # model = get_fitted_model(
    #     train_z,
    #     train_obj,
    #     state_dict=state_dict,
    # ).to(device)

    qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES).to(device)
    qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max()).to(device)

    # optimize and get new observation
    new_z, new_obj = optimize_acqf_and_get_observation(qEI, Q, nnet_model, image_goal_tensor, bounds)

    if epoch % 25 == 0 and Q == 2:
        new_y = nnet_model.decode(new_z).reshape(1, 1, image_goal_tensor.shape[2], image_goal_tensor.shape[3]).clone().detach()
        new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
        image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
#        img_best_cost = -((target_label - image_label)**2).sum().sqrt()
        img_best_cost = score(img_best * 255, image_goal_tensor * 255).unsqueeze(-1)
        print('img BO step cost: %f'%(img_best_cost))
        train_z = torch.cat((train_z, new_z), 0)
        train_z = torch.cat((train_z, mu_target), 0)
        train_obj = torch.cat((train_obj, img_best_cost.reshape(-1, 1)), 0)
        train_obj = torch.cat((train_obj, torch.tensor([0]).reshape(-1, 1).to(device)), 0)
        plot_GP(model, normalize(train_z, bounds=bounds), standardize(train_obj), epoch, save_path, N_orig)

    state_dict = model.state_dict()

    return new_z, -new_obj, state_dict

def min_max_norm(train_obj, min_cost, max_cost):
    return (train_obj - min_cost)/(max_cost - min_cost)

def obj_standardise(train_obj, train_obj_mean, train_obj_std):
    return (train_obj - train_obj_mean)/(train_obj_std)

def run_BO_step_simple(train_z, train_obj, nnet_model, image_goal_tensor, state_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = train_z.shape[1]
    state_dict = state_dict
    min_cost = -1410
    max_cost = 0
    print('Running BO simple EI.')
    # min-max normalisation bounds
    bounds = torch.stack([torch.min(train_z, dim=0).values, torch.max(train_z, dim=0).values]).to(device)
#    train_obj_norm = min_max_norm(train_obj, min_cost, max_cost)
#    train_obj_transform = torch.log(1 + train_obj_norm)

#    train_obj_standardised = min_max_norm(train_obj, train_obj_min, train_obj_max)
#    train_obj_transform = torch.log(1 + train_obj_standardised)
    #    print(normalize(train_z, bounds=bounds).shape)
    #    print(standardize(train_obj).shape)
    # fit the model
    model = get_fitted_model(
        normalize(train_z, bounds=bounds),
        standardize(train_obj),
        state_dict=state_dict,
    ).to(device)

    # model = get_fitted_model(
    #     train_z,
    #     train_obj,
    #     state_dict=state_dict,
    # ).to(device)

    EI = ExpectedImprovement(model, best_f=standardize(train_obj).max(), maximize=True).to(device)

#    qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed).to(device)
#    qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max()).to(device)

    # optimize and get new observation
    new_z, new_obj = optimize_acqf_and_get_observation(EI, Q, nnet_model, image_goal_tensor, bounds)

    state_dict = model.state_dict()

    return new_z, -new_obj, state_dict


def run_BO_step_UCB(train_z, train_obj, nnet_model, image_goal_tensor, state_dict, MC_SAMPLES, seed, train_obj_min,
                    train_obj_max, epoch, save_path, original_image_pad_tensor, target_label, mu_target, N_orig):
    print('Running UCB')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = train_z.shape[1]
    # min-max normalisation bounds
    bounds = torch.stack([torch.min(train_z, dim=0).values, torch.max(train_z, dim=0).values]).to(device)
    #    print(normalize(train_z, bounds=bounds).shape)
    #    print(standardize(train_obj).shape)
    # fit the model
    model = get_fitted_model(
        normalize(train_z, bounds=bounds),
        standardize(train_obj),
        state_dict=state_dict,
    ).to(device)


    ucb = UpperConfidenceBound(model, beta=0.5, maximize=True).to(device)

    #    qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed).to(device)
    #    qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max()).to(device)

    # optimize and get new observation
    new_z, new_obj = optimize_acqf_and_get_observation(ucb, Q, nnet_model, image_goal_tensor, bounds)
    if epoch % 25 == 0 and Q == 2:
        new_y = nnet_model.decode(new_z).reshape(1, 1, image_goal_tensor.shape[2], image_goal_tensor.shape[3]).clone().detach()
        new_rotation_value, new_x_shift_value, new_y_shift_value, img_best = covariate_finder(new_y, original_image_pad_tensor)
        #        image_label = torch.tensor([new_x_shift_value.reshape(1), new_y_shift_value.reshape(1), new_rotation_value.reshape(1)]).reshape(1, 3).to(device)
        #        img_best_cost = -((target_label - image_label)**2).sum().sqrt()
        img_best_cost = score(img_best * 255, image_goal_tensor * 255).unsqueeze(-1)
        print('img BO step cost: %f'%(img_best_cost))
        train_z = torch.cat((train_z, new_z), 0)
        train_z = torch.cat((train_z, mu_target), 0)
        train_obj = torch.cat((train_obj, img_best_cost.reshape(-1, 1)), 0)
        train_obj = torch.cat((train_obj, torch.tensor([0]).reshape(-1, 1).to(device)), 0)
        plot_GP(model, normalize(train_z, bounds=bounds), standardize(train_obj), epoch, save_path, N_orig)

    state_dict = model.state_dict()

    return new_z, -new_obj, state_dict


def covariate_finder(new_y, original_image_pad_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running grid search...')
    xval_min = 0
    yval_min = 0
    angle_min = 0
    min_loss = float('inf')
    new_y = new_y.to(device)
    for xval in range(-10, 10, 2):
        for yval in range(-10, 10, 2):
            for angle in range(0, 180, 2):
                img_tensor = K.geometry.transform.rotate(original_image_pad_tensor, torch.Tensor([angle]))
                img_tensor = K.geometry.transform.translate(img_tensor, torch.Tensor([[xval, yval]])).to(device)
                loss = F.mse_loss(img_tensor.float(), new_y.float()).sqrt()
                if loss < min_loss:
                    xval_min = xval
                    yval_min = yval
                    angle_min = angle
                    min_loss = loss

    # instantiate model
    m = Model(angle_min, xval_min, yval_min, original_image_pad_tensor.cpu()).to(device)

    # Instantiate optimizer
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    losses = []
    if min_loss == 0:
        print('Optimal found. Skipping optmisation.')
    else:
        print('Optimising...')
        losses = training_loop(m, opt, new_y)
        print('Covariate estimation loss: %f'%(losses[-1]))
    m.eval()

    # fig, ax = plt.subplots(1,1)
    # ax.imshow(np.reshape(new_y,[52,52]), cmap='gray')
    # ax.set_title('Target image')
    # plt.savefig('target.pdf', bbox_inches='tight')

    print('Angle: %f, Shift_X: %f, Shift_Y: %f'%(m.rotation, m.translate[0][0], m.translate[0][1]))
    img_best_tensor = K.geometry.transform.rotate(original_image_pad_tensor, m.rotation.detach().cpu())
    img_best_tensor = K.geometry.transform.translate(img_best_tensor, torch.Tensor([[m.translate[0][0].detach().cpu(),
                                                                                     m.translate[0][1].detach().cpu()]])).to(device)

    return m.rotation.detach().cpu().numpy()[0], m.translate[0][0].detach().cpu().numpy(), m.translate[0][1].detach().cpu().numpy(), img_best_tensor
