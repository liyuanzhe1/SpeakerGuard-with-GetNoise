import soundfile as sf
from attack.Attack import Attack
from attack.utils import resolve_loss
from adaptive_attack.NES import NES
from adaptive_attack.EOT import EOT
import torch
import numpy as np
import os

class FAKEBOB(Attack):
    global_index = 0

    def __init__(self, model, threshold=None,
                task='CSI', targeted=False, confidence=0.,
                epsilon=0.002, max_iter=1000,
                max_lr=0.001, min_lr=1e-6,
                samples_per_draw=50, samples_per_draw_batch_size=50, sigma=0.001, momentum=0.9,
                plateau_length=5, plateau_drop=2.,
                stop_early=True, stop_early_iter=100,
                batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1,
                noise_folder='/home/adgroup2/mnt1/xvectors/SpeakerGuard/adv_output/noise/fakebob',
                original_filenames_path='/home/adgroup2/mnt1/xvectors/SpeakerGuard/data/adv_origin/123504'):
        
        self.model = model
        self.threshold = threshold
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.samples_per_draw = samples_per_draw
        self.samples_per_draw_batch_size = samples_per_draw_batch_size
        self.sigma = sigma
        self.momentum = momentum
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.batch_size = batch_size
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose
        self.noise_folder = noise_folder
        self.original_filenames = []
        self.i = 0  # Initialize file index

        # Ensure the noise folder exists
        os.makedirs(self.noise_folder, exist_ok=True)

        # Load original filenames
        if original_filenames_path and os.path.isdir(original_filenames_path):
            self.original_filenames = sorted([
                os.path.join(original_filenames_path, f) 
                for f in os.listdir(original_filenames_path) 
                if os.path.isfile(os.path.join(original_filenames_path, f))
            ])

    def save_noise(self, noise, file_name):
        """保存噪声到指定文件夹"""
        noise_file_path = os.path.join(self.noise_folder, file_name)
        # 使用默认的采样率，例如16000
        sample_rate = 16000
        sf.write(noise_file_path, noise, sample_rate)
        print(f"已生成噪音文件: {noise_file_path}")

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):
     with torch.no_grad():
        n_audios, _, _ = x_batch.shape

        last_ls = [[]] * n_audios
        lr = [self.max_lr] * n_audios
        prev_loss = [np.infty] * n_audios

        adver_x = x_batch.clone()
        grad = torch.zeros_like(x_batch, dtype=x_batch.dtype, device=x_batch.device)

        best_adver_x = adver_x.clone()
        best_loss = [np.infty] * n_audios
        consider_index = list(range(n_audios))

        for iter in range(self.max_iter + 1):
            prev_grad = grad.clone()
            loss, grad, adver_loss, _, y_pred = self.get_grad(adver_x, y_batch)

            for ii, adver_l in enumerate(adver_loss):
                index = consider_index[ii]
                if adver_l < best_loss[index]:
                    best_loss[index] = adver_l.cpu().item()
                    best_adver_x[index] = adver_x[ii]

            if self.verbose:
                print("batch: {} iter: {}, loss: {}, y: {}, y_pred: {}, best loss: {}".format(
                    batch_id, iter, 
                    adver_loss.cpu().numpy(), y_batch.cpu().numpy(), y_pred, best_loss))
            
            result = self.delete_found(adver_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                                       consider_index, last_ls, lr, prev_loss, loss)
            if result is None:  # all found
                break
            
            adver_x, y_batch, prev_grad, grad, lower, upper, consider_index, last_ls, lr, prev_loss, loss = result

            if iter < self.max_iter:
                grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                for jj, loss_ in enumerate(loss):
                    last_ls[jj].append(loss_)
                    last_ls[jj] = last_ls[jj][-self.plateau_length:]
                    if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                        if lr[jj] > self.min_lr:
                            lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                        last_ls[jj] = []
                
                lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                if self.stop_early and iter % self.stop_early_iter == 0:
                    loss_np = np.array([l.cpu() for l in loss])
                    converge_loss = np.array(prev_loss) * 0.9999 - loss_np
                    result = self.delete_found(converge_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                                               consider_index, last_ls, lr, prev_loss, loss)
                    if result is None:  # all converged
                        break
                    
                    adver_x, y_batch, prev_grad, grad, lower, upper, consider_index, last_ls, lr, prev_loss, loss = result

                    prev_loss = loss_np
        
        success = [False] * n_audios
        for kk, best_l in enumerate(best_loss):
            if best_l < 0:
                success[kk] = True

        # 保存噪声
        noise = (best_adver_x - x_batch).detach().cpu().numpy()
        for noise_sample in noise:
            file_name = os.path.basename(self.original_filenames[self.i])
            self.i += 1  # Increment self.i for the next iteration
            file_name = os.path.splitext(file_name)[0] + '_noise.wav'
            self.save_noise(noise_sample[0], file_name)
        
        return best_adver_x, success

    def delete_found(self, adver_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                 consider_index, last_ls, lr, prev_loss, loss):
        adver_x_u = None
        y_batch_u = None
        prev_grad_u = None
        grad_u = None
        lower_u = None
        upper_u = None

        consider_index_u = []
        last_ls_u = []
        lr_u = []
        prev_loss_u = []
        loss_u = []
    
        for ii, adver_l in enumerate(adver_loss):
            if adver_l < 0:
                pass
            else:
                if adver_x_u is None:
                    adver_x_u = adver_x[ii:ii+1, ...]
                    y_batch_u = y_batch[ii:ii+1]
                    prev_grad_u = prev_grad[ii:ii+1, ...]
                    grad_u = grad[ii:ii+1, ...]
                    lower_u = lower[ii:ii+1, ...]
                    upper_u = upper[ii:ii+1, ...]
                else:
                    adver_x_u = torch.cat((adver_x_u, adver_x[ii:ii+1, ...]), 0)
                    y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii+1]))
                    prev_grad_u = torch.cat((prev_grad_u, prev_grad[ii:ii+1, ...]), 0)
                    grad_u = torch.cat((grad_u, grad[ii:ii+1, ...]), 0)
                    lower_u = torch.cat((lower_u, lower[ii:ii+1, ...]), 0)
                    upper_u = torch.cat((upper_u, upper[ii:ii+1, ...]), 0)
                index = consider_index[ii]
                consider_index_u.append(index)
                last_ls_u.append(last_ls[ii])
                lr_u.append(lr[ii])
                prev_loss_u.append(prev_loss[ii])
                loss_u.append(loss[ii])

        if adver_x_u is None:
            return None

        return adver_x_u, y_batch_u, prev_grad_u, \
               grad_u, lower_u, upper_u, consider_index_u, \
               last_ls_u, lr_u, prev_loss_u, loss_u


    def get_grad(self, x, y):
        NES_wrapper = NES(self.samples_per_draw, self.samples_per_draw_batch_size, self.sigma, self.EOT_wrapper)
        mean_loss, grad, adver_loss, adver_score, predict = NES_wrapper(x, y)
        
        return mean_loss, grad, adver_loss, adver_score, predict

    def attack(self, x, y):

        if self.task in ['SV', 'OSI'] and self.threshold is None:
            raise NotImplementedError('You are running black box attack for {} task, \
                        but the threshold not specified. Consider calling estimate threshold')
        self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold, False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
            upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success
    
    def estimate_threshold_run(self, x, step=0.1):

        n_audios, _, _ = x.shape

        d, s = self.model.make_decision(x)
        d = d[0]
        s = s[0]
        if d != -1:
            return # already accept, cannot be used to estimate threshold
        y = torch.tensor([-1] * n_audios, dtype=torch.long, device=x.device)
        init_score = np.max(s.cpu().numpy())
        delta = np.abs(init_score * step)
        threshold = init_score + delta

        adver_x = x.clone()
        grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        lower = -1
        upper = 1
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        iter_outer = 0
        n_iters = 0

        while True:
            self.loss, self.grad_sign = resolve_loss('Margin', False, 0., self.task, threshold, False)
            self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

            iter_inner = 0

            last_ls = [[]] * n_audios
            lr = [self.max_lr] * n_audios

            while True:

                # test whether succeed
                decision, score = self.model.make_decision(adver_x)
                decision = decision[0]
                score = score[0]
                score = np.max(score.cpu().numpy())
                print(iter_outer, iter_inner, score, self.model.threshold)
                if decision != -1: # succeed, found the threshold
                    return score
                elif score >= threshold: # exceed the candidate threshold, but not succeed, exit the inner loop and increase the threshold
                    break

                # not succeed, update
                prev_grad = grad.clone()
                loss, grad, _, _, _ = self.get_grad(adver_x, y)

                grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                for jj, loss_ in enumerate(loss):
                    last_ls[jj].append(loss_)
                    last_ls[jj] = last_ls[jj][-self.plateau_length:]
                    if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                        if lr[jj] > self.min_lr:
                            lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                        last_ls[jj] = []
                
                lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                iter_inner += 1
                n_iters += 1
            
            threshold += delta
            iter_outer += 1

    def estimate_threshold(self, x, step=0.1):
        if self.task == 'CSI':
            print("--- Warning: no need to estimate threshold for CSI, quitting ---")
            return
        
        with torch.no_grad():
            estimated_thresholds = []
            for xx in x.unsqueeze(0): # parallel running, not easy for batch running
                estimated_threshold = self.estimate_threshold_run(xx, step)
                if estimated_threshold is not None:
                    estimated_thresholds.append(estimated_threshold)
            if len(estimated_thresholds) > 0:
                self.threshold = np.mean(estimated_thresholds)
            else:
                self.threshold = None
            return self.threshold
