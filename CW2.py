import os
import torch
import numpy as np
import soundfile as sf
from attack.FGSM import FGSM
from attack.utils import SEC4SR_MarginLoss

class CW2(FGSM):
    global_index = 0

    def __init__(self, model, task='CSI',
                targeted=False,
                confidence=0.,
                initial_const=1e-3, 
                binary_search_steps=9,
                max_iter=10000,
                stop_early=True,
                stop_early_iter=1000,
                lr=1e-2,
                batch_size=1,
                verbose=1,
                noise_folder='/home/adgroup2/mnt1/xvectors/SpeakerGuard/adv_output/noise/CW2',
                original_filenames_path='/home/adgroup2/mnt1/xvectors/SpeakerGuard/data/adv_origin/123504'):

        self.model = model
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.noise_folder = noise_folder
        self.original_filenames = []

        # Ensure the noise folder exists
        os.makedirs(self.noise_folder, exist_ok=True)

        # Load original filenames
        if original_filenames_path and os.path.isdir(original_filenames_path):
            self.original_filenames = sorted([
                os.path.join(original_filenames_path, f) 
                for f in os.listdir(original_filenames_path) 
                if os.path.isfile(os.path.join(original_filenames_path, f))
            ])

        self.threshold = None
        if self.task in ['SV', 'OSI']:
            self.threshold = self.model.threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))

        self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=True)

        self.i = 0  # Initialize i as an instance attribute

    def save_noise(self, noise, file_name):
        """保存噪声到指定文件夹"""
        noise_file_path = os.path.join(self.noise_folder, file_name)
        # 使用默认的采样率，例如16000
        sample_rate = 16000
        sf.write(noise_file_path, noise, sample_rate)
        print(f"已生成噪音文件: {noise_file_path}")

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        n_audios, _, _ = x_batch.shape

        const = torch.tensor([self.initial_const] * n_audios, dtype=torch.float, device=x_batch.device)
        lower_bound = torch.tensor([0] * n_audios, dtype=torch.float, device=x_batch.device)
        upper_bound = torch.tensor([1e10] * n_audios, dtype=torch.float, device=x_batch.device)

        global_best_l2 = [np.infty] * n_audios
        global_best_adver_x = x_batch.clone()
        global_best_score = [-2] * n_audios

        for _ in range(self.binary_search_steps):

            self.modifier = torch.zeros_like(x_batch, dtype=torch.float, requires_grad=True, device=x_batch.device)
            self.optimizer = torch.optim.Adam([self.modifier], lr=self.lr)

            best_l2 = [np.infty] * n_audios
            best_score = [-2] * n_audios

            continue_flag = True
            prev_loss = np.infty

            for n_iter in range(self.max_iter + 1): 
                if not continue_flag:
                    break
                
                input_x = torch.tanh(self.modifier + torch.atanh(x_batch * 0.999999))
                decisions, scores = self.model.make_decision(input_x)
                loss1 = self.loss(scores, y_batch)
                loss2 = torch.sum(torch.square(input_x - x_batch), dim=(1,2))
                loss = const * loss1 + loss2

                if n_iter < self.max_iter: 
                    loss.backward(torch.ones_like(loss))
                    self.optimizer.step()
                    self.modifier.grad.zero_()

                predict = decisions.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy().tolist()
                loss1 = loss1.detach().cpu().numpy().tolist()
                loss2 = loss2.detach().cpu().numpy().tolist()
                if self.verbose:
                    print("batch: {}, c: {}, iter: {}, loss: {}, loss1: {}, loss2: {}, y_pred: {}, y: {}".format(
                        batch_id, const.detach().cpu().numpy(), n_iter, 
                        loss, loss1, loss2, predict, y_batch.detach().cpu().numpy()))
                
                if self.stop_early and n_iter % self.stop_early_iter == 0:
                    if np.mean(loss) > 0.9999 * prev_loss:
                        print("Early Stop!")
                        continue_flag = False
                    prev_loss = np.mean(loss)

                for ii, (l2, y_pred, adver_x, l1) in enumerate(zip(loss2, predict, input_x, loss1)):
                    if l1 <= 0 and l2 < best_l2[ii]:
                        best_l2[ii] = l2
                        best_score[ii] = y_pred
                    if l1 <= 0 and l2 < global_best_l2[ii]:
                        global_best_l2[ii] = l2
                        global_best_score[ii] = y_pred
                        global_best_adver_x[ii] = adver_x

            for jj, y_pred in enumerate(best_score):
                if y_pred != -2:
                    upper_bound[jj] = min(upper_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                else:
                    lower_bound[jj] = max(lower_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                    else:
                        const[jj] *= 10
            
            print(const.detach().cpu().numpy(), best_l2, global_best_l2)
        
        success = [False] * n_audios
        for kk, y_pred in enumerate(global_best_score):
            if y_pred != -2:
                success[kk] = True 
        
        # 保存噪声
        # Naming noise files based on original filenames
        noise = (global_best_adver_x - x_batch).detach().cpu().numpy()
        for noise_sample in noise:
            file_name = os.path.basename(self.original_filenames[self.i])
            print(file_name)
            print(self.i)
            self.i += 1  # Increment self.i for the next iteration
            file_name = os.path.splitext(file_name)[0] + '_noise.wav'
            self.save_noise(noise_sample[0], file_name)

        return global_best_adver_x, success

    def attack(self, x, y):
        batch_size = self.batch_size
        n_audios = x.shape[0]
        n_batches = (n_audios + batch_size - 1) // batch_size
        adver_x = torch.zeros_like(x)
        success = []
        for batch_id in range(n_batches):
            start = batch_id * batch_size
            end = min(start + batch_size, n_audios)
            x_batch = x[start:end]
            y_batch = y[start:end]
            lower_batch = torch.zeros_like(x_batch)
            upper_batch = torch.ones_like(x_batch)
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            adver_x[start:end] = adver_x_batch
            success.extend(success_batch)
        return adver_x, success
