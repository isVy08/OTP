import time

from trainer_base import TrainerBase
from util import *
from torch import nn


class OTP_Trainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(OTP_Trainer, self).__init__(
            cfgs, flgs, train_loader, val_loader, test_loader)
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
        }
        
    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        self.model.train()
        start_time = time.time()

        for batch_idx, (real_images, _) in enumerate(self.train_loader):
            
            real_images = real_images.cuda()
            _, _, loss = self.model(real_images, flg_train=True, flg_quant_det=False)

            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].detach().cpu().item())
            ms_error.append(loss["mse"].detach().cpu().item())
            perplexity.append(loss["perplexity"].detach().cpu().item())
        
        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
                
        return result    
    
    def _test(self, mode="validation"):

        self.model.eval()
        #_ = self._test_sub(False, mode)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result

    def _test_sub(self, flg_quant_det, mode="validation"):
        
        os.makedirs('image_results', exist_ok=True)
        os.makedirs('latent_results', exist_ok=True)
        if mode == "validation":
            data_loader = self.val_loader

        elif mode == "train":
            data_loader = self.train_loader
            
            save_path = 'image_results/train_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            
            save_latent_path = 'latent_results/train_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)
        
        elif mode == "test":
            data_loader = self.test_loader
            save_path = 'image_results/seed_1_test_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            save_latent_path = 'latent_results/seed_1_test_{}_{}_KL_{}_Beta_{}_{}'.format(self.cfgs.dataset.name,
                self.cfgs.quantization.name, self.cfgs.quantization.kl_regularization, 
                self.cfgs.quantization.beta, self.cfgs.quantization.size_dict)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path + '/train/', exist_ok=True)
            os.makedirs(save_path + '/rec/', exist_ok=True)
        

        start_time = time.time()
        
        test_loss = []
        recon_loss = 0.0
        histogram = torch.zeros(64, self.cfgs.quantization.size_dict).cuda()
        len_data  = len(data_loader.dataset)
        save_data = None
        save_label = None
        with torch.no_grad():
            i = 0
            for x, y in data_loader:
                x = x.cuda()
                if len(y.shape) > 1:
                    y = y.sum(1)
                x_reconst, min_encodings, e_indices, loss = self.model(x)
                #import pdb; pdb.set_trace()
                
                histogram += min_encodings.reshape(x_reconst.shape[0], 64, self.cfgs.quantization.size_dict).sum(0)
                recon_loss += ((x_reconst - x)**2).mean(3).mean(2).mean(1).sum()

                test_loss.append(loss["all"].item())

                if mode == "test" or mode == "train":
                    for idx in range(x.shape[0]):
                        save_image(tensor2im(x[idx]), save_path + '/train/' + str(i*self.cfgs.test.bs+idx)+'.png')
                        save_image(tensor2im(x_reconst[idx]), save_path + '/rec/' + str(i*self.cfgs.test.bs+idx)+'.png')
                
                    latent_size = int(x_reconst.shape[-1] / 4)
                    indices_numpy = e_indices.view(x.shape[0],latent_size,latent_size, 1).cpu().numpy()
                    if save_data is None:
                        save_data = indices_numpy
                        save_label = y.numpy()
                    else:
                        save_data = np.concatenate([save_data, indices_numpy],0)
                        save_label = np.concatenate([save_label, y.numpy()],0)
                i+=1
            recon_loss /= len_data  
            e_mean = histogram.sum(0)/(len_data*x_reconst.shape[-1]*x_reconst.shape[-1]/16)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            if mode == 'test':
                np.savez(save_latent_path, data=save_data, label=save_label, hist=histogram.cpu().numpy()) 
                np.savez(save_latent_path+'codebook_weight', weight=self.model.codebook_weight.cpu().numpy()) 
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = recon_loss
        
        result["perplexity"] = perplexity
        self.print_loss(result, mode, time.time()-start_time)
        
        return result
    

    def generate_reconstructions(self, filename, nrows=4, ncols=8):
        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    

    def print_loss(self, result, mode, time_interval):
        #import pdb; pdb.set_trace()
        myprint(mode.capitalize().ljust(16) +
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec"
            .format(
                result["loss"], result["mse"], result["perplexity"], time_interval
            ), self.flgs.noprint)

