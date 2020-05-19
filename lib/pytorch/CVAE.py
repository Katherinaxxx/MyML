import torch
import torch.nn as nn
from utils import idx2onehot, load_data
import utils, torch, time, os, pickle
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt




class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        # return torch.reshape(recon_x, (-1,28, 28))
        return recon_x


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)


class CVAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 2
        self.class_num = 10
        self.minority = 6
        self.use_fake_data = args.use_fake_data
        self.fake_num = args.fake_num
        if self.use_fake_data:
            self.sample_num = self.fake_num
        else:
            self.sample_num = self.class_num ** 2
        self.conditional = True
        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.data_loader = load_data(self.dataset, imbalance=args.imbalance, batch_size=self.batch_size, shuffle=True)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.vae = VAE(
            encoder_layer_sizes=[784, 256],
            latent_size=self.z_dim,
            decoder_layer_sizes=[256, 784],
            conditional=self.conditional,
            num_labels=self.class_num if self.conditional else 0)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=0.0002)
        # self.optimizer = optim.Adam(self.vae.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.loss_fn = loss_fn

        if self.gpu_mode:
            self.vae.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.vae)
        print('-----------------------------------------------')

        # fixed noise(每一组0,1,2,..,9是一样的) & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        if self.use_fake_data:
            self.temp_y = torch.ones((self.sample_num, 1))*self.minority
        else:
            temp = torch.zeros((self.class_num, 1))
            for i in range(self.class_num):
                temp[i, 0] = i
            self.temp_y = torch.zeros((self.sample_num, 1))
            for i in range(self.class_num):
                self.temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, self.temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['CVAE_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.vae.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                recon_x, mean, log_var, z = self.vae(x_, y_)

                loss = self.loss_fn(recon_x, x_, mean, log_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_hist['CVAE_loss'].append(loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] CVAE_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.vae.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            # 跟其他不一样 y不需要onehot
            samples = self.vae.inference(self.sample_num, self.temp_y.long())
            # samples = self.vae.inference(self.sample_num, self.sample_y_.long())
        else:
            """ random noise """

            samples = self.vae.inference(self.batch_size)
        samples = (samples + 1) / 2

        if self.use_fake_data:
            return (samples.reshape([-1,1,self.input_size,self.input_size]), torch.zeros(self.sample_num)*self.minority)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:

            # samples = samples.data.numpy().transpose(0, 2, 3, 1)
            samples = samples.data.numpy().reshape([-1,28,28,1])

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

        # np.save(args.generate_file, data.data.numpy())

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.vae.state_dict(), os.path.join(save_dir, self.model_name + '_cvae.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.vae.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_cvae.pkl')))

        if self.use_fake_data:
            with torch.no_grad():
                return self.visualize_results(0)

