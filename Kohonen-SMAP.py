import os
import time
import torch
import argparse
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torchvision.utils import save_image


# SOM function

class SOM(nn.Module):
    def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None):
        '''
        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.lr = lr
        if sigma is None:
            self.sigma = max(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=False)
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)

    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location of best matching unit, loss
        '''
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]

        return bmu_locations, losses.sum().div_(batch_size).item()

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        bmu_locations, loss = self.forward(input)

        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        lr_locations = self._neighborhood_fn(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--color', dest='dataset', action='store_const',
                        const='color', default=None,
                        help='use color')
    parser.add_argument('--mnist', dest='dataset', action='store_const',
                        const='mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--fashion_mnist', dest='dataset', action='store_const',
                        const='fashion_mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--train', action='store_const',
                        const=True, default=False,
                        help='train network')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.3, help='input learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='input total epoch')
    parser.add_argument('--data_dir', type=str, default='datasets', help='set a data directory')
    parser.add_argument('--res_dir', type=str, default='results', help='set a result directory')
    parser.add_argument('--model_dir', type=str, default='model', help='set a model directory')
    parser.add_argument('--row', type=int, default=20, help='set SOM row length')
    parser.add_argument('--col', type=int, default=20, help='set SOM col length')
    args = parser.parse_args()

    # Hyper parameters
    DATA_DIR = args.data_dir + '/' + args.dataset
    RES_DIR = args.res_dir + '/' + args.dataset
    MODEL_DIR = args.model_dir + '/' + args.dataset

    # Create results dir
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # Create results/datasetname dir
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    # Create model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Create model/datasetname dir
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    dataset = args.dataset
    batch_size = args.batch_size
    total_epoch = args.epoch
    row = args.row
    col = args.col
    train = args.train

    if train is True:
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        if dataset == 'mnist':
            train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        elif dataset == 'fashion_mnist':
            train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            print('Please set specify dataset. --mnist, --fashion_mnist')
            exit(0)
    train_data.data = train_data.data[:10000]
    train_data.targets = train_data.targets[:10000]

    print('Building Model...')
    som = SOM(input_size=28 * 28 * 1, out_size=(row, col))
    if os.path.exists('%s/som.pth' % MODEL_DIR):
        som.load_state_dict(torch.load('%s/som.pth' % MODEL_DIR))
        print('Model Loaded!')
    else:
        print('Create Model!')
    som = som.to(device)

    if train == True:
        losses = list()
        for epoch in range(total_epoch):
            running_loss = 0
            start_time = time.time()
            for idx, (X, Y) in enumerate(train_loader):
                X = X.view(-1, 28 * 28 * 1).to(device)    # flatten
                loss = som.self_organizing(X, epoch, total_epoch)    # train som
                running_loss += loss

            losses.append(running_loss)
            print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss, time.time() - start_time))

            if epoch % 5 == 0:
                # save
                som.save_result('%s/som_epoch_%d.png' % (RES_DIR, epoch), (1, 28, 28))
                torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)

        torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)
        plt.title('SOM loss')
        plt.plot(losses)
        plt.show()

    som.save_result('%s/som_result.png' % (RES_DIR), (1, 28, 28))
    torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)
    plt.plot()
    plt.show()