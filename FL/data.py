import copy
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataset, Subset
from utility.cutout import Cutout
import numpy as np

class Dataset(dataset.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        np.random.seed(114514)
        targets = np.array(self.targets)
        n_classes = targets.max() + 1
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        class_idcs = [np.argwhere(targets == y).flatten() for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs
    
    def uniform_split_iid(self, n_clients):
        np.random.seed(19260817)
        targets = np.array(self.targets)
        n_classes = targets.max() + 1
        label_distribution = np.array([[1.0 / n_clients] * n_clients] * n_classes)
        class_idcs = [np.argwhere(targets == y).flatten() for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs
    
    
class Dataset_CIFAR10(Dataset):
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_CIFAR10, self).__init__(data, targets)
        
    def load(self, path='./dataset/cifar10', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        dataset = torchvision.datasets.CIFAR10(root=path, train=train, download=True, transform=transform)
        data = []; targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_CIFAR10, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_CIFAR10, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR10(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_CIFAR10, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR10(data, targets))
        return datasets
        
    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./dataset/cifar10/temp', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    

class Dataset_CIFAR100(Dataset):
    
    classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',
    )
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_CIFAR100, self).__init__(data, targets)
        
    def load(self, path='./dataset/cifar100', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        dataset = torchvision.datasets.CIFAR100(root=path, train=train, download=True, transform=transform)
        data = []; targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_CIFAR100, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_CIFAR100, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR100(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_CIFAR100, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR100(data, targets))
        return datasets
        
    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./dataset/cifar100/temp', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

    
class Dataset_MNIST(Dataset):
    
    classes = (
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    )
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_MNIST, self).__init__(data, targets)
        
    def load(self, path='./dataset/mnist', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        dataset = torchvision.datasets.MNIST(root=path, train=train, download=True, transform=transform)
        data = []; targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = torch.cat([img, img, img])
            data.append(img)
            targets.append(target)

        super(Dataset_MNIST, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_MNIST, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_MNIST(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_MNIST, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_MNIST(data, targets))
        return datasets
        
    def _get_statistics(self):
        train_set = torchvision.datasets.MNIST(root='./dataset/mnist/temp', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    
    
class Dataset_FMNIST(Dataset):
    
    classes = (
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    )
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_FMNIST, self).__init__(data, targets)
        
    def load(self, path='./dataset/fmnist', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        dataset = torchvision.datasets.FashionMNIST(root=path, train=train, download=True, transform=transform)
        data = []; targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = torch.cat([img, img, img])
            data.append(img)
            targets.append(target)

        super(Dataset_FMNIST, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_FMNIST, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FMNIST(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_FMNIST, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FMNIST(data, targets))
        return datasets
        
    def _get_statistics(self):
        train_set = torchvision.datasets.FashionMNIST(root='./dataset/fmnist/temp', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    
    
class Dataset_FLOWERS102(Dataset):
    
    classes = {'21': 'fire lily',
                '3': 'canterbury bells',
                '45': 'bolero deep blue',
                '1': 'pink primrose',
                '34': 'mexican aster',
                '27': 'prince of wales feathers',
                '7': 'moon orchid',
                '16': 'globe-flower',
                '25': 'grape hyacinth',
                '26': 'corn poppy',
                '79': 'toad lily',
                '39': 'siam tulip',
                '24': 'red ginger',
                '67': 'spring crocus',
                '35': 'alpine sea holly',
                '32': 'garden phlox',
                '10': 'globe thistle',
                '6': 'tiger lily',
                '93': 'ball moss',
                '33': 'love in the mist',
                '9': 'monkshood',
                '102': 'blackberry lily',
                '14': 'spear thistle',
                '19': 'balloon flower',
                '100': 'blanket flower',
                '13': 'king protea',
                '49': 'oxeye daisy',
                '15': 'yellow iris',
                '61': 'cautleya spicata',
                '31': 'carnation',
                '64': 'silverbush',
                '68': 'bearded iris',
                '63': 'black-eyed susan',
                '69': 'windflower',
                '62': 'japanese anemone',
                '20': 'giant white arum lily',
                '38': 'great masterwort',
                '4': 'sweet pea',
                '86': 'tree mallow',
                '101': 'trumpet creeper',
                '42': 'daffodil',
                '22': 'pincushion flower',
                '2': 'hard-leaved pocket orchid',
                '54': 'sunflower',
                '66': 'osteospermum',
                '70': 'tree poppy',
                '85': 'desert-rose',
                '99': 'bromelia',
                '87': 'magnolia',
                '5': 'english marigold',
                '92': 'bee balm',
                '28': 'stemless gentian',
                '97': 'mallow',
                '57': 'gaura',
                '40': 'lenten rose',
                '47': 'marigold',
                '59': 'orange dahlia',
                '48': 'buttercup',
                '55': 'pelargonium',
                '36': 'ruby-lipped cattleya',
                '91': 'hippeastrum',
                '29': 'artichoke',
                '71': 'gazania',
                '90': 'canna lily',
                '18': 'peruvian lily',
                '98': 'mexican petunia',
                '8': 'bird of paradise',
                '30': 'sweet william',
                '17': 'purple coneflower',
                '52': 'wild pansy',
                '84': 'columbine',
                '12': "colt's foot",
                '11': 'snapdragon',
                '96': 'camellia',
                '23': 'fritillary',
                '50': 'common dandelion',
                '44': 'poinsettia',
                '53': 'primula',
                '72': 'azalea',
                '65': 'californian poppy',
                '80': 'anthurium',
                '76': 'morning glory',
                '37': 'cape flower',
                '56': 'bishop of llandaff',
                '60': 'pink-yellow dahlia',
                '82': 'clematis',
                '58': 'geranium',
                '75': 'thorn apple',
                '41': 'barbeton daisy',
                '95': 'bougainvillea',
                '43': 'sword lily',
                '83': 'hibiscus',
                '78': 'lotus lotus',
                '88': 'cyclamen',
                '94': 'foxglove',
                '81': 'frangipani',
                '74': 'rose',
                '89': 'watercress',
                '73': 'water lily',
                '46': 'wallflower',
                '77': 'passion flower',
                '51': 'petunia'
            }
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_FLOWERS102, self).__init__(data, targets)
        
    def load(self, path='./dataset/flowers102', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        dataset = torchvision.datasets.Flowers102(root=path, split=('train' if train else 'test'), download=True, transform=transform)
        data = []; targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_FLOWERS102, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_FLOWERS102, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FLOWERS102(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_FLOWERS102, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FLOWERS102(data, targets))
        return datasets
        
    def _get_statistics(self):
        train_set = torchvision.datasets.Flowers102(root='./dataset/flowers102/temp', split='train', download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]))
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    
    
class Dataset_EUROSAT(Dataset):
    
    classes = ('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')
        
    def __init__(self, data=[], targets=[]):
        super(Dataset_EUROSAT, self).__init__(data, targets)
        
    def load(self, path='./dataset/eurosat/2750', train=True):

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            Cutout()
        ]) if train == True else transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        Dataset_EUROSAT.classes = dataset.classes
        random.seed(19260817)
        dataset_len = len(dataset)
        group = [[] for _ in range(len(Dataset_EUROSAT.classes))]
        for i in range(dataset_len):
            img, target = dataset[i]
            group[target].append(i)
            
        indices = []
        for g in group:
            train_len = int(10.0 / 27.0 * len(g))
            
            random.shuffle(g)
            if train == True:
                indices += g[:train_len]
            else:
                indices += g[train_len:]
                
        print("{} data finish, length = {}".format('train' if train ==True else 'test', len(indices)))
        
        data = []; targets = []
        for i in indices:
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_EUROSAT, self).__init__(data, targets)
        
    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)
        
    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_EUROSAT, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_EUROSAT(data, targets))
        return datasets
    
    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_EUROSAT, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []; targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_EUROSAT(data, targets))
        return datasets