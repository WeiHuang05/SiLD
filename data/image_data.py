"""
Image dataset wrappers for diffusion experiments.
Provides the same interface as manifold_data (sample_clean, sample_noisy, true_score).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os


class MNISTDataset:
    """
    MNIST flattened to R^784. Images normalized to [0, 1].
    Intrinsic dimension is estimated ~10-20.
    """

    def __init__(self, data_root='/home/weihuang/data', digit=None, max_samples=None):
        self.d = 784  # 28 * 28

        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1]
        ])

        dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )

        # Flatten and store all data in memory
        all_images = []
        all_labels = []
        for img, label in dataset:
            if digit is not None and label != digit:
                continue
            all_images.append(img.view(-1))  # (784,)
            all_labels.append(label)
            if max_samples and len(all_images) >= max_samples:
                break

        self.data = torch.stack(all_images)  # (N, 784)
        self.labels = torch.tensor(all_labels)
        self.n_total = len(self.data)

        # Compute PCA for analysis
        self._compute_pca()

        print(f"  MNIST loaded: {self.n_total} samples, d={self.d}")
        if digit is not None:
            print(f"  Filtered to digit {digit}")

    def _compute_pca(self):
        """Compute PCA of the dataset for analysis and LDM projection."""
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        # Use SVD on a subset for efficiency
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt  # (min(n_sub, 784), 784)
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        """Return top-d_latent PCA projection matrix (d_latent, 784)."""
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        """Sample n clean images."""
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        """Sample noisy images: x_t = x_0 + sigma_t * eps."""
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        """Denoising score: -(xt - x0) / sigma_t^2."""
        return -(xt - x0) / (sigma_t ** 2)


class CIFAR10Dataset:
    """
    CIFAR-10 flattened to R^3072. Images normalized to [0, 1].
    """

    def __init__(self, data_root='/home/weihuang/data', class_idx=None, max_samples=None):
        self.d = 3072  # 32 * 32 * 3

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )

        all_images = []
        all_labels = []
        for img, label in dataset:
            if class_idx is not None and label != class_idx:
                continue
            all_images.append(img.view(-1))
            all_labels.append(label)
            if max_samples and len(all_images) >= max_samples:
                break

        self.data = torch.stack(all_images)
        self.labels = torch.tensor(all_labels)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  CIFAR-10 loaded: {self.n_total} samples, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class SVHNDataset:
    """
    SVHN (Street View House Numbers) 32x32x3, flattened to R^3072.
    """

    def __init__(self, data_root='/home/weihuang/data', max_samples=20000):
        self.d = 3072

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.SVHN(
            root=data_root, split='train', download=False, transform=transform
        )

        all_images = []
        all_labels = []
        for i, (img, label) in enumerate(dataset):
            all_images.append(img.view(-1))
            all_labels.append(label)
            if max_samples and len(all_images) >= max_samples:
                break

        self.data = torch.stack(all_images)
        self.labels = torch.tensor(all_labels)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  SVHN loaded: {self.n_total} samples, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class FashionMNISTDataset:
    """
    FashionMNIST flattened to R^784. Images normalized to [0, 1].
    More complex than MNIST (clothing items vs digits).
    """

    def __init__(self, data_root='/home/weihuang/data', max_samples=None):
        self.d = 784

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=transform
        )

        all_images = []
        all_labels = []
        for img, label in dataset:
            all_images.append(img.view(-1))
            all_labels.append(label)
            if max_samples and len(all_images) >= max_samples:
                break

        self.data = torch.stack(all_images)
        self.labels = torch.tensor(all_labels)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  FashionMNIST loaded: {self.n_total} samples, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class ImageNet256Dataset:
    """
    ImageNet-1K resized to 256x256x3, flattened to R^196608.
    Images loaded on-the-fly from disk, preprocessed subset cached in memory.
    """

    def __init__(self, data_root='/home/weihuang/data', img_size=256,
                 max_samples=50000, split='train'):
        self.img_size = img_size
        self.d = img_size * img_size * 3  # 196608

        imagenet_dir = os.path.join(data_root, 'imagenet', 'images', split)
        if not os.path.exists(imagenet_dir):
            raise FileNotFoundError(f"ImageNet not found at {imagenet_dir}")

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=imagenet_dir, transform=transform,
        )

        # Load subset into memory
        print(f"  Loading ImageNet ({split}), max {max_samples} samples...")
        all_images = []
        all_labels = []
        for i, (img, label) in enumerate(dataset):
            if img.shape[0] != 3:
                continue  # skip grayscale
            all_images.append(img.view(-1))
            all_labels.append(label)
            if len(all_images) >= max_samples:
                break
            if (i + 1) % 10000 == 0:
                print(f"    loaded {len(all_images)}/{max_samples}...")

        self.data = torch.stack(all_images)
        self.labels = torch.tensor(all_labels)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  ImageNet-{img_size} loaded: {self.n_total} samples, "
              f"{img_size}x{img_size}x3, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(2000, self.n_total)
        # For 256x256 images, use randomized SVD (full SVD too expensive)
        # Project to smaller subspace first
        n_components = min(n_sub, 1000)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt[:n_components]
        self.pca_singular_values = S[:n_components]
        self.pca_mean = mean
        self.pca_explained_variance = (S[:n_components] ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class ImageNet256LazyDataset:
    """
    ImageNet-1K resized to 256x256x3. On-the-fly loading from disk.
    Only stores file paths in memory, not pixel data.
    Provides same interface as other datasets (sample_clean, etc.).
    """

    def __init__(self, data_root='/home/weihuang/data', img_size=256,
                 max_samples=200000, split='train'):
        self.img_size = img_size
        self.d = img_size * img_size * 3

        imagenet_dir = os.path.join(data_root, 'imagenet', 'images', split)
        if not os.path.exists(imagenet_dir):
            raise FileNotFoundError(f"ImageNet not found at {imagenet_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

        # Only store file paths, not images
        self.dataset = torchvision.datasets.ImageFolder(
            root=imagenet_dir, transform=self.transform,
        )

        # Filter to max_samples and record valid indices
        self.indices = []
        for i in range(min(len(self.dataset), max_samples * 2)):
            try:
                img, _ = self.dataset[i]
                if img.shape[0] == 3:
                    self.indices.append(i)
                if len(self.indices) >= max_samples:
                    break
            except Exception:
                continue
            if (len(self.indices)) % 50000 == 0:
                print(f"    indexed {len(self.indices)}/{max_samples}...")

        self.n_total = len(self.indices)
        self.labels = torch.zeros(self.n_total, dtype=torch.long)

        # Compute PCA on a small subset loaded into memory
        self._compute_pca()
        print(f"  ImageNet-{img_size} (lazy): {self.n_total} samples, "
              f"{img_size}x{img_size}x3, d={self.d}")

    def _compute_pca(self):
        n_sub = min(2000, self.n_total)
        sub_idx = torch.randperm(self.n_total)[:n_sub]
        imgs = []
        for i in sub_idx:
            img, _ = self.dataset[self.indices[i.item()]]
            imgs.append(img.view(-1))
        data_sub = torch.stack(imgs)
        mean = data_sub.mean(dim=0)
        centered = data_sub - mean
        n_components = min(n_sub, 1000)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.pca_components = Vt[:n_components]
        self.pca_singular_values = S[:n_components]
        self.pca_mean = mean
        self.pca_explained_variance = (S[:n_components] ** 2) / (n_sub - 1)

    def sample_clean(self, n):
        """Sample n clean images by loading from disk."""
        idx = torch.randint(0, self.n_total, (n,))
        imgs = []
        labels = []
        for i in idx:
            img, label = self.dataset[self.indices[i.item()]]
            imgs.append(img.view(-1))
            labels.append(label)
        return torch.stack(imgs), torch.tensor(labels)

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class CelebADataset:
    """
    CelebA faces resized to 64x64x3, flattened to R^12288. Images normalized to [0, 1].
    Loads from preprocessed tensor file for speed.
    """

    def __init__(self, data_root='/home/weihuang/data', img_size=64, max_samples=20000):
        self.img_size = img_size
        self.d = img_size * img_size * 3  # 12288 for 64x64

        # Try preprocessed tensor first
        tensor_path = os.path.join(data_root, 'celeba_faces', f'celeba_{img_size}x{img_size}.pt')
        if os.path.exists(tensor_path):
            self.data = torch.load(tensor_path)[:max_samples]
        else:
            # Fallback to torchvision CelebA
            transform = transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
            dataset = torchvision.datasets.CelebA(
                root=data_root, split='train', download=False, transform=transform
            )
            all_images = []
            for i, (img, _) in enumerate(dataset):
                all_images.append(img.view(-1))
                if max_samples and len(all_images) >= max_samples:
                    break
            self.data = torch.stack(all_images)

        self.labels = torch.zeros(len(self.data), dtype=torch.long)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  CelebA loaded: {self.n_total} samples, "
              f"{img_size}x{img_size}x3, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class CelebALazyDataset:
    """
    CelebA — loads from preprocessed .pt file.
    Same interface as CelebADataset.
    """

    def __init__(self, data_root='/home/weihuang/data', img_size=64, max_samples=200000):
        self.img_size = img_size
        self.d = img_size * img_size * 3

        tensor_path = os.path.join(data_root, 'celeba_faces', f'celeba_{img_size}x{img_size}.pt')
        if os.path.exists(tensor_path):
            self.data = torch.load(tensor_path)[:max_samples]
        else:
            raise FileNotFoundError(f"CelebA tensor not found at {tensor_path}")

        self.n_total = len(self.data)
        self.labels = torch.zeros(self.n_total, dtype=torch.long)

        self._compute_pca()
        print(f"  CelebA: {self.n_total} samples, "
              f"{img_size}x{img_size}x3, d={self.d}")

    def _compute_pca(self):
        n_sub = min(5000, self.n_total)
        mean = self.data[:n_sub].mean(dim=0)
        centered = self.data[:n_sub] - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class CelebAHQDataset:
    """
    CelebA-HQ resized to 64x64x3, flattened to R^12288. Images normalized to [0, 1].
    Loads from preprocessed tensor file.
    """

    def __init__(self, data_root='/home/weihuang/data', img_size=64, max_samples=20000):
        self.img_size = img_size
        self.d = img_size * img_size * 3

        tensor_path = os.path.join(data_root, 'celebahq', f'celebahq_{img_size}x{img_size}.pt')
        if os.path.exists(tensor_path):
            self.data = torch.load(tensor_path)[:max_samples]
        else:
            raise FileNotFoundError(
                f"Preprocessed CelebA-HQ not found at {tensor_path}. "
                "Run the download script first."
            )

        self.labels = torch.zeros(len(self.data), dtype=torch.long)
        self.n_total = len(self.data)

        self._compute_pca()
        print(f"  CelebA-HQ loaded: {self.n_total} samples, "
              f"{img_size}x{img_size}x3, d={self.d}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def get_pca_projection(self, d_latent):
        return self.pca_components[:d_latent]

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        xt = x0 + sigma_t * eps
        return xt, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)


class StackedMNISTDataset:
    """
    Stacked MNIST: 3 random MNIST digits stacked into a 3-channel 28x28 image.

    The classic 1000-mode benchmark. Each sample's "mode" is the tuple
    (d1, d2, d3) of digit classes in the R/G/B channels — 10^3 = 1000
    distinct modes total. Generative models that suffer mode collapse (e.g.
    VAEs because of KL pull, GANs from nonconvex losses) typically cover
    far fewer than 1000 modes.

    sample_clean returns (x_flat, labels) where x_flat is (n, 3*28*28)
    and labels is (n, 3) per-channel digit classes.
    """

    def __init__(self, data_root='/home/weihuang/data', n_pool=60000, seed=42):
        self.d = 3 * 28 * 28
        self.image_shape = (3, 28, 28)

        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        transform = transforms.Compose([transforms.ToTensor()])
        base = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform,
        )
        # Materialize all base images / labels for fast random indexing.
        all_imgs = torch.stack([img for img, _ in base])  # (60000, 1, 28, 28)
        all_lbls = torch.tensor([lbl for _, lbl in base])

        # Pre-build the stacked pool with random triples.
        n_base = all_imgs.shape[0]
        idx = rng.randint(0, n_base, size=(n_pool, 3))
        ch_imgs = all_imgs[idx]                # (n_pool, 3, 1, 28, 28)
        ch_imgs = ch_imgs.squeeze(2)           # (n_pool, 3, 28, 28)
        self.data = ch_imgs.reshape(n_pool, -1)  # (n_pool, 2352)
        self.labels = all_lbls[idx]            # (n_pool, 3)
        self.n_total = n_pool

        # 1000-mode index encoding for fast histogramming
        self.mode_idx = (self.labels[:, 0] * 100
                         + self.labels[:, 1] * 10
                         + self.labels[:, 2])  # (n_pool,) in [0, 1000)

        self._compute_pca()
        print(f"  StackedMNIST loaded: {self.n_total} pool samples, "
              f"d={self.d}, image_shape={self.image_shape}")

    def _compute_pca(self):
        mean = self.data.mean(dim=0)
        centered = self.data - mean
        n_sub = min(5000, self.n_total)
        U, S, Vt = torch.linalg.svd(centered[:n_sub], full_matrices=False)
        self.pca_components = Vt
        self.pca_singular_values = S
        self.pca_mean = mean
        self.pca_explained_variance = (S ** 2) / (n_sub - 1)

    def sample_clean(self, n):
        idx = torch.randint(0, self.n_total, (n,))
        return self.data[idx], self.labels[idx]

    def sample_noisy(self, n, sigma_t):
        x0, labels = self.sample_clean(n)
        eps = torch.randn_like(x0)
        return x0 + sigma_t * eps, x0, eps, labels

    def true_score(self, xt, x0, sigma_t):
        return -(xt - x0) / (sigma_t ** 2)
