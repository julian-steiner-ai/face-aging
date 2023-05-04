"""
PFA-GAN.
"""

from torch import nn
from typing import Union
from torch.nn.utils import spectral_norm

from apex.parallel import DistributedDataParallel

import torch
import torch.utils.data as tordata
import torchvision
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        m.weight.data.normal_(0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.zero_()


def to_ddp(modules: Union[list, nn.Module], optimizer: torch.optim.Optimizer = None, opt_level: int = 0) -> Union[
    DistributedDataParallel, tuple]:
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if optimizer is not None:
        modules, optimizer = amp.initialize(modules, optimizer, opt_level="O{}".format(opt_level), verbosity=1)
    if isinstance(modules, list):
        modules = [DistributedDataParallel(x, delay_allreduce=True) for x in modules]
    else:
        modules = DistributedDataParallel(modules, delay_allreduce=True)
    if optimizer is not None:
        return modules, optimizer
    else:
        return modules


def age2group(age, age_group):
    if isinstance(age, np.ndarray):
        groups = np.zeros_like(age)
    else:
        groups = torch.zeros_like(age).to(age.device)
    if age_group == 4:
        section = [30, 40, 50]
    elif age_group == 5:
        section = [20, 30, 40, 50]
    elif age_group == 7:
        section = [10, 20, 30, 40, 50, 60]
    else:
        raise NotImplementedError
    for i, thresh in enumerate(section, 1):
        groups[age > thresh] = i
    return groups


def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
            module,
            nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return nn.utils.spectral_norm(module, **kwargs)
    else:
        return NotImplementedError


def group2onehot(groups, age_group):
    code = torch.eye(age_group)[groups.squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0)


def group2feature(group, age_group, feature_size):
    onehot = group2onehot(group, age_group)
    return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)


def get_dex_age(pred):
    pred = F.softmax(pred, dim=1)
    value = torch.sum(pred * torch.arange(pred.size(1)).to(pred.device), dim=1)
    return value


def pfa_encoding(source, target, age_group):
    source, target = source.long(), target.long()
    code = torch.zeros((source.size(0), age_group - 1, 1, 1, 1)).to(source)
    for i in range(source.size(0)):
        code[i, source[i]: target[i], ...] = 1
    return code


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root):
        assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        for i in range(len(stats)):
            var = stats[i]
            if isinstance(var, dict):
                group_name = list(var.keys())[0]
                for j in range(len(var[group_name])):
                    scalar_name = get_varname(var[group_name][j])
                    scalar = reduce_tensor(var[group_name][j].detach().mean(), self.world_size).item()
                    output_str += '{} {:2.5f}, '.format(scalar_name, scalar)
                    self.writer.add_scalars(group_name, {
                        scalar_name: scalar
                    }, step)
            else:
                var_name = get_varname(stats[i])
                var = reduce_tensor(var.detach().mean(), self.world_size).item()
                output_str += '{} {:2.5f}, '.format(var_name, var)
        if self.local_rank == 0:
            print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim_loss(img1, img2, window_size=11):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()

def normalize(input, mean, std):
    mean = torch.Tensor(mean).to(input.device)
    std = torch.Tensor(std).to(input.device)
    return input.sub(mean[None, :, None, None]).div(std[None, :, None, None])

def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
            module,
            nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return spectral_norm(module, **kwargs)
    else:
        return NotImplementedError

class PFA_GAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.prefetcher = self.get_train_loader()
        self.test_images = self.get_test_images()
        self.logger = LoggerX(osp.join('materials', 'checkpoints', opt.name))
        self.init_model()

    def get_test_images(self):
        opt = self.opt
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        test_dataset = GroupDataset(
            age_group=opt.age_group,
            train=False,
            group=opt.source,
            dataset_name=opt.dataset_name,
            transforms=transforms)
        test_sampler = tordata.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_loader = tordata.DataLoader(
            dataset=test_dataset,
            batch_size=opt.batch_size * 5,
            shuffle=False,
            drop_last=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=test_sampler
        )
        return next(iter(test_loader)).cuda()

    def get_train_loader(self):
        opt = self.opt
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.pretrained_image_size),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = PFADataset(
            age_group=opt.age_group,
            max_iter=opt.max_iter,
            batch_size=opt.batch_size * len(opt.device_ids),
            dataset_name=opt.dataset_name,
            source=opt.source,
            transforms=transforms)
        train_sampler = tordata.distributed.DistributedSampler(train_dataset, shuffle=False)

        train_loader = tordata.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            drop_last=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        # source_img, true_img, source_label, target_label, true_label, true_age, mean_age
        return data_prefetcher(train_loader, [0, 1])

    def init_model(self):
        opt = self.opt
        generator = Generator(norm_layer='in', age_group=opt.age_group)
        generator.apply(weights_init)
        discriminator = PatchDiscriminator(age_group=opt.age_group,
                                           repeat_num=int(np.log2(opt.image_size) - 4), norm_layer='sn')

        vgg_face = torchvision.models.vgg16(num_classes=2622)
        vgg_face.load_state_dict(load_network(osp.join('materials', 'DeepFaceVGG_RGB.pth')))
        vgg_face = vgg_face.features[:23]
        vgg_face.eval()

        age_classifier = AuxiliaryAgeClassifier(age_group=opt.age_group,
                                                repeat_num=int(np.log2(opt.pretrained_image_size) - 4))
        age_classifier.load_state_dict(load_network(osp.join('materials',
                                                             'dex_simple_{}_{}_age_classifier.pth'.format(
                                                                 opt.dataset_name, opt.pretrained_image_size))))
        age_classifier.eval()

        d_optim = torch.optim.Adam(discriminator.parameters(), opt.init_lr, betas=(0.5, 0.99))
        g_optim = torch.optim.Adam(generator.parameters(), opt.init_lr, betas=(0.5, 0.99))

        self.logger.modules = [generator, discriminator, d_optim, g_optim]

        if opt.restore_iter > 0:
            self.logger.load_checkpoints(opt.restore_iter)

        self.generator, self.g_optim = to_ddp(generator, g_optim)
        self.discriminator, self.d_optim = to_ddp(discriminator, d_optim)
        self.vgg_face = to_ddp(vgg_face)
        self.age_classifier = to_ddp(age_classifier)

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.generator.eval()
        real_img = self.test_images
        bs, ch, w, h = real_img.size()
        fake_imgs = [real_img, ]
        # generate fake images
        for target in range(opt.source + 1, opt.age_group):
            output = self.generator(real_img, torch.ones(bs) * opt.source, torch.ones(bs) * target)
            fake_imgs.append(output)
        fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))

        fake_imgs = fake_imgs * 0.5 + 0.5
        grid_img = torchvision.utils.make_grid(fake_imgs.clamp(0., 1.), nrow=opt.age_group - opt.source)
        self.logger.save_image(grid_img, n_iter, 'test')

    def fit(self):
        opt = self.opt
        for n_iter in range(opt.restore_iter + 1, opt.max_iter + 1):
            inputs = self.prefetcher.next()
            self.train(inputs, n_iter)
            if n_iter % opt.save_iter == 0 or n_iter == opt.max_iter:
                # self.logger.checkpoints(n_iter)
                self.generate_images(n_iter)

    def age_criterion(self, input, gt_age):
        opt = self.opt
        age_logit, group_logit = self.age_classifier(input)
        return F.mse_loss(get_dex_age(age_logit), gt_age) + \
               F.cross_entropy(group_logit, age2group(gt_age, opt.age_group).long())

    def extract_vgg_face(self, inputs):
        inputs = normalize((F.hardtanh(inputs) * 0.5 + 0.5) * 255,
                           [129.1863, 104.7624, 93.5940],
                           [1.0, 1.0, 1.0])
        return self.vgg_face(inputs)

    def train(self, inputs, n_iter):
        opt = self.opt
        source_img, true_img, source_label, target_label, true_label, true_age, mean_age = inputs
        self.generator.train()
        self.discriminator.train()

        if opt.image_size < opt.pretrained_image_size:
            source_img_small = F.interpolate(source_img, opt.image_size)
            true_img_small = F.interpolate(true_img, opt.image_size)
        else:
            source_img_small = source_img
            true_img_small = true_img
        g_source = self.generator(source_img_small, source_label, target_label)
        if opt.image_size < opt.pretrained_image_size:
            g_source_pretrained = F.interpolate(g_source, opt.pretrained_image_size)
        else:
            g_source_pretrained = g_source

        ########Train D###########
        self.d_optim.zero_grad()
        d1_logit = self.discriminator(true_img_small, true_label)
        # d2_logit = self.discriminator(true_img, source_label)
        d3_logit = self.discriminator(g_source.detach(), target_label)

        # d_loss = 0.5 * (ls_gan(d1_logit, 1.) + ls_gan(d2_logit, 0.) + ls_gan(d3_logit, 0.))
        d_loss = 0.5 * (ls_gan(d1_logit, 1.) + ls_gan(d3_logit, 0.))

        with amp.scale_loss(d_loss, self.d_optim) as scaled_loss:
            scaled_loss.backward()
        self.d_optim.step()

        ########Train G###########
        self.g_optim.zero_grad()
        ################################GAN_LOSS##############################
        gan_logit = self.discriminator(g_source, target_label)
        # g_loss = 0.5 * ls_gan(gan_logit, 1.)
        g_loss = ls_gan(gan_logit, 1.)

        ################################Age_Loss##############################
        age_loss = self.age_criterion(g_source_pretrained, mean_age)

        ################################L1_loss##############################
        l1_loss = F.l1_loss(g_source_pretrained, source_img)

        ################################SSIM_loss##############################
        ssim_loss = compute_ssim_loss(g_source_pretrained, source_img, window_size=10)

        ################################ID_loss##############################
        id_loss = F.mse_loss(self.extract_vgg_face(g_source_pretrained), self.extract_vgg_face(source_img))

        pix_loss_weight = max(opt.pix_loss_weight,
                              opt.pix_loss_weight * (opt.decay_pix_factor ** (n_iter // opt.decay_pix_n)))

        total_loss = g_loss * opt.gan_loss_weight + \
                     (l1_loss * (1 - opt.alpha) + ssim_loss * opt.alpha) * pix_loss_weight + \
                     id_loss * opt.id_loss_weight + \
                     age_loss * opt.age_loss_weight

        with amp.scale_loss(total_loss, self.g_optim) as scaled_loss:
            scaled_loss.backward()
        self.g_optim.step()

        self.logger.msg([
            d1_logit, d3_logit, gan_logit, age_loss, l1_loss, ssim_loss, id_loss
        ], n_iter)

class Generator(nn.Module):
    def __init__(self, age_group, norm_layer='bn'):
        super(Generator, self).__init__()
        self.gs = nn.ModuleList()
        for _ in range(age_group - 1):
            self.gs.append(SubGenerator(norm_layer=norm_layer))
        self.age_group = age_group

    def forward(self, x, source_label: torch.Tensor, target_label: torch.Tensor):
        condition = self.pfa_encoding(source_label, target_label, self.age_group).to(x).float()
        for i in range(self.age_group - 1):
            aging_effects = self.gs[i](x)
            x = x + aging_effects * condition[:, i]
        return x

    def pfa_encoding(self, source, target, age_group):
        source, target = source.long(), target.long()
        code = torch.zeros((source.size(0), age_group - 1, 1, 1, 1)).to(source)
        for i in range(source.size(0)):
            code[i, source[i]: target[i], ...] = 1
        return code

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            get_norm_layer(norm_layer, nn.Conv2d(channels, channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(channels, channels, 3, 1, 1)),
        )

    def forward(self, x):
        residual = x
        x = self.main(x)
        return F.leaky_relu(residual + x, 0.2, inplace=True)

class SubGenerator(nn.Module):

    def __init__(self, in_channels=3, repeat_num=4, norm_layer='bn'):
        super(SubGenerator, self).__init__()
        layers = [
            get_norm_layer(norm_layer, nn.Conv2d(in_channels, 32, 9, 1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(repeat_num):
            layers.append(ResidualBlock(128, norm_layer))
        layers.extend([
            get_norm_layer(norm_layer, nn.ConvTranspose2d(128, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            get_norm_layer(norm_layer, nn.ConvTranspose2d(64, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 9, 1, 4),
        ])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, age_group, conv_dim=64, repeat_num=3, norm_layer='bn'):
        super(PatchDiscriminator, self).__init__()

        use_bias = True
        self.age_group = age_group

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        sequence = []
        nf_mult = 1

        for n in range(1, repeat_num):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                get_norm_layer(norm_layer, nn.Conv2d(conv_dim * nf_mult_prev + (self.age_group if n == 1 else 0),
                                                     conv_dim * nf_mult, kernel_size=4, stride=2, padding=1,
                                                     bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** repeat_num, 8)

        sequence += [
            get_norm_layer(norm_layer,
                           nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=4,
                                     stride=1, padding=1,
                                     bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim * nf_mult, 1, kernel_size=4, stride=1,
                      padding=1)  # output 1 channel prediction map
        ]
        self.main = nn.Sequential(*sequence)
    
    def forward(self, inputs, condition):
        x = F.leaky_relu(self.conv1(inputs), 0.2, inplace=True)
        condition = self.group2feature(condition, feature_size=x.size(2), age_group=self.age_group).to(x)
        return self.main(torch.cat([x, condition], dim=1))
    
    def group2feature(self, group, age_group, feature_size):
        onehot = self.group2onehot(group, age_group)
        return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)

    def group2onehot(self, groups, age_group):
        code = torch.eye(age_group)[groups.squeeze()]
        if len(code.size()) > 1:
            return code
        return code.unsqueeze(0)
    
class AuxiliaryAgeClassifier(nn.Module):
    def __init__(self, age_group, conv_dim=64, repeat_num=3):
        super(AuxiliaryAgeClassifier, self).__init__()
        age_classifier = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
        ]
        nf_mult = 1
        for n in range(1, repeat_num + 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            age_classifier += [
                nn.Conv2d(conv_dim * nf_mult_prev,
                          conv_dim * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(conv_dim * nf_mult),
                nn.ReLU(True),
            ]
        age_classifier += [
            nn.Flatten(),
            nn.Linear(conv_dim * nf_mult * 16, 101),
        ]
        self.age_classifier = nn.Sequential(*age_classifier)
        self.group_classifier = nn.Linear(101, age_group)

    def forward(self, inputs):
        age_logit = self.age_classifier(F.hardtanh(inputs))
        group_logit = self.group_classifier(age_logit)
        return age_logit, group_logit
