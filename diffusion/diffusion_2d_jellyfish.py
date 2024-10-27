import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import math
import pdb
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from PIL import Image
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import datetime
import logging
from torch.utils.data import DataLoader
from dataset.data_2d import Jellyfish

from torch.optim import Adam
from torch.autograd import grad
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_noise_w', 'pred_x_start'])
# helpers functions

def exists(x):
    return x is not None

def get_device():
    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class ForceUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = None),
                block_klass(dim_in, dim_in, time_emb_dim = None),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = None)

        self.final = nn.Linear(512, out_dim)

    def forward(self, x, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        # r = x.clone()
        # h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            # h.append(x)

            x = block2(x)
            x = attn(x)
            # h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = self.final(x)

        return x

    

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        frames=20,
        cond_steps=0,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        backward_steps=5,
        backward_lr=0.01, # used in universal-backward sampling
        standard_fixed_ratio=0.01,
        forward_fixed_ratio = 0.01, # used in standard sampling
        coeff_ratio_J = 0.3, # used in gradient of objective J in standard-alpha sampling
        coeff_ratio_w = 0.3, # used in predicted noise of p(w) in standard-alpha sampling
        only_vis_pressure=False,
        eval_2ddpm = False,
        w_prob_exp = 1.0,
        use_guidance_in_model_predictions = False,
        return_all_timesteps = True,
        device
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not (type(self) == GaussianDiffusion)

        # assert not model.random_or_learned_sinusoidal_cond

        if eval_2ddpm:
            self.model_states, self.model_thetas = model
            self.channels = self.model_states.channels
            self.self_condition = self.model_states.self_condition
        else:
            self.model = model
            self.channels = self.model.channels
            self.self_condition = self.model.self_condition
        self.frames = frames
        self.cond_steps = cond_steps
        self.image_size = image_size
        self.objective = objective
        self.backward_steps = backward_steps
        self.backward_lr = backward_lr
        self.standard_fixed_ratio = standard_fixed_ratio
        self.forward_fixed_ratio = forward_fixed_ratio
        self.coeff_ratio_J = coeff_ratio_J # used in gradien of objective J in standard-alpha sampling
        self.coeff_ratio_w = coeff_ratio_w # used in predicted noise of p(w) standard-alpha sampling
        self.only_vis_pressure = only_vis_pressure
        self.w_prob_exp = w_prob_exp
        self.use_guidance_in_model_predictions = use_guidance_in_model_predictions
        self.return_all_timesteps = return_all_timesteps

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', "pred_optimal_design"}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_optimal_design':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, bd_0_expand, state_cond, design_fn, design_guidance, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, use_guidance_in_model_predictions=False):
        pred_noise_joint = self.model_states(x, t, x_self_cond)
        x_w = torch.cat([state_cond, x[:,:,-4:,:,:]], dim=2) # x_w： [batch, 20, 5, 64, 64] if self.only_vis_pressure esle [batch, 20, 7, 64, 64]
        pred_noise_w = self.model_thetas(x_w, t, x_self_cond) # pred_noise_w: [batch, 20, 1, 64, 64]

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        if self.only_vis_pressure:
            x = torch.cat([x[:,:,:1,:,:], x[:,:,-1:,:,:]], dim=2) # only use pressure and theta, not boundary masks and offsets, or velocity
        else:
            x = torch.cat([x[:,:,:3,:,:], x[:,:,6:,:,:]], dim=2) # only use states and theta, not boundary masks and offsets

        if self.objective == 'pred_noise': 
            x_start = self.predict_start_from_noise(x, t, pred_noise_joint)
            if use_guidance_in_model_predictions:
                batch_size = x.shape[0]
                coeff_design_schedual_J = self.coeff_ratio_J * (self.betas).clone().flip(0)
                coeff_design_schedual_w = self.coeff_ratio_w * (self.betas).clone().flip(0)
                eta_J = extract(coeff_design_schedual_J, t, x.shape)
                eta_w = extract(coeff_design_schedual_w, t, x.shape)
                if self.only_vis_pressure:
                    pred_noise_w_pad = torch.zeros(batch_size, 20, 2, 64, 64, device=x.device) # pred_noise_w: [batch, 20, 1+1, 64, 64], contains pressure and thetas, no boundarys mask and offsets
                    pred_noise_w_pad[:,:,1:,:,:] = pred_noise_w
                else:
                    pred_noise_w_pad = torch.zeros(batch_size, 20, 4, 64, 64, device=x.device) # pred_noise_w: [batch, 20, 3+1, 64, 64], contains states and thetas, no boundarys mask and offsets
                    pred_noise_w_pad[:,:,3:,:,:] = pred_noise_w
                pred_noise_w = pred_noise_w_pad
                with torch.enable_grad():
                    x_clone = x_start.clone().detach().requires_grad_()
                    g = design_fn(x_clone, bd_0_expand)
                if design_guidance == "standard":
                    grad_final = self.standard_fixed_ratio * g + (self.w_prob_exp - 1) * pred_noise_w
                elif design_guidance == "standard-alpha":
                    # grad_final = eta * g + (self.w_prob_exp - 1) * pred_noise_w
                    grad_final = eta_J * g - eta_w * pred_noise_w # eta_w acts as 1 - \beta
                else:
                    raise
                pred_noise_joint = pred_noise_joint + grad_final
            
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise_joint, pred_noise_w, x_start)
    

    def p_mean_variance(self, x, t, bd_0_expand, state_cond, design_fn, design_guidance, x_self_cond = None, clip_denoised = True, use_guidance_in_model_predictions=False):
        preds = self.model_predictions(x, t, bd_0_expand, state_cond, design_fn, design_guidance,  x_self_cond, use_guidance_in_model_predictions)
        x_start = preds.pred_x_start
        pred_noise_w = preds.pred_noise_w
        if clip_denoised:
            x_start.clamp_(-1., 1.) 
        if self.only_vis_pressure:
            x = torch.cat([x[:,:,:1,:,:], x[:,:,-1:,:,:]], dim=2) # only use pressure and theta, not boundary masks and offsets, or velocity
        else:
            x = torch.cat([x[:,:,:3,:,:], x[:,:,6:,:,:]], dim=2) # only use states and theta, not boundary masks and offsets
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)

        return model_mean, posterior_variance, posterior_log_variance, x_start, pred_noise_w

    def sample_noise(self, shape, device):
        return torch.randn(shape, device = device)

    @torch.no_grad()
    def p_sample(self, x, t: int, bd_0_expand, state_cond, x_self_cond = None, clip_denoised = True, design_fn = None, design_guidance = "standard"):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"
        """
        b, *_, device = *x.shape, x.device # x shape: [batch, 20, 3, 64, 64]
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        coeff_design_schedual_J = self.coeff_ratio_J * (self.betas).clone().flip(0)
        coeff_design_schedual_w = self.coeff_ratio_w * (self.betas).clone().flip(0)
        eta_J = extract(coeff_design_schedual_J, batched_times, x.shape)
        eta_w = extract(coeff_design_schedual_w, batched_times, x.shape)
        p_mean_variance = self.p_mean_variance(x = x, t = batched_times, bd_0_expand=bd_0_expand, state_cond=state_cond, design_fn=design_fn, design_guidance=design_guidance, x_self_cond = x_self_cond, clip_denoised = clip_denoised, use_guidance_in_model_predictions=self.use_guidance_in_model_predictions)
        if "recurrence" not in design_guidance:
            model_mean, _, model_log_variance, x_start, pred_noise_w = p_mean_variance
            noise = self.sample_noise(model_mean.shape, device) if t > 0 else 0
            pred = model_mean + (0.5 * model_log_variance).exp() * noise
            if not self.use_guidance_in_model_predictions and design_fn is not None:
                if design_guidance.startswith("standard"):
                    with torch.enable_grad():
                        x_clone = x_start.clone().detach().requires_grad_() 
                        g = design_fn(x_clone, bd_0_expand)
                    if design_guidance == "standard":
                        grad_final = self.standard_fixed_ratio * g  - self.standard_fixed_ratio * pred_noise_w
                    elif design_guidance == "standard-alpha":
                        grad_final = eta_J * g - eta_w * pred_noise_w # eta_w acts as 1 - \beta
                    else:
                        raise
                
                pred = pred - grad_final

            return pred, x_start
            

    def update_bd(self, bd_updater, theta_expand_start, bd_0_expand, thetas_0_frame_expand):
        theta_start = torch.mean(torch.mean(theta_expand_start, dim=4), dim=3).squeeze(2) # theta_start: [batch, 20]
        pred_bd = bd_updater(
            bd_0_expand.reshape(bd_0_expand.shape[0] * bd_0_expand.shape[1], bd_0_expand.shape[2], bd_0_expand.shape[3], bd_0_expand.shape[4]), 
            (theta_start - thetas_0_frame_expand).reshape(theta_start.shape[0] * theta_start.shape[1])
        )
        pred_bd = pred_bd.reshape(bd_0_expand.shape)
        
        return pred_bd
    
    @torch.no_grad()
    def p_sample_loop(self, shape, design_fn = None, design_guidance="standard", return_all_timesteps=None, cond=None, thetas_0=None, bd_updater=None, device=None):
        b, f, c, h, w = shape # [batch, 20, 3, 64, 64]
        batch, device = shape[0], self.betas.device
        assert cond is not None
        state_0, bd_0 = cond[0].to(device), cond[1].to(device)
        if self.only_vis_pressure:
            noise_state = self.sample_noise([b, f, 1, h, w], device) # [batch, 20, 1, 64, 64]
        else:
            noise_state = self.sample_noise([b, f, 3, h, w], device) # [batch, 20, 3, 64, 64]
        noise_bd = self.sample_noise([b, f, 3, h, w], device) # [batch, 20, 3, 64, 64]
        thetas_0 = thetas_0.to(device)

        noisy_thetas = self.sample_noise([b, f, 1, h, w], device) # [batch, 20, 1, 64, 64]
        thetas_0_expand = thetas_0.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, 1, 1, h, w)
        thetas_0_frame_expand = thetas_0.unsqueeze(1).expand(-1, self.frames) # thetas_0_frame_expand: [batch, 20]
        bd_0_expand = bd_0.unsqueeze(1).expand(-1, self.frames, -1, -1, -1) 
        bd_updater.to(device)
        bd_updater.eval()
        if self.cond_steps > 0: # use conditional model
            noise_state[:, :self.cond_steps] = state_0.unsqueeze(1)
            noise_bd[:, :self.cond_steps] = bd_0.unsqueeze(1)
            noisy_thetas[:, :self.cond_steps] = thetas_0_expand
            noisy_thetas[:, -self.cond_steps:] = thetas_0_expand
        state_cond = state_0.unsqueeze(1)
        state_cond = state_cond.expand(-1, f, -1, -1, -1)
        
        x = torch.cat([noise_state, noise_bd, noisy_thetas], dim=2) # [batch, 20, 7, 64, 64]
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_states_start if self.self_condition else None
            pred, x_start = self.p_sample(x, t, bd_0_expand, state_cond, self_cond, design_fn=design_fn, design_guidance=design_guidance)
            
            if self.only_vis_pressure:
                pred_states, pred_theta_expand = pred[:,:,:1,:,:], pred[:,:,-1:,:,:]
                # state_start, theta_expand_start = x_start[:,:,:1,:,:], x_start[:,:,-1:,:,:]
            else:
                pred_states, pred_theta_expand = pred[:,:,:3,:,:], pred[:,:,3:,:,:]
                # state_start, theta_expand_start = x_start[:,:,:3,:,:], x_start[:,:,3:,:,:]
                    
            # pred_bd_start = self.update_bd(bd_updater, theta_expand_start, bd_0_expand, thetas_0_frame_expand)
            pred_bd = self.update_bd(bd_updater, pred_theta_expand, bd_0_expand, thetas_0_frame_expand)
            if self.cond_steps > 0: # use conditional model
                pred_states[:, :self.cond_steps] = state_0.unsqueeze(1) # pred_states: {u_0, u_[1,T]^(k-1)}
                pred_bd[:, :self.cond_steps] = bd_0.unsqueeze(1)
                pred_bd[:, -self.cond_steps:] = bd_0.unsqueeze(1)
                pred_theta_expand[:, :self.cond_steps] = thetas_0_expand
                pred_theta_expand[:, -self.cond_steps:] = thetas_0_expand # pred_theta_expand: {w_0, w_[1,T-1]^(k-1), w_T}
            else: # use unconditional model
                t_tensor = torch.LongTensor([t]).to(device).repeat(batch)
                state_0_t = self.q_sample(x_start=state_0, t=t_tensor).unsqueeze(1) # repaint: noisy condition
                bd_0_t = self.q_sample(x_start=bd_0, t=t_tensor).unsqueeze(1) # repaint: noisy condition
                thetas_0_expand_t = self.q_sample(x_start=thetas_0_expand, t=t_tensor) # repaint: noisy condition
                pred_states[:, :1] = state_0_t
                pred_bd[:, :1] = bd_0_t
                pred_theta_expand[:, :1] = thetas_0_expand_t
                pred_theta_expand[:, -1:] = thetas_0_expand_t
            
            pred_theta = torch.mean(torch.mean(pred_theta_expand, dim=4), dim=3).squeeze(2)
            x = torch.cat([pred_states, pred_bd, pred_theta_expand], dim=2)
            final_result = [pred_states, pred_theta]

        return final_result

    @torch.no_grad()
    def ddim_sample(self, #state_0, bd_0, shape, cond, return_all_timesteps = False
                   shape, design_fn = None, design_guidance="standard", return_all_timesteps=None, cond=None, thetas_0=None, bd_updater=None, device=None):
        # batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        eta, objective = self.ddim_sampling_eta, self.objective
        
        b, f, c, h, w = shape # [batch, 20, 3, 64, 64]
        batch, device = shape[0], self.betas.device

        times = torch.linspace(-1, self.num_timesteps - 1, steps = self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when self.sampling_timesteps == self.num_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # img = torch.randn((shape[0], shape[1] - cond.size(1), shape[2], shape[3]), device = device)        
        
        assert cond is not None
        state_0, bd_0 = cond[0].to(device), cond[1].to(device)
        if self.only_vis_pressure:
            noise_state = self.sample_noise([b, f, 1, h, w], device) # [batch, 20, 1, 64, 64]
        else:
            noise_state = self.sample_noise([b, f, 3, h, w], device) # [batch, 20, 3, 64, 64]
        noise_bd = self.sample_noise([b, f, 3, h, w], device) # [batch, 20, 3, 64, 64]
        thetas_0 = thetas_0.to(device)

        noisy_thetas = self.sample_noise([b, f, 1, h, w], device) # [batch, 20, 1, 64, 64]
        thetas_0_expand = thetas_0.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, 1, 1, h, w)
        thetas_0_frame_expand = thetas_0.unsqueeze(1).expand(-1, self.frames) # thetas_0_frame_expand: [batch, 20]
        bd_0_expand = bd_0.unsqueeze(1).expand(-1, self.frames, -1, -1, -1) 
        bd_updater.to(device)
        bd_updater.eval()
        if self.cond_steps > 0: # use conditional model
            noise_state[:, :self.cond_steps] = state_0.unsqueeze(1)
            noise_bd[:, :self.cond_steps] = bd_0.unsqueeze(1)
            noisy_thetas[:, :self.cond_steps] = thetas_0_expand
            noisy_thetas[:, -self.cond_steps:] = thetas_0_expand
        state_cond = state_0.unsqueeze(1)
        state_cond = state_cond.expand(-1, f, -1, -1, -1)
        
        x = torch.cat([noise_state, noise_bd, noisy_thetas], dim=2) # [batch, 20, 7, 64, 64]
        

        imgs = [x]
        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            preds = self.model_predictions(
                x, time_cond, bd_0_expand, state_cond, design_fn, design_guidance, use_guidance_in_model_predictions=True
                # img, cond, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True
            )
            x_start = preds.pred_x_start
            pred_noise_w = preds.pred_noise_w
            pred_noise = preds.pred_noise

            if time_next < 0:
                x = x_start
                imgs.append(x)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            state_theta_noise = torch.randn_like(x_start) # [batch, 20, 4, 64, 64] if not self.only_vis_pressure else [batch, 20, 2, 64, 64]
            pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * state_theta_noise
            
            if self.only_vis_pressure:
                pred_states, pred_theta_expand = pred[:,:,:1,:,:], pred[:,:,-1:,:,:]
            else:
                pred_states, pred_theta_expand = pred[:,:,:3,:,:], pred[:,:,3:,:,:]
                    
            pred_bd = self.update_bd(bd_updater, pred_theta_expand, bd_0_expand, thetas_0_frame_expand)
            if self.cond_steps > 0: # use conditional model
                pred_states[:, :self.cond_steps] = state_0.unsqueeze(1) # pred_states: {u_0, u_[1,T]^(k-1)}
                pred_bd[:, :self.cond_steps] = bd_0.unsqueeze(1)
                pred_bd[:, -self.cond_steps:] = bd_0.unsqueeze(1)
                pred_theta_expand[:, :self.cond_steps] = thetas_0_expand
                pred_theta_expand[:, -self.cond_steps:] = thetas_0_expand # pred_theta_expand: {w_0, w_[1,T-1]^(k-1), w_T}
                
            pred_theta = torch.mean(torch.mean(pred_theta_expand, dim=4), dim=3).squeeze(2)
            x = torch.cat([pred_states, pred_bd, pred_theta_expand], dim=2)
            final_result = [pred_states, pred_theta]
            
            imgs.append(final_result)

        ret = final_result if not return_all_timesteps else torch.stack(imgs, dim = 1)

        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, design_fn = None, design_guidance="standard", return_all_timesteps = False, cond=None, thetas_0=None, bd_updater=None, device = None):
        image_size, channels, frames = self.image_size, self.channels // 2, self.frames
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        batch_size = cond[0].shape[0]
        return sample_fn((batch_size, frames, channels, image_size, image_size), design_fn, design_guidance, return_all_timesteps = return_all_timesteps, cond=cond, thetas_0=thetas_0, bd_updater=bd_updater, device = device)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )        


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, state_start, bd_start, thetas_start, t, model_type, noise = None):
        if model_type == "states":
            b, f, c, h, w = state_start.shape # [batch, 20, 3, 64, 64]
            noise_state = default(noise, lambda: torch.randn_like(state_start))
            thetas_start = thetas_start.unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(-1, -1, 1, h, w)
            # noisy sample
            state = self.q_sample(x_start = state_start, t = t, noise = noise_state)
            if self.cond_steps > 0:
                # condition on start time step of state, and start and end time step of bd/theta
                state[:, :self.cond_steps] = state_start[:, :self.cond_steps]
                noise_state[:, :self.cond_steps] = torch.zeros_like(noise_state[:, :self.cond_steps])

            # if doing self-conditioning, 50% of the time, predict x_start from current set of times
            # and condition with unet with that
            # this technique will slow down training by 25%, but seems to lower FID significantly
            x_self_cond = None
            if self.self_condition and random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model_predictions(x, t).pred_x_start
                    x_self_cond.detach_()

            state.requires_grad_(requires_grad=True)
            state_bd_thetas = torch.cat([state, bd_start, thetas_start], dim=2) # [batch, 20, 7, 64, 64] use clean bd_start and thetas_start as conditions
            state_out = self.model(state_bd_thetas, t, x_self_cond)
            state_energy = -torch.sum(torch.square(state_out))
            state_grad= grad(state_energy, state, retain_graph=True, create_graph=True)[0]
            if self.objective == 'pred_noise':
                # loss = self.loss_fn(state_out, noise_state, reduction = 'mean')
                loss = self.loss_fn(state_grad, noise_state, reduction = 'mean')
                state.requires_grad_(requires_grad=False) 
            else:
                raise ValueError(f'unknown objective {self.objective}')
        
        elif model_type == "thetas":
            b, f, c, h, w = state_start.shape # [batch, 20, 3, 64, 64]
            noise_bd = torch.randn_like(bd_start) # [batch, 20, 3, 64, 64]
            thetas_start = thetas_start.unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(-1, -1, 1, h, w)
            noise_thetas = torch.randn_like(thetas_start) # [batch, 20, 1, 64, 64]

            # noisy sample
            bd = self.q_sample(x_start = bd_start, t = t, noise = noise_bd)
            theta = self.q_sample(x_start = thetas_start, t = t, noise = noise_thetas)
            if self.cond_steps > 0:
                # condition on start time step of state, and start and end time step of bd/theta
                state_cond = state_start[:, :self.cond_steps]
                state_cond = state_cond.expand(-1, f, -1, -1, -1) # output state_cond size: [batch, 20, 3, 64, 64]
                bd[:, :self.cond_steps] = bd_start[:, :self.cond_steps]
                noise_thetas[:, :self.cond_steps] = torch.zeros_like(noise_thetas[:, :self.cond_steps])
                noise_thetas[:, -self.cond_steps:] = torch.zeros_like(noise_thetas[:, -self.cond_steps:])

            # if doing self-conditioning, 50% of the time, predict x_start from current set of times
            # and condition with unet with that
            # this technique will slow down training by 25%, but seems to lower FID significantly
            x_self_cond = None
            if self.self_condition and random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model_predictions(x, t).pred_x_start
                    x_self_cond.detach_()

            state_bd_thetas = torch.cat([state_cond, bd, theta], dim=2) # [batch, 20, 7, 64, 64]
            thetas_out = self.model(state_bd_thetas, t, x_self_cond)
            if self.objective == 'pred_noise':
                loss = self.loss_fn(thetas_out, noise_thetas, reduction = 'mean')
            else:
                raise ValueError(f'unknown objective {self.objective}')
        
        else:
            raise ValueError(f'unknown model type {model_type}')


        return loss.mean()

    def forward(self, state, bd, thetas, model_type, *args, **kwargs):
        #pdb.set_trace()
        b, f, c, h, w, device, img_size, = *state.shape, state.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(state, bd, thetas, t, model_type, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        dataset_path,
        frames,
        traj_len,
        ts,
        log_path,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_path = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        is_testdata=False,
        is_schedule = True,
        resume = False,
        resume_step = 0,
        only_vis_pressure = False,
        model_type = None
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp
        self.dataset = dataset
        self.frames = frames
        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.model_type = model_type

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        if dataset == "jellyfish":
            self.ds = Jellyfish(
                dataset="jellyfish", 
                dataset_path=dataset_path,
                time_steps=traj_len, 
                steps=frames, 
                time_interval=ts, 
                is_train=True, 
                is_testdata=is_testdata,
                only_vis_pressure=only_vis_pressure
            )
        else:
            assert False

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 16)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.resume = resume
        self.resume_step = resume_step
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        if is_schedule == True:
            self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=[50000, 150000, 300000], gamma=0.1)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok = True)
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        # if self.resume:
        #     self.load(self.resume_step // self.save_and_sample_every)
        #     self.step = self.resume_step
        
        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_path / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_path / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        print("model loaded: ", str(self.results_path / f'model-{milestone}.pt'))
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = os.path.join(self.log_path, "{}.log".format(current_time))
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    state, bd, thetas, sim_id, time_id = data
                    if self.dataset == "jellyfish":
                        bd_pad = torch.zeros(bd.shape[0], self.frames, 3, self.image_size, self.image_size).to(bd.device) # [batch, 20, 3, 64, 64]
                        bd_pad[:, :, :, 1:-1, 1:-1] = bd

                    with self.accelerator.autocast():
                        loss = self.model(state, bd_pad, thetas, self.model_type)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.step != 0 and self.step % 10 == 0:
                    pbar.set_description(f'loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                    logging.info(f'step: {self.step}, loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    # if True:
                        self.ema.ema_model.eval()

                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
