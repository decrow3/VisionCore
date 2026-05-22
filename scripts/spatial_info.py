#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye

def embed_time_lags(movie, n_lags=32):
    """
    Embed time lags into a movie tensor.
    
    Input: movie (T, H, W) or (T, 1, H, W)
    Output: (T - n_lags + 1, 1, n_lags, H, W)
    """
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)  # (T, 1, H, W)
    
    T, C, H, W = movie.shape
    # Create lagged indices: for each output frame t, we want frames [t, t+1, ..., t+n_lags-1]
    # But stim uses negative lags (past frames), so we want [t-n_lags+1, ..., t]
    out_frames = T - n_lags + 1
    
    # Build lagged tensor
    lagged = torch.zeros(out_frames, C, n_lags, H, W, dtype=movie.dtype, device=movie.device)
    for lag in range(n_lags):
        # lag 0 = current frame, lag 1 = 1 frame ago, etc.
        lagged[:, :, lag] = movie[n_lags - 1 - lag : T - lag]
    
    return lagged

def spatial_ssi_population(y, dt=1.0, eps=1e-8, log_base=2.0, spike_weighted=True):
    """
    Spatial single-spike information from a rate map.
    y: rates, shape [T, N, H, W]. Must be >= 0.
    Returns: ispikepop (bits/spike), iratepop (bits/sec), I_tn (T, N)
    """
    T, N, H, W = y.shape
    # T = time, N = units, H = height, W = width
    P = H * W # number of spatial bins
    r = y.reshape(T, N, P)
    rbar = r.mean(dim=2) # mean across space
    g = r / (rbar[..., None] + eps) # r/rbar
    logg = torch.log2(g + eps) if log_base == 2.0 else torch.log(g + eps) # log(r/rbar)
    
    I_tn = (g * logg).mean(dim=2) # expectation over space 
    
    # rescale to get bits per spike and per bin
    spikes_tn = rbar * dt                    # (T, N) expected spikes in bin

    # bits/sec per (time, neuron)
    bits_per_sec_tn = rbar * I_tn            # (T, N)

    if spike_weighted:
        # population bits/spike at each time t: sum_n spikes*I / sum_n spikes
        bits_t = (spikes_tn * I_tn).sum(dim=1)                 # (T,)
        spikes_t = spikes_tn.sum(dim=1)                        # (T,)
        ispike_t = bits_t / (spikes_t + eps)                   # (T,)
    else:
        # equal-weight neuron average
        ispike_t = I_tn.mean(dim=1)                            # (T,)

    # population bits/sec at each time t: sum_n rbar*I  (i.e., total bits/sec across neurons)
    irate_t = bits_per_sec_tn.sum(dim=1) 
    return ispike_t, irate_t, I_tn

class PopulationReadout(nn.Module):
    def __init__(self, feat_weights, biases, space_weights):
        super().__init__()
        self.features = nn.Conv2d(feat_weights.shape[1], feat_weights.shape[0], kernel_size=1, bias=False)
        self.features.weight = nn.Parameter(feat_weights, requires_grad=False)
        self.bias = nn.Parameter(biases, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None, :, :], requires_grad=False)
        self.n_units = space_weights.shape[0]
    
    def forward(self, x):
        feat = self.features(x)
        
        space = F.conv2d(feat, self.space_weights, groups=self.n_units, padding="valid")
        out = space + self.bias[None, :, None, None]

        return out
    
def get_spatial_readout(model, outputs):
    """
    Combine readouts from multiple datasets into a single readout.
    """
    sessions = [outputs[i]['sess'] for i in range(len(outputs))]

    model_dataset_idx = [i for i, name in enumerate(model.names) if name in sessions]
    cids2use = [np.where(outputs[sessions.index(model.names[i])]['ccnorm']['ccnorm']>.5)[0] for i in model_dataset_idx]

    # make single readout

    feat_weights = []
    biases = []
    space_weights = []
    for i in range(len(model_dataset_idx)):
        model_readout_idx = model_dataset_idx[i]

        readout = model.model.readouts[model_readout_idx]
        feat_weight = readout.features.weight.detach().cpu()
        bias = readout.bias.detach().cpu()
        space_weight = readout.compute_gaussian_mask(14, 14, model.device).detach().cpu()

        feat_weight = feat_weight[cids2use[i]]
        bias = bias[cids2use[i]]
        space_weight = space_weight[cids2use[i]]

        feat_weights.append(feat_weight)
        biases.append(bias)
        space_weights.append(space_weight)

    feat_weights = torch.cat(feat_weights, dim=0)
    biases = torch.cat(biases, dim=0)
    space_weights = torch.cat(space_weights, dim=0)

    # print(feat_weights.shape, biases.shape, space_weights.shape)
    readout = PopulationReadout(feat_weights, biases, space_weights)
    return readout

def compute_rate_map(model, readout, stim):
    """Compute rate map from stimulus and behavior."""
    x = model.model.core_forward(stim, None)
    y_batch = readout(x[:,:,-1])

    return model.model.activation(y_batch)

def compute_rate_map_batched(model, readout, stim, batch_size=32):
    """Compute rate map from stimulus and behavior in batches."""
    device = next(model.model.parameters()).device
    T = stim.shape[0]
    y_chunks = []
    
    model.model.eval()
    readout.eval()

    with torch.no_grad():
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)

            # Move batch to GPU
            x = stim[t_start:t_end].to(device)

            y_batch = compute_rate_map(model,readout, x)

            # Move to CPU immediately
            y_chunks.append(y_batch.cpu())
            del y_batch
            torch.cuda.empty_cache()

    return torch.cat(y_chunks, dim=0)

def make_movie(y, save_path='', n_units_to_show=100, fps=15):
    from torchvision.utils import make_grid
    from matplotlib.animation import FFMpegWriter

    # if n_units_to_show is list or array, use it as index
    if isinstance(n_units_to_show, (list, np.ndarray)):
        units_to_show = np.array(n_units_to_show)
        n_units_to_show = len(units_to_show)
    else:
        units_to_show = np.arange(n_units_to_show)
    
    y_subset = y[:, units_to_show].detach().cpu()  # (T, N, H, W)
    
    # normalize each unit to [0,1]
    # miny = torch.tensor(np.array([y_subset[:,i].min() for i in range(n_units_to_show)]))
    # maxy = torch.tensor(np.array([y_subset[:,i].max() for i in range(n_units_to_show)]))
    # y_subset = (y_subset - miny[None,:,None,None]) / (maxy[None,:,None,None] - miny[None,:,None,None] + 1e-8)

    # normalize each unit to 0 mean, 1 std (better)
    std = y_subset.std(dim=(0, 2, 3), keepdim=True)
    mu = y_subset.mean(dim=(0, 2, 3), keepdim=True)
    y_subset = (y_subset - mu) / (std + 1e-8)
    
    T = y_subset.shape[0]
    nrow = int(np.ceil(np.sqrt(n_units_to_show)))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')

    #writer = FFMpegWriter(fps=15, codec='libx264', bitrate=8000)
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='VisionCore'), bitrate=1800)

    save_path = f'../figures/{save_path}.mp4'
    with writer.saving(fig, save_path, dpi=100):
        for t in range(T):
            ax.clear()
            # make_grid expects (N, C, H, W), add channel dim
            frames = y_subset[t].unsqueeze(1)  # (N, 1, H, W)
            grid = make_grid(frames, nrow=nrow, normalize=False, padding=1, pad_value=0.0)
            ax.imshow(grid[0].numpy(), cmap='gray', vmin=-6, vmax=6)
            ax.set_title(f'Spatial Activations - Frame {t}/{T}', fontsize=14)
            ax.axis('off')
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved spatial activations movie to {save_path}")

# Reconstruct stimulus
def make_stimulus_stack(type='fixrsvp', frame=None, frames_per_im=6, num_frames=500):
    """
    Make a stimulus stack for a given type. 
    Input:
        type: 
            'fixrsvp': fixrsvp images
            'face': marmoset face images
            'nat': natural images
        frame: frame number to use for all time points (None flashes frames at framerate specified by frames_per_im)
        frames_per_im: number of frames to show each image for (if frame is None)
            This specifies the frame rate (if frames_per_im = 1, then the update is the screen rate (e.g., 120Hz), if 2, then 60Hz)
        num_frames: number of frames to generate (if frame is None)
    """

    if type == 'fixrsvp':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='im')
    elif type == 'face':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='face')
    elif type == 'nat':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='nat')

    if frame is not None:
        full_stack = full_stack[[frame]].repeat(num_frames, axis=0)

    return full_stack

def make_counterfactual_stim(full_stack, eyepos,
                            ppd = 37.50476617,
                            scale_factor = 1.0,
                            n_lags = 32,
                            out_size = (101, 101)):
    '''
    Reconstruct stimulus from eye positions.
    
    Input:
        eyepos: [T, 2] eye positions in degrees
        type: 'fixrsvp', 'face', 'nat'
        frame: frame number to use for all time points (None flashes frames at framerate specified by frames_per_im)
        frames_per_im: number of frames to show each image for (if frame is None)
        ppd: pixels per degree
        scale_factor: scale factor for stimulus (1.0 is no scaling)
        n_lags: number of time lags to use
        out_size: (H, W) size of output stimulus
    '''

    #eye_norm = eye_deg_to_norm(torch.fliplr(eyepos), ppd, full_stack.shape[1:3])
    #removing the flip, since shift_movie_with_eye expects (x,y) in that order, and eye_deg_to_norm also expects (x,y) in that order.
    eye_norm = eye_deg_to_norm(eyepos, ppd, full_stack.shape[1:3])

    eye_movie = shift_movie_with_eye(
        torch.from_numpy(full_stack[:eyepos.shape[0] + n_lags]).float(),
        torch.cat([eye_norm[:n_lags], eye_norm], dim=0),  # pad beginning
        out_size=out_size,
        center=(0.0, 0.0),
        scale_factor=scale_factor,
        mode="bilinear"
    )

    # Embed time lags to match stim shape
    eye_stim = embed_time_lags(eye_movie, n_lags=n_lags)

    return eye_stim

def make_integrated_counterfactual_stim(full_stack, eyepos, ppd=37.50476617, n_lags=32, sub_frames=10):
    """
    Reconstruct stimulus with sub-frame temporal integration (simulating retinal motion blur).
    
    Args:
        full_stack: [T_stim, C, H, W] The base images
        eyepos: [T, 2] eye positions in degrees at the native model framerate (e.g., 100Hz)
        ppd: pixels per degree
        n_lags: number of time lags to use
        sub_frames: The upsampling factor for physical integration (e.g., 10x = 1000Hz internal rendering)
    """
    device = eyepos.device
    T_native = eyepos.shape[0]
    
    # 1. Upsample the eye trace linearly
    # eyepos is [T, 2]. We transpose to [2, T] for 1D interpolation
    eyepos_T = eyepos.T.unsqueeze(0) # Shape: [1, 2, T]
    
    # Interpolate to high resolution
    high_res_T = T_native * sub_frames
    eyepos_high_res = F.interpolate(eyepos_T, size=high_res_T, mode='linear', align_corners=False)
    eyepos_high_res = eyepos_high_res.squeeze(0).T # Back to [high_res_T, 2]
    
    # 2. Convert to normalized coordinates
    img_shape = full_stack.shape[2:] # (H, W)
    #eye_norm_high_res = eye_deg_to_norm(torch.fliplr(eyepos_high_res), ppd, img_shape)
    #removing the flip, since shift_movie_with_eye expects (x,y) in that order, and eye_deg_to_norm also expects (x,y) in that order.
    eye_norm_high_res = eye_deg_to_norm(eyepos_high_res, ppd, img_shape)

    # 3. Upsample the base image stack to match (nearest neighbor so we don't blend images across cuts)
    # This assumes full_stack changes slowly or we are in a single continuous trial
    full_stack_high_res = torch.repeat_interleave(full_stack, sub_frames, dim=0)
    
    # We need to grab enough padding for the lags. 
    # n_lags at native resolution = n_lags * sub_frames at high resolution
    pad_frames = n_lags * sub_frames
    
    # 4. Shift the movie at high resolution
    # Note: We shift the chunk of the movie corresponding to the eye trace length + padding
    eye_movie_high_res = shift_movie_with_eye(
        full_stack_high_res[:high_res_T + pad_frames].float(),
        torch.cat([eye_norm_high_res[:pad_frames], eye_norm_high_res], dim=0)
    )
    
    # 5. Integrate (Average) down to native frame rate
    # Reshape from [high_res_T + pad_frames, C, H, W] to [T_native + n_lags, sub_frames, C, H, W]
    integrated_movie = eye_movie_high_res.view((T_native + n_lags), sub_frames, *eye_movie_high_res.shape[1:])
    
    # Mean across the sub_frame dimension
    eye_movie_native = integrated_movie.mean(dim=1)
    
    return eye_movie_native
# %%
