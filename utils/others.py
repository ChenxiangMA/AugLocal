import logging
import math
import torch.nn as nn
def get_pool_layer(channels, dim, target_dim):
    '''Resolve average-pooling kernel size in order for flattened dim to match target_dim'''
    ks_h, ks_w = 1, 1
    dim_out_h, dim_out_w = dim, dim
    dim_in_decoder = channels*dim_out_h*dim_out_w
    while dim_in_decoder > target_dim and ks_h < dim:
        ks_h*=2
        dim_out_h = math.ceil(dim / ks_h)
        dim_in_decoder = channels*dim_out_h*dim_out_w
        if dim_in_decoder > target_dim:
            ks_w*=2
            dim_out_w = math.ceil(dim / ks_w)
            dim_in_decoder = channels*dim_out_h*dim_out_w
    if ks_h > 1 or ks_w > 1:
        pad_h = (ks_h * (dim_out_h - dim // ks_h)) // 2
        pad_w = (ks_w * (dim_out_w - dim // ks_w)) // 2
        return nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w)), dim_in_decoder
    else:
        return nn.Identity(), dim_in_decoder

def setup_logging(log_file='log.txt'):
    """Setup logging configuration"""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(51)