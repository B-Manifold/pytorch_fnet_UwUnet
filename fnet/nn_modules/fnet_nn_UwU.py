import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mult_chan = 32
        depth = 4
        starting_chan = 200
        intermediate_chan = 100
        final_chan = 17
        self.spec_conv = SubNet2Conv(starting_chan,intermediate_chan) #First Spectral Channel Convoultions
        self.spec_down_conv = SubNet2Conv(intermediate_chan, final_chan) #Second spectral Channel Convolutions
        self.spec_down_down_conv = SubNet2Conv(final_chan,1) #Third Spectral Channel Convolution to reduce to 1 channel
        self.net_recurse = _Net_recurse(n_in_channels=1, mult_chan=mult_chan, depth=depth) #Spatial U-net defined below (4 layers deep)
        self.conv_out = torch.nn.Conv2d(mult_chan,  1, kernel_size=3, padding=1) #Conv of 32 -> 1 -> 1 channels post spatial U-Net
        self.spec_convt = torch.nn.ConvTranspose2d(1, final_chan, kernel_size=3, padding=1) #Transpose convolution to return to desired channel size
        self.spec_bn = torch.nn.BatchNorm2d(final_chan) #batchnorm for transpose conv
        self.spec_relu = torch.nn.ReLU() #ReLu for transpose conv
        self.spec_final = SubNet2Conv(2*final_chan, final_chan) #Final convolution after concatenation of pre and post unet stacks
        self.spec_final_pool = SubNet2Conv(final_chan, 1) #pooling convolution to take you to single channel output

    def forward(self, x):
        x_spec_conv = self.spec_conv(x)
        x_spec_down = self.spec_down_conv(x_spec_conv)        
        x_list = list(torch.split(x_spec_down, 1, 1))        
        #x_spec_down_down = self.spec_down_down_conv(x_spec_down)
        for chan in x_list:
            sing_chan_spat = self.net_recurse(chan)
            sing_chan_spat = self.conv_out(sing_chan_spat)
            x_spec_down = torch.cat((x_spec_down,sing_chan_spat),1)
        x_spec_prepool = self.spec_final(x_spec_down)
        
        #if final_chan == 1 use this return statement
        #return torch.squeeze(self.spec_final_pool(x_spec_prepool),1) #This corrects an error in bufferedpatchdataset.py that doesn't like the mismatch of tensor dimensions

        #otherwise (i.e. final_chan > 1) use this return statement
        return x_spec_prepool

class _UwU_net(torch.nn.Module):
    def __init__(self, n_in_channels, n_out_channels, inter=0, inter_inter=0, inter_inter_inter=0, depth=4):
        """Class for spectral reduction of UwU-network
        
        Parameters:
        in_channels - (int) number of spectral channels for input
        inters - (int) intermediate spectral channels for reduction and recovery
        depth - (int) "depth" of the internal spatial U-Net
        """
        super().__init__()
        self.depth = depth
        self.inter = inter
        self.inter_inter = inter_inter
        self.inter_inter_inter = inter_inter_inter
        self.spec_conv_1down = SubNet2Conv(n_in_channels, inter)
        self.spec_conv_2down = SubNet2Conv(inter, inter_inter)
        self.spec_conv_3down = SubNet2Conv(inter_inter, inter_inter_inter)
        self.spec_conv_4down = SubNet2Conv(inter_inter_inter, 1)
        self.spatial_net = _Net_recurse(1, 32, depth)
        self.spatial_out = SubNet2Conv(32, 1)
        self.spec_convt_4up = SubNet2Convt(1, inter_inter_inter)
        self.spec_4halve = SubNet2Conv(2*inter_inter_inter, inter_inter_inter)
        self.spec_convt_3up = SubNet2Convt(inter_inter_inter, inter_inter)
        self.spec_3halve = SubNet2Conv(2*inter_inter, inter_inter)
        self.spec_convt_2up = SubNet2Convt(inter_inter, inter)
        self.spec_2halve = SubNet2Conv(2*inter, inter)
        self.spec_convt_1up = SubNet2Convt(inter, n_in_channels)
        self.spec_1halve = SubNet2Conv(2*n_in_channels, n_in_channels)
        self.spec_1down_soft = SubNet2Conv(n_in_channels, inter_inter)
        self.spec_2down_soft = SubNet2Conv(inter_inter, n_out_channels)
        
    def forward(self, x):
        x_1down = self.spec_conv_1down(x)
        x_2down = self.spec_conv_2down(x_1down)
        x_3down = self.spec_conv_3down(x_2down)
        x_4down = self.spec_conv_4down(x_3down)
        x_spatial = self.spatial_net(x_4down)
        x_4up = self.spec_convt_4up(self.spatial_out(x_spatial))
        x_4up_cat = torch.cat((x_3down, x_4up), 1)
        x_3up = self.spec_convt_3up(self.spec_4halve(x_4up_cat))
        x_3up_cat = torch.cat((x_2down, x_3up), 1)
        x_2up = self.spec_convt_2up(self.spec_3halve(x_3up_cat))
        x_2up_cat = torch.cat((x_1down, x_2up), 1)
        x_1up = self.spec_convt_1up(self.spec_2halve(x_2up_cat))
        x_1up_cat = torch.cat((x, x_1up), 1)
        x_soft = self.spec_1down_soft(self.spec_1halve(x_1up_cat))
        x_out = self.spec_2down_soft(x_soft)
        return x_out

class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv2d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose2d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class SubNet2Convt(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.convt1 = torch.nn.ConvTranspose2d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(n_out)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x
