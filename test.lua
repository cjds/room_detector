require 'hdf5'
require 'nn'
local h5= 'captions.h5'
local no_of_classes=5
local h5_file = hdf5.open(h5, 'r')
local images_size = h5_file:read('/images'):dataspaceSize()
local seq_size = h5_file:read('/labels'):dataspaceSize()

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 8, 8)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(4,4,4,4))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(5, no_of_classes))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems


-- this is the size of images in the Byte Tensor
local img_raw = torch.DoubleTensor(1, 3, 256, 256)
local label_batch = torch.DoubleTensor(1, no_of_classes)
for i=1,images_size[1] do

    -- fetch the image from h5
    img_raw= h5_file:read('/images'):partial({i,i},{1,3},
                            {1,256},{1,256})
    img_raw=img_raw[1]
    -- fetch the sequence labels
    local seq = h5_file:read('/labels'):partial({i,i+no_of_classes})
    net_output = net:forward(torch.DoubleTensor(img_raw:size()):copy(img_raw))
    print (net_output)
    -- net:zeroGradParameters()
    -- print seq
    -- gradInput = net:backward(net_output, seq)
end
