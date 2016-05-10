require 'hdf5'

local h5= 'captions.h5'
local no_of_classes=5
local h5_file = hdf5.open(h5, 'r')
local images_size = h5_file:read('/images'):dataspaceSize()
local seq_size = h5_file:read('/labels'):dataspaceSize()

print(seq_size)
-- this is the size of images in the Byte Tensor
local img_raw = torch.ByteTensor(1, 3, 256, 256)
local label_batch = torch.LongTensor(1, no_of_classes)
for i=1,images_size[1] do

    -- fetch the image from h5
    local img = h5_file:read('/images'):partial({i,i},{1,3},
                            {1,256},{1,256})
    -- fetch the sequence labels
    local seq = h5_file:read('/labels'):partial({i,i+no_of_classes})

end
