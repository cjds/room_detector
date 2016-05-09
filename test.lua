require 'hdf5'

local h5= 'captions.h5'

local h5_file = hdf5.open(h5, 'r')
local images_size = self.h5_file:read('/images'):dataspaceSize()
local seq_size = self.h5_file:read('/labels'):dataspaceSize()

