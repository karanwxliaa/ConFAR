require 'torch'
require 'nn'
require 'paths'

-- Load the pre-trained model
local net = torch.load('./VGG_FACE.t7')

local weights = {}
for i=1,#net.modules do
    local layer = net.modules[i]
    if layer.weight then
        table.insert(weights, {weight = layer.weight:float(), bias = layer.bias:float()})
    end
end

-- Save weights and biases in a format accessible by Python
torch.save('vgg_face_weights.t7', weights)
-- Alternatively, if you have matio installed:
-- require 'matio'
-- matio.save('vgg_face_weights.mat', weights)
