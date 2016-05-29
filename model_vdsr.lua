local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local depth = opt.depth
	local nChannel = opt.nChannel

	local model = nn.Sequential()
	model:add(Convolution(1,nChannel,3,3,1,1,1,1))
	model:add(ReLU(true))
	for i=1,depth-2 do
		model:add(Convolution(nChannel,nChannel,3,3,1,1,1,1))
		model:add(ReLU(true))
	end
	model:add(Convolution(nChannel,1,3,3,1,1,1,1))

	local function ConvInit(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			if cudnn.version >= 4000 then --what for?
				v.bias = nil
				v.gradBias = nil
			else
				v.bias:zero()
			end
		end
	end

	local function BNInit(name)
		for k,v in pairs(model:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end

	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')

	--criterion
	local criterion = nn.MSECriterion()

	return model, criterion
end

return createModel
