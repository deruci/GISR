local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local depth = opt.depth
	local nChannel = opt.nChannel

	local function bottleneck(nInputPlane, nOutputPlane, stride)

		local nBottleneckPlane = nOutputPlane / 4
		
		if nInputPlane == nOutputPlane then
			local convs = nn.Sequential()
			--conv1x1
			convs:add(SBatchNorm(nInputPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
			--conv3x3
			convs:add(SBatchNorm(nBottleneckPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,stride,stride,1,1))
			--conv1x1
			convs:add(SBatchNorm(nBottleneckPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,stride,stride,0,0))

			local shortcut = nn.Identity()
			
			return nn.Sequential():add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
		else
			local block = nn.Sequential()
			--common BN, ReLU
			block:add(SBatchNorm(nInputPlane))
			block:add(ReLU(true))

			local convs = nn.Sequential()
			--conv1x1
			convs:add(SBatchNorm(nInputPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
			--conv3x3
			convs:add(SBatchNorm(nBottleneckPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,stride,stride,1,1))
			--conv1x1
			convs:add(SBatchNorm(nBottleneckPlane))
			convs:add(ReLU(true))
			convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,stride,stride,0,0))

			local shortcut = nn.Sequential()
			shortcut:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))

			return block:add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
		end
	end
	
	--Stacking Residual Units on the same stage
	local function layer(block, nInputPlane, nOutputPlane, count, stride)
		local s = nn.Sequential()
		s:add(block(nInputPlane, nOutputPlane, stride))
		for i=2,count do
			s:add(block(nInputPlane, nOutputPlane, stride))
		end
		return s
	end

	local model = nn.Sequential()
	model:add(Convolution(1,nChannel,3,3,1,1,1,1))
	model:add(layer(bottleneck,nChannel,nChannel,depth,1))
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

	model:cuda()
	model:get(1).gradInput = nil

	--criterion
	local criterion = nn.MSECriterion()

	return model, criterion
end

return createModel
