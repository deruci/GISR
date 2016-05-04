require 'torch'
require 'nn'
require 'optim'

util = paths.dofile('util.lua')
createModel = paths.dofile('model.lua') --now model always :cuda()
DataLoader = paths.dofile('taskMultiThread.lua')

opt = {
	dataset = 'data/training',
	depth = 15,
	batchSize = 64,
	loadSize = 35,
	fineSize = 35,
	nThreads = 4,
	nIter = 10000,
	lr = 0.00001,
	momentum = 0.9,
	nChannel = 64,
	display = 0,
	display_id = 10,
	gpu = 1,
	sopt = 2,
	weightDecay = 0.0001
}
opt.name = string.format('sr_exp_sf_%d',opt.sopt)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local model, criterion = createModel(opt)
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

optimState = {
	learningRate = opt.lr,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	nesterov = true,
	dampening = 0
}

local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local loss

if opt.gpu > 0 then
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	input = input:cuda(); label = label:cuda()
	model:cuda(); criterion:cuda()
end

local parameters, gradParameters = model:getParameters()

if opt.display then disp = require 'display' end

local fx = function(x)
	gradParameters:zero()

	local lowBatch, highBatch = data:getBatch()
	input:copy(lowBatch); label:copy(highBatch)

	local output = model:forward(input)
	loss = criterion:forward(output, label)
	local dfdo = criterion:backward(output, label)
	model:backward(input, dfdo)

	return loss, gradParameters
end

for iterNum = 1, opt.nIter do
	optim.sgd(fx, parameters, optimState)
	if (iterNum % 50) == 0 then	--visualize / caculate intermidiate result
		
	end

	if (iterNum % 100) == 0 then --Save model.
	end
end

