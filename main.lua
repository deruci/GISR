require 'torch'
require 'nn'
require 'optim'

eval = paths.dofile('eval.lua')
createModel = paths.dofile('model_vdsr.lua') 
trainLoader = paths.dofile('taskTrain.lua')
testLoader = paths.dofile('taskTest.lua')

opt = {
	dataset = 'data/training',
	depth = 20,
	batchSize = 64,
	loadSize = 41,
	fineSize = 41,
	nThreads = 4,
	nIter = 80000,
	lr = 0.1,
	clip = 0.01,
	momentum = 0.9,
	nChannel = 64,
	display = 1,
	display_id = 10,
	gpu = 1,
	sopt = 2,
	color = 0,
	weightDecay = 0.0001
}
opt.name = string.format('VDSR_sgd_exp_sf_%d',opt.sopt)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local model, criterion = createModel(opt)
local trainData = trainLoader.new(opt.nThreads, opt.dataset, opt)
opt.dataset = 'data/test/Set5'
local testData = testLoader.new(opt.dataset, opt)
opt.dataset = 'data/training'
print("Dataset: " .. opt.dataset, " Size: ", trainData:size())

optimState = {
	learningRate = opt.lr,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum
}

local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local loss
local lossPlot = {}
local samplePSNR = {}

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

	local lowBatch, highBatch = trainData:getBatch()
	input:copy(lowBatch); label:copy(highBatch)
	local output = model:forward(input)
	loss = criterion:forward(output, label)
	local dfdo = criterion:backward(output, label)
	model:backward(input, dfdo)

	gradParameters:clamp(-opt.clip/opt.lr, opt.clip/opt.lr)

	return loss, gradParameters
end

for iterNum = 1, opt.nIter do
	if iterNum == 20000 or iterNum == 40000 or iterNum == 60000 then
		opt.lr = opt.lr * 0.1
	end

	optim.sgd(fx, parameters, optimState)
	if (iterNum % 50) == 0 then	--visualize / caculate intermidiate result
		local low, high = testData:getTest(2)
		if opt.gpu > 0 then
			low = low:cuda()
		end
		model:evaluate()
		local resPred = model:forward(low:resize(1,1,low:size(2),low:size(3)))
		model:training()
		resPred = resPred[1]; low = low[1];
		local pred = resPred + low
		--print(pred:max(), resPred:max(), low:max())
		--pred:apply(function(x) if x < 0 then return 0 end end)
		--pred:apply(function(x) if x > 1 then return 1 end end)
		disp.image(low, {win=opt.display_id, titie='LR image'})
		disp.image(resPred, {win=opt.display_id + 1, title ='pred residual image'})
		disp.image(pred, {win=opt.display_id + 2, title = 'SR image'})
		disp.image(high, {win=opt.display_id + 3, title = 'GT image'})
		pred = pred:type('torch.FloatTensor')
		local psnr =  eval.psnr(eval.shave(pred,opt.sopt), eval.shave(high,opt.sopt))
		table.insert(samplePSNR, {iterNum, psnr})
		disp.plot(samplePSNR, {win=opt.display_id + 4, title='sample psnr', labels = {'iter', 'psnr'}})

		table.insert(lossPlot, {iterNum, loss*1000})
		disp.plot(lossPlot, {win=opt.display_id + 5, title='loss x1000 plot', labels = {'iter', 'loss'}})
	end

	if (iterNum % 10000) == 0 then --Save model.
		parameters, gradParameters = nil, nil
		torch.save('checkpoints/' .. opt.name .. '_' .. iterNum .. '_net.t7', model)
		parameters, gradParameters = model:getParameters()
	end
end

