--[[Multithread caller that calls taskmaster]]--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}
local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(nThread, dataset_route, opt_)
	opt_ = opt_ or {}
	local self = {}
	for k,v in pairs(data) do
		self[k] = v
	end

	if nThread > 0 then
		local options = opt_
		self.threads = Threads(nThread,
								function () require 'torch' end,
								function (idx)
									opt = options
									tid = idx
									local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
									torch.manualSeed(seed)
									torch.setnumthreads(1)
									print(string.format('Starting tm with id: %d seed :%d', tid, seed))
									assert(options, 'options not found')
									assert(opt, 'opt not given')
									print(opt)
									paths.dofile('taskmaster.lua')
								end
		)
	else
		paths.dofile('taskmaster.lua')
		self.threads = {}
		function self.threads:addjob(f1, f2) f2(f1()) end
		function self.threads:dojob() end
		function self.threads:synchronize() end
	end
	local nSamples = 0
	self.threads:addjob(function() return loader:size() end,
			function(c) nSamples = c end)
	self.threads:synchronize()
	self._size = nSamples

	for i = 1,n do
		self.threads:addjob(self._sampleFromThreads, self._pushResult)
	end

	return self
end

function data._sampleFromThreads()
	assert(opt.batchSize, 'opt.batchSize not found')
	assert(opt.sopt, 'opt.sopt not set')
	assert(loader)
	return loader:sample(opt.batchSize, opt.sopt)
end

--[[function data._getFromThreads()
	assert(opt.sopt, 'opt.sopt not set')
	assert(opt.color, 'opt.color not set')
	assert(trainLoader)
	return trainLoader:get(opt.sopt, opt.color)
end]]--

function data._pushResult(...)
	local res = {...}
	if res == nil then
		self.threads:synchronize()
	end
	result[1] = res
end	

function data:getBatch()
	self.threads:addjob(self._sampleFromThreads, self._pushResult)
	self.threads:dojob()
	local res = result[1]
	result[1] = nil
	if torch.type(res) == 'table' then
		return unpack(res)
	end
	--print(type(res))
	return res
end

--[[function data:getTest()
	self.threads:addjob(self._getFromThreads, self._pushResult)
	self.threads:dojob()
	local res = result[1]
	result[1] = nil
	if torch.type(res) == 'table' then
		return unpack(res)
	end
	--print(type(res))
	return res
end]]--

function data:size()
	return self._size
end

return data
