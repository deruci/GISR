--[[get Test image by index using dataLoader class]]--

local data = {}

function data.new(dataset_route, opt_)
	opt_ = opt_ or {}
	local self = {}
	for k,v in pairs(data) do
		self[k] = v
	end

	paths.dofile('taskmaster.lua')
	return self
end

function data:getTest(index)
	assert(opt.sopt, 'opt.sopt not set')
	assert(opt.color, 'opt.color not set')
	return loader:get(index, opt.sopt, opt.color)
end

return data
