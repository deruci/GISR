-- [[ file for dataLoader class definition ]] --

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'sys'
require 'xlua'

--local ffi = require 'ffi'
local class = require('pl.class')
--local dir = require 'pl.dir'
local argcheck = require 'argcheck'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
	pack=true,
	help=[[
		A simplified version of dataLoader by Soumith.
		Function will be added whenever needed.
    ]],
	
	{check=function(dirpath)
		local out = true;
		for k,v in ipairs(dirpath) do
			if type(v) ~= 'string' then
				print('paths can only be of string input');
				out = false
			end
		end
		return out
	end,
	name="dirpath",
	type="table",
	help="Multiple paths of directories with images"},

	{name="sampleSize",
	 type="table",
	 help="CxWxH"},

	{name="sampleImageNum",
	 type="number",
	 help="Number of images used per sample quantity",
	 default="8"},

	{name="samplingMode",
	 type="string",
	 help="Sampling mode : random / balanced",
	 default="balanced"},

	{name="verbose",
	 type="boolean",
	 help="Verbose mode",
	 default=false},

	{name="loadSize",
	 type="table",
	 help="a size to load the images, CxWxH",
	 opt=true},

	{name="sampleHookTrain",
	 type="function",
	 help="to sample during training. input would be image path",
	 opt=true},

	{name="sampleHookTest",
	 type="function",
	 help="to sample during test",
	 opt=true},
}

function dataset:__init(...)

	--argcheck first
	local args = initcheck(...)
	print(args)
	for k,v in pairs(args) do self[k] = v end

	if not self.loadSize then self.loadSize = self.sampleSize end
	if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
	if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end
	--seems to be not defined..?
		
	--enlist all paths into datapaths table
	local dataPath = {}

	for i=1,#self.dirpath do
		table.insert(dataPath, paths.concat(self.dirpath[i]))
		for p in paths.iterdirs(self.dirpath[i]) do
			print(self.dirpath[i])
			table.insert(dataPath, paths.concat(self.dirpath[i],p))
		end
	end

	print(dataPath)
	
	--options for find command
	local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}

	--find the image path names
	self.imagePath = {}

	--iterate over all paths
	for i=1,#dataPath do
		for f in paths.files(dataPath[i],"png") do
			table.insert(self.imagePath, paths.concat(dataPath[i],f))
		end
	end

	print('load the large concatenated list of sample paths to self.imagePath')
	--assert(length > 0, "Cannot find any files")
	--assert(maxPathLength > 0, "Please check if the given paths are valid")
	--self.imagePath:resize(gtLength, maxGtPathLength):fill(0)												   
	self.numImages = #self.imagePath
	if self.verbose then print(self.numImages .. ' images found.') end
end

function dataset:size()
	return self.numImages
end

function dataset:show(i)
	--print(self.gtImagePath[i])
	--print(self.lrImagePath[i])
	return self.imagePath[i]
end

local function tableToOutput(self, lrTable, gtTable, count)
	local lr, gt
	assert(lrTable[1]:dim()==3)
	assert(gtTable[1]:dim()==3)
	lr = torch.Tensor(count, self.sampleSize[1],self.sampleSize[2], self.sampleSize[3])
	gt = torch.Tensor(count, self.sampleSize[1],self.sampleSize[2], self.sampleSize[3])
	for i=1,count do
		lr[i]:copy(lrTable[i])
		gt[i]:copy(gtTable[i])
	end
	return lr, gt
end

function dataset:sample(quantity, sopt)
	if type(sopt) == 'number' then
		sf = sopt
	end
	assert(quantity > tonumber(self.sampleImageNum))
	assert(quantity % tonumber(self.sampleImageNum) == 0)
	local quantityPerImage = math.ceil(quantity / self.sampleImageNum)
	local lrTable = {}
	local gtTable = {}
	for i=1,self.sampleImageNum do
		local index = math.ceil(torch.uniform() * self.numImages)
		local imgPath = self.imagePath[index]
		self:sampleHookTrain(imgPath, sf, quantityPerImage, lrTable, gtTable)
	end
	local lr, gt = tableToOutput(self, lrTable, gtTable, quantity)
	return lr, gt
end

function dataset:get(index, sopt, ifColor)
	local ifColor = ifColor or false
	if type(sopt) == 'number' then
		sf = sopt
	end
	--print(self.paths)
	local lrTable = {}
	local gtTable = {}
	local imgPath = self.imagePath[index]
	local lr, gt = self:sampleHookTest(imgPath, sf, ifColor)
	return lr, gt
end

return dataset
