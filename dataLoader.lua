-- [[ file for dataLoader class definition ]] --

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'sys'
require 'xlua'

local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local argcheck = require 'argcheck'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
	pack=true,
	help=[[
		A simplified version of dataLoader by Soumith.
		Function will be added whenever needed.
    ]],
	
	{check=function(paths)
		local out = true;
		for k,v in ipairs(paths) do
			if type(v) ~= 'string' then
				print('paths can only be of string input');
				out = false
			end
		end
		return out
	end,
	name="paths",
	type="table",
	help="Multiple paths of directories with images"},

	{name="sampleSize",
	 type="table",
	 help="WxHxC"},

	{name="sampleImageNum",
	 type="number",
	 help="Number of images used per sample quantity",
	 default="10"},

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
	 help="a size to load the images, WxHxC",
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

	--define command-line tools
	local wc = 'wc'
	local cut = 'cut'
	local find = 'find'
	
	--enlist all paths into datapaths table
	local dataPaths = {}

	for k,path in ipairs(self.paths) do
		local dirs = dir.getdirectories(path)
		for k,dirpath in ipairs(dirs) do
			table.insert(dataPaths, dirpath)
		end
	end

	--options for find command
	local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
	local findOptions = ' -name "*.' .. extensionList[1] .. '"'
	for i=2,#extensionList do
		findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
	end

	--find the image path names
	self.imagePath = torch.CharTensor()

	local findFiles = os.tmpname()
	local tmpfile = os.tmpname()
	local tmphandle = assert(io.open(tmpfile, 'w'))

	--iterate over all dataPaths
	for i, path in ipairs(dataPaths) do
		local command = find .. ' "' .. path .. '" ' .. findOptions .. ' >>"' .. findFiles .. '" \n'
		tmphandle:write(command)
	end
	io.close(tmphandle)
	os.execute('bash ' .. tmpfile)
	--os.execute('rm -f ' .. tmpfile)

	print('load the large concatenated list of sample paths to self.imagePath')
	local maxPathLength = tonumber(sys.fexecute(wc .. " -L '" .. findFiles .. "' |" .. cut .. " -f1 -d' '")) + 1
	local length = tonumber(sys.fexecute(wc .. " -l '" .. findFiles .. "' |" .. cut .. " -f1 -d' '"))
	assert(length > 0, "Cannot find any files")
	assert(maxPathLength > 0, "Please check if the given paths are valid")
	self.imagePath:resize(length, maxPathLength):fill(0)
	local s_data = self.imagePath:data() --LuaJUT FFI access
	local count = 0												 
	for line in io.lines(findFiles) do
		ffi.copy(s_data, line)
		s_data = s_data + maxPathLength
		if self.verbose and count % 5000 == 0 then
			xlua.progress(count, length)
		end
		count = count + 1
	end
	
	self.numImages = self.imagePath:size(1)
	if self.verbose then print(self.numImages .. ' images found.') end
	
	--clean up temporary file
	os.execute('rm -f "' ..  findFiles .. '"')
end

function dataset:size()
	return self.numImages
end

local function tableToOutput(self, degTable, gtTable, count)
	local deg, gt
	assert(degTable[1]:dim()==3)
	assert(gtTable[1]:dim()==3)
	deg = torch.Tensor(count, self.sampleSize[1],self.sampleSize[2],self.sampleSize[3])
	gt = torch.Tensor(count, self.sampleSize[1],self.sampleSize[2],self.sampleSize[3])
	for i=1,count do
		deg[i]:copy(degTable[i])
		gt[i]:copty(gtTable[i])
	end
	return deg, gt
end

function dataset:sample(quantity)
	assert(quantity)
	local quantityPerImage = math.ceil(quantity / self.sampleImageNum)
	local degTable = {}
	local gtTable = {}
	local count = 0
	for i=1,self.sampleImageNum do
		local index = math.ceil(torch.uniform() * self.numImages)
		local imgpath = ffi.string(torch.data(self.imagePath[index]))
		local d, g = self:sampleHookTrain(quantityPerImage,imgpath)
		table.insert(degTable, d)
		table.insert(gtTable, g)
		count = count + 1
	end
	local deg, gt = tableToOutput(self, degTable, gtTable, count)
	return deg, gt
end

return dataset



