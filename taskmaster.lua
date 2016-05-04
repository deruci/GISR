--[[taskmaster to utilize dataLoader, containing how to I/O files]]--

require 'image'
paths.dofile('dataLoader.lua')

--check if there is opt.data
--opt.data = os.getenv('DATA_ROOT') or 'data/training'
if not paths.dirp(opt.dataset) then
	error('Cannot find directory : ', opt.dataset)
end

--keep cache file of meta-training data.
local cache = "cache"
local cache_prefix = opt.dataset:gsub('/','_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

local sampleSize = {1, opt.fineSize} --would be #layers*2 + 1

local function loadImage(path)
	local input = image.rgb2yuv(image.load(path, 3, 'float'))[{{1},{},{}}]
	return input
end

local function loadColorImage(path)
	local input = image.rgb2yuv(image.load(path, 3, 'float'))
	return input
end

local function modcrop(im, sf)
	local sz_w = im:size(3)
	local sz_h = im:size(2)
	sz_w = sz_w - (sz_w % sf)
	sz_h = sz_h - (sz_h % sf)
	return im[{{},{1,sz_h},{1,sz_w}}]
end

local trainHook = function(self, path, sf, quantityPerImage, degTable, gtTable)
	collectgarbage()
	local im = loadImage(path) --Y channel only, return 1*H*W size single-channel image.
	-- do random filp / rotate (not implemented yet)

	local imhigh = modcrop(im, sf)
	local imlow = image.scale(imhigh, imhigh:size(3)/2, imhigh:size(2)/2, 'bicubic')
	imlow = image.scale(imlow, imhigh:size(3), imhigh:size(2), 'bicubic')
	imlow:clamp(16.0/255, 235.0/255)
	imhigh = imhigh - imlow
	assert(imhigh:size(2)==imlow:size(2))
	assert(imhigh:size(3)==imlow:size(3))

	local iW = imhigh:size(3)
	local iH = imhigh:size(2)
	--do random crop
	local oW = sampleSize[2]
	local oH = sampleSize[2]
	for i = 1,quantityPerImage do
		local h1 = math.ceil(torch.uniform(1e-2, iH-oH))	
		local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
		local imsub = image.crop(imhigh, w1, h1, w1+oW, h1+oH)
		local imsublow = image.crop(imlow, w1, h1, w1+oW, h1+oH)
		assert(imsub:size(2)-oH == 0)
		assert(imsub:size(3)-oW == 0)
		--return imsublow, imsub
		table.insert(degTable, imsublow)
		table.insert(gtTable, imsub)
	end
end

local testHook = function(self, path, sf, ifColor)
	collectgarbage()
	local im
	if ifColor == 1 then
		im = loadColorImage(path)
	else
		im = loadImage(path)
	end

	local imhigh = modcrop(im, sf)
	local imlow = image.scale(imhigh, imhigh:size(3)/2, imhigh:size(2)/2, 'bicubic')
	imlow = image.scale(imlow, imhigh:size(3), imhigh:size(2), 'bicubic')
	imlow:clamp(16.0/255, 235.0/255)

	assert(imhigh:size(2)==imlow:size(2))
	assert(imhigh:size(3)==imlow:size(3))
	return imlow, imhigh
end

if paths.filep(trainCache) then
	print('Loading metadata from cache')
	loader = torch.load(trainCache)
	loader.sampleHookTrain = trainHook
	loader.sampleHookTest = testHook
	loader.sampleSize = {1, sampleSize[2], sampleSize[2]}
else
	print('Creating train metadata')
	loader = dataLoader{
		paths = {opt.dataset},
		sampleSize = {1, sampleSize[2], sampleSize[2]},
		verbose = true
	}
	torch.save(trainCache, trainLoader)
	print('saved metadata cache at', trainCache)
	loader.sampleHookTrain = trainHook
	loader.sampleHookTest = testHook
end
collectgarbage()
