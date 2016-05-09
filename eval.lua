require 'torch'

local eval = {}

function eval.shave(img, sf)	
	local simg = img[{{},{1+sf,img:size(2)-sf},{1+sf,img:size(3)-sf}}]
	--simg:apply(function(x) if x < 0 then return 0 end end)
	--simg:apply(function(x) if x > 1 then return 1 end end)
	simg = torch.trunc(simg*255)
	simg:apply(function(x) if x < 0 then return 0 end end)
	simg:apply(function(x) if x > 255 then return 255 end end)
	return simg
end

function eval.psnr(img1, img2)
	--assert(img1:size() ==  img2:size())	
	--print(img1:size(), img2:size())
	--print(img1:type(), img2:type())
	local imdiff = img1 - img2
	local imdiffPow = imdiff:pow(2)

	local rmse = math.sqrt(imdiffPow:mean())
	local psnr = 20*math.log10(255/rmse)

	return psnr
end

return eval
