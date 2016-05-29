require 'torch'
require 'image'

local eval = {}

function eval.shave(img, sf)	
	local simg = img[{{},{1+sf,img:size(2)-sf},{1+sf,img:size(3)-sf}}]
	--simg:apply(function(x) if x < 0 then return 0 end end)
	--simg:apply(function(x) if x > 1 then return 1 end end)
	simg = torch.round(simg*255)
	return simg
end

function eval.psnr(img1, img2)
	if img1:size(1) == 3 then
		img1 = image.rgb2yuv(img1)[{{1},{},{}}]
	end

	if img2:size(1) == 3 then
		img2 = image.rgb2yuv(img2)[{{1},{},{}}]
	end

	local imdiff = img1 - img2
	local imdiffPow = imdiff:pow(2)

	local rmse = math.sqrt(imdiffPow:mean())
	local psnr = 20*math.log10(255/rmse)

	return psnr
end

return eval
