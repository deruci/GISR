local util = {}

function cleanupModel(node)
	if node.output ~= nil then
		node.output = zeroDataSize(node.output)
	end
	if node.gradInput ~= nil then
		node.gradInput = zeroDataSize(node.gradInput)
	end
	if node.finput ~= nil then
		node.finput = zeroDataSize(node.finput)
	end
	-- Recurse on nodes with 'modules'
	if (node.modules ~= nil) then
		if (type(node.modules) == 'table') then
			for i = 1, #node.modules do
				local child = node.modules[i]
				cleanupModel(child)
			end
		end
	end
	collectgarbage()
end

return util
