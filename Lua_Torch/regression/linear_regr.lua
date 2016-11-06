require 'torch'
require 'optim'
require 'clnn' -- neural network package with openCl bakcend

logger = optim.Logger('loss_log.txt')

-- note to self: in non-liner models we add 1
-- SGD for linear neurons
-- there is a single layer (corn, fertilizer) ->  corn
--  {corn, fertilizer, insecticide}
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

-- one layer, linear model specified below has 3 parameters
model = nn.Sequential()                 -- define the container
ninputs = 2; noutputs = 1cd
model:add(nn.Linear(ninputs, noutputs)) -- define the only module

-- loss function
criterion = nn.MSECriterion()

print(criterion)
x, dl_dx = model:getParameters()

-- min the loss using SGD

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end
   _nidx_ = (_nidx_ or 0) + 1
   print(_nidx)
   -- check
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
      local target = sample[{ {1} }]      -- this funny looking syntax allows
      local inputs = sample[{ {2,3} }]    -- slicing of arrays.

      dl_dx:zero()

      -- evaluate the loss function and its derivative wrt x, for that sample
      local loss_x = criterion:forward(model:forward(inputs), target)
      model:backward(inputs, criterion:backward(model.output, target))

  return loss_x, dl_dx
end
