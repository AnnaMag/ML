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
ninputs = 2; noutputs = 1
model:add(nn.Linear(ninputs, noutputs)) -- define the only module
