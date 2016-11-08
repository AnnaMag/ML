----------------------------------------------------------------------
-- logreg.lua
--
-- Multinomial logistic regression
--

require 'torch'
require 'paths'
require 'nn'
require 'optim'
require 'csvigo'

data = csvigo.load('data.csv')

brands = torch.Tensor(data.brand)
females = torch.Tensor(data.female)
ages = torch.Tensor(data.age)

dataset_inputs = torch.Tensor( (#brands)[1],2 )
dataset_inputs[{ {},1 }] = females
dataset_inputs[{ {},2 }] = ages
dataset_outputs = brands

numberOfBrands = torch.max(dataset_outputs) - torch.min(dataset_outputs) + 1

-- defining the model
-- There are two inputs (female and age) and three outputs (one for each
-- value that brand can take on)

linLayer = nn.Linear(2,3)

-- The soft max layer takes the 3 outputs from the linear layer and
-- transforms them to lie in the range (0,1) and to sum to 1.
-- The log soft max layer takes the log of these 3 outputs (thus the
-- LogSoftMax and not the SoftMax).
-- to be fed into the ClassNLLCriterion

softMaxLayer = nn.LogSoftMax()  -- the input and output are a single tensor

-- sequential container
model = nn.Sequential()
model:add(linLayer)
model:add(softMaxLayer)

-- Define a loss function: the cross entropy between
-- the predictions of the linear model and the groundtruth available
-- in the dataset.
-- Minimizing the cross-entropy is equivalent to maximizing the
-- maximum a-posteriori (MAP) prediction, which is equivalent to
-- minimizing the negative log-likelihoood (NLL), hence the use of
-- the NLL loss.

criterion = nn.ClassNLLCriterion()
-- Training the model (SGD)

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these
-- parameters:

x, dl_dx = model:getParameters()
-- x= weights, dl_dx= derivatives wrt the weights

-- Define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights:
-- all the weights of the linear matrix plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#dataset_inputs)[1] then _nidx_ = 1 end

   local inputs = dataset_inputs[_nidx_]
   local target = dataset_outputs[_nidx_]

   -- reset gradients (gradients are always accumulated, to accomodate
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   -- forward pass
   local loss_x = criterion:forward(model:forward(inputs), target)
   --backpropagation
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

epochs = 10 --1e2  -- number of times to cycle over the training data

print('===========')
print('Training with SGD')
print('')

for i = 1,epochs do

   -- the average loss
   current_loss = 0

   -- epochs
   for i = 1,(#dataset_inputs)[1] do

      -- optim parameters:
      --   + a closure that computes the loss, and its gradient wrt to x,
      --     given a point x
      --   + a point x
      --   + algorithm-specific parameters

      _,fs = optim.sgd(feval, x, sgd_params)

      -- Optim returs:
      --   + the new x, found by the optimization method
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end
   -- report average error on epoch
   current_loss = current_loss / (#dataset_inputs)[1]
   print('epoch = ' .. i ..	 ' current loss = ' .. current_loss)
end

-- return index of largest value
function maxIndex(a,b,c)
   if a >=b and a >= c then return 1
   elseif b >= a and b >= c then return 2
   else return 3 end
end

-- return predicted brand and the probabilities of each brand for this model
function prediction(age, female)
   local input = torch.Tensor(2)
   input[1] = female  -- maintain the order of variables
   input[2] = age
   local logProbs = model:forward(input)
   local probs = torch.exp(logProbs)
   local prob1, prob2, prob3 = probs[1], probs[2], probs[3]
   return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
end

print(prediction(1,2))
