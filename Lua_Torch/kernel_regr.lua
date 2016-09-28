require 'torch'
require 'gnuplot'

-- Kernel regression

--data samples
local nData = 10

--kernel width
local kWidth =1

--generating data (uniform grid)
local xTrain = torch.linspace(-1,1,nData)

--quadratic fun: y^2
local yTrain = torch.pow(xTrain,2)

-- adding Gaussian noise ()
local yTrain = yTrain + torch.mul(torch.randn(nData),0.1)

--create kernel function
local function phi(x, y)
  return torch.exp(-(1/kWidth)*torch.sum(torch.pow(x-y,2)))
end

--
local Phi = torch.Tensor(nData, nData)
for i = 1,nData do
  for j = 1,nData do
    Phi[i][j] = phi(xTrain[{{i}}],xTrain[{{j}}])
  end
end

-- delta = 0.001
local regularizer = torch.mul(torch.eye(nData), 0.001)

-- [Phi^TPhi + delta^2 * Id]^-1*Phi^T*Y
local theta = torch.inverse((Phi:t()*Phi) + regularizer) * Phi:t() *yTrain
print(Phi)
print(theta)

--test data
local nTestData = 100
local xTest = torch.linspace(-1,1,nTestData)

local PhiTest = torch.Tensor(nData, nTestData)
for i = 1,nData do
  for j = 1,nTestData do
    PhiTest[i][j] = phi(xTrain[{{i}}],xTest[{{j}}])
  end
end

--prediction
local yPred = PhiTest:t() * theta

gnuplot.plot({'Data', xTrain, yTrain,'+'}, {'Prediction', xTest, yPred, '-'})

--print(yPred)
