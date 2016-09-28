require 'torch'
require 'gnuplot'

--data samples
local nData = 10

--kernel width
local kWidth =1

--generating data (uniform grid)
local xTrain = torch.linspace(-1,1,nData)

--quadratic fun: y^2
local xTrain = torch.pow(xTrain,2)

-- adding Gaussian noise ()
local yTrain = yTrain + torch.mul(torch.randn(nData),0,1)

--create kernel function
