open System
open MathNet.Numerics.LinearAlgebra

let ComputeDerivative (data : Matrix<float>) (theta : Matrix<float>) : Matrix<float> =
    theta

let GradientDescent (trainingData : Matrix<float>) (testData : Matrix<float>) (learningRate : float) (trials : int) : Matrix<float> =
    //* STEP 1: pick a random value of theta
    let theta = Matrix<float>.Build.Random(3, 1) * 0.01
    printfn "Random initial guess for Theta:"
    printfn "%A" theta
    printfn "-----------------------------------"

    //* STEP 2: Do gradient descent for a fixed number of learning trials using the training data
    for index in 0 .. trials do
        let derivative : Matrix<float> = (ComputeDerivative trainingData theta)
        theta.Subtract(learningRate * derivative)

    theta

[<EntryPoint>]
let main argv =
    let trainingData        = Matrix<float>.Build.Dense(1, 12, [|25733.0; 25971.0; 26458.0; 26430.0; 24874.0; 25413.0; 25538.0; 24527.0; 22878.0; 23996.0; 24579.0; 25426.0|])
    let testData            = Matrix<float>.Build.Dense(1, 12, [|25916.0; 25703.0; 25928.0; 26143.0; 26592.0; 25325.0; 24815.0; 26004.0; 26599.0; 27219.0; 27198.0; 26287.0|])
    let normalizationFactor = 10000.0
    let learningRate        = 0.0008
    let trials              = 120

    let normalizedTrainingData  = trainingData / normalizationFactor
    let normalizedTestData      = testData / normalizationFactor

    let theta = (GradientDescent normalizedTrainingData normalizedTestData learningRate trials)

    //printfn "%A" trainingData

    0 // return an integer exit code