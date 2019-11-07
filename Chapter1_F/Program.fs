open System
open MathNet.Numerics.LinearAlgebra
open XPlot.GoogleCharts

let ComputeEmpiricalRisk (data : Matrix<float>) (theta : Matrix<float>) : float =
    let row     = data.Row(0)
    let weights = theta.Column(0)
    let range   = [2 .. row.Count - 1]

    range
        |> Seq.map (fun index -> Math.Pow(row.[index] - ((weights.[0] * row.[index - 1]) + (weights.[1] * row.[index - 2]) + weights.[2]), 2.0))
        |> Seq.average

let ComputeDerivative (data : Matrix<float>) (theta : Matrix<float>) : Matrix<float> =
    let row     : Vector<float> = data.Row(0)
    let weights : Vector<float> = theta.Column(0)
    let range   : List<int>     = [2 .. row.Count - 1]

    let sum : Matrix<float> =
        range
            |> Seq.map (fun index -> row.[index] - ((weights.[0] * row.[index - 1]) + (weights.[1] * row.[index - 2]) + weights.[2]))
            |> Seq.mapi (fun offset errorSignal -> Matrix<float>.Build.Dense(3, 1, [| -2.0 * errorSignal * row.[offset + 1]; -2.0 * errorSignal * row.[offset]; -2.0 * errorSignal;|]))
            |> Seq.fold (fun acc elem -> acc + elem) (Matrix<float>.Build.Dense(3, 1))

    let average : Matrix<float> = sum / (float range.Length)

    average

let GradientDescent (trainingData : Matrix<float>) (testData : Matrix<float>) (learningRate : float) (trials : int) : Matrix<float> =
    //* STEP 1: pick a random value of theta
    let theta = Matrix<float>.Build.Random(3, 1) * 0.01
    let range = [0 .. trials - 1]

    printfn "Random initial guess for Theta:"
    printfn "%A" theta
    printfn "-----------------------------------"

    let trainingError : float array = Array.zeroCreate(trials)
    let testError : float array = Array.zeroCreate(trials)

    let test = ComputeDerivative trainingData theta

    //* STEP 2: Do gradient descent for a fixed number of learning trials using the training data
    let estimatedTheta =
        range
            |> Seq.fold (fun acc elem ->
                let derivative  = ComputeDerivative trainingData acc
                let update      = acc - (learningRate * derivative)

                Array.set trainingError elem (ComputeEmpiricalRisk trainingData update)
                Array.set testError elem (ComputeEmpiricalRisk testData update)

                update) (theta)

    //* STEP 3: Print out Value of THETA which was estimated
    printfn "Estimated Parameter Value Theta:"
    printfn "%A" estimatedTheta
    printfn "====================================================================="

    //* STEP 4: Plot Final Results fo Training Data and Test Data as a Function of Learning Trials
    let series = ["lines"; "lines"]
    let inputs = 
        [
            trainingError
            |> Seq.mapi (fun index data -> (index.ToString(), data))
            testError
            |> Seq.mapi (fun index data -> (index.ToString(), data))
        ]

    inputs
    |> Chart.Combo
    |> Chart.WithOptions
        (Options(title = "Supervised Learning: Stock Market Prediction Problem",
            series = [| for typ in series -> Series(typ) |]))
    |> Chart.WithLabels ["Training Error"; "Test Error"]
    |> Chart.WithLegend true
    |> Chart.WithSize(600, 250)
    |> Chart.Show

    estimatedTheta

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