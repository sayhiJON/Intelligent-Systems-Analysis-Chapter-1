//* ****************************************************************************************************
//* Ported from Matlab code by Dr. Richard Golden for the Intelligent Systems Analysis Fall 2019 class *
//* ****************************************************************************************************

using System;
using System.Diagnostics;
using MathNet.Numerics;

namespace firstPass_Core {
    class Program {
        private static int[]    trainingData        = new int[] { 25733, 25971, 26458, 26430, 24874, 25413, 25538, 24527, 22878, 23996, 24579, 25426 },
                                testData            = new int[] { 25916, 25703, 25928, 26143, 26592, 25325, 24815, 26004, 26599, 27219, 27198, 26287 };

        private static double   normalizationFactor = 10000d,
                                learningRate        = 0.0008d;

        private static int      trials              = 120;

        static void Main(string[] args) {
            double[]    normalizedTrainingData  = Normalize(trainingData, normalizationFactor),
                        normalizedTestData      = Normalize(testData, normalizationFactor);

            GradientDescent(normalizedTrainingData, normalizedTestData, learningRate, trials);
        }

        private static double[] Normalize(in int[] data, in double factor = 10000d) {
            if (data == null || data.Length == 0)
                throw new ArgumentNullException();

            double[] normalized = new double[data.Length];

            for (int index = 0; index < data.Length; index++)
                normalized[index] = data[index] / factor;

            return normalized;
        }

        private static void GradientDescent(in double[] trainingData, in double[] testData, in double learningRate, in int trials) {
            double[]    theta           = GetRandomDouble(3, 0.01),
                        riskDerivative  = null,
                        trainingError   = new double[trials],
                        testError       = new double[trials];

            Debug.WriteLine("Initial Theta:");
            Array.ForEach(theta, x => Debug.WriteLine(x.ToString()));

            for (int index = 0; index < trials; index++) {
                riskDerivative = ComputeDerivative(trainingData, theta);

                for (int t = 0; t < theta.Length; t++)
                    theta[t] = theta[t] - (learningRate * riskDerivative[t]);

                trainingError[index]    = ComputeEmpiricalRisk(trainingData, theta);
                testError[index]        = ComputeEmpiricalRisk(testData, theta);
            }

            Debug.WriteLine("Final Training Error:");
            Debug.WriteLine(trainingError[trainingError.Length - 1].ToString());

            Debug.WriteLine("Final Test Error:");
            Debug.WriteLine(testError[testError.Length - 1].ToString());

            Debug.WriteLine("Estimated Theta:");
            Array.ForEach(theta, x => Debug.WriteLine(x.ToString()));
        }

        private static double ComputeEmpiricalRisk(in double[] data, in double[] theta) {
            int     length  = data.Length;
            double  total   = 0d;

            for (int index = 2; index < length; index++) {
                double  predictedDJI    = theta[0] * data[index - 1] + theta[1] * data[index - 2] + theta[2],
                        observedDJI     = data[index],
                        errorSignal     = observedDJI - predictedDJI;

                total += Math.Pow(errorSignal, 2);
            }

            return total / (length - 2);
        }

        private static double[] ComputeDerivative(in double[] data, in double[] theta) {
            int length = data.Length;
            double[] sumTotal = new double[theta.Length];

            for (int index = 2; index < length; index++) {
                double  predictedDJI    = theta[0] * data[index - 1] + theta[1] * data[index - 2] + theta[2],
                        observedDJI     = data[index],
                        errorSignal     = observedDJI - predictedDJI,
                        dcdTheta1       = -2 * (observedDJI - predictedDJI) * data[index - 1],
                        dcdTheta2       = -2 * (observedDJI - predictedDJI) * data[index - 2],
                        dcdTheta3       = -2 * (observedDJI - predictedDJI);

                sumTotal[0] += dcdTheta1;
                sumTotal[1] += dcdTheta2;
                sumTotal[2] += dcdTheta3;
            }

            for (int index = 0; index < sumTotal.Length; index++)
                sumTotal[index] = sumTotal[index] / (length - 2);

            return sumTotal;
        }

        private static double[] GetRandomDouble(in int size, in double factor) {
            if (size <= 0)
                throw new ArgumentOutOfRangeException();

            Random      random  = new Random();
            double[]    numbers = new double[size];

            for (int index = 0; index < size; index++) {
                int multiplier = random.Next(0, 2) == 0 ? -1 : 1;
                numbers[index] = factor * random.NextDouble() * multiplier;
            }

            return numbers;
        }
    }
}
