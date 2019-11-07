using System;
using System.Linq;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace Chapter1 {
    class Program {
        private static readonly Matrix<double>  trainingData        = Matrix<double>.Build.Dense(1, 12, new double[] { 25733, 25971, 26458, 26430, 24874, 25413, 25538, 24527, 22878, 23996, 24579, 25426 }),
                                                testData            = Matrix<double>.Build.Dense(1, 12, new double[] { 25916, 25703, 25928, 26143, 26592, 25325, 24815, 26004, 26599, 27219, 27198, 26287 });

        private static readonly double          normalizationFactor = 10000d,
                                                learningRate        = 0.0008d;

        private static readonly int             trials              = 120;

        static void Main(string[] args) {
            Matrix<double>  normalizedTrainingData  = trainingData / normalizationFactor,
                            normalizedTestData      = testData / normalizationFactor,
                            theta;

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            theta = GradientDescent(normalizedTrainingData, normalizedTestData, learningRate, trials);

            stopwatch.Stop();

            Debug.Write("Time to run: ");
            Debug.WriteLine(stopwatch.ElapsedMilliseconds.ToString());

            //* STEP 3: Print out Value of THETA which was estimated
            Console.WriteLine("Estimated Parameter Value of Theta:");
            Console.WriteLine(theta.ToString());
        }

        private static Matrix<double> GradientDescent(in Matrix<double> trainingData, in Matrix<double> testData, double learningRate, int trials) {
            //* STEP 1: pick a random value of theta
            Matrix<double> theta = Matrix<double>.Build.Random(3, 1) * 0.01d;

            theta = Matrix<double>.Build.Dense(3, 1, new double[] { -0.0019, 0.0089, -0.0076 });

            Debug.WriteLine("Random Initial Guess for Theta:");
            Debug.WriteLine(theta.ToString());
            Debug.WriteLine("---------------------------------------");

            double[]    trainingError   = new double[trials],
                        testError       = new double[trials];

            //* STEP 2: do gradient descent for a fixed number of learning trials using the training data
            for (int index = 0; index < trials; index++) {
                //* this is the gradient descent algorithm
                Matrix<double> derivative = ComputeDerivative(trainingData, theta);
                theta -= (learningRate * derivative);

                //* compute empirical risk function on training data for display purposes
                trainingError[index]    = ComputeEmpiricalRisk(trainingData, theta);

                //* compute empirical risk function on testing data for display purposes
                testError[index]        = ComputeEmpiricalRisk(testData, theta);
            }

            return theta;
        }

        private static double ComputeEmpiricalRisk(in Matrix<double> data, in Matrix<double> theta) {
            int     length  = Math.Max(data.ColumnCount, data.RowCount);
            double  total   = 0d;

            for (int index = 2; index < length; index++) {
                double  predictedDJI    = (theta[0, 0] * data[0, index - 1]) + (theta[1, 0] * data[0, index - 2]) + theta[2, 0],
                        observedDJI     = data[0, index],
                        errorSignal     = observedDJI - predictedDJI;

                total += Math.Pow(errorSignal, 2);
            }

            return total / (length - 2);
        }

        private static Matrix<double> ComputeDerivative(in Matrix<double> data, in Matrix<double> theta) {
            int             length          = Math.Max(data.ColumnCount, data.RowCount);
            Matrix<double>  total           = Matrix<double>.Build.Dense(3, 1, 0);
            double[]        predictedDJI    = new double[length],
                            observedDJI     = new double[length];

            for (int index = 2; index < length; index++) {
                predictedDJI[index] = (theta[0, 0] * data[0, index - 1]) + (theta[1, 0] * data[0, index - 2]) + theta[2, 0];
                observedDJI[index]  = data[0, index];

                double          errorSignal = observedDJI[index] - predictedDJI[index];
                Matrix<double>  dcdTheta    = Matrix<double>.Build.Dense(3, 1, new double[] {
                    -2 * errorSignal * data[0, index - 1],
                    -2 * errorSignal * data[0, index - 2],
                    -2 * errorSignal
                });

                total += dcdTheta;
            }

            return total / (length - 2);
        }
    }
}
