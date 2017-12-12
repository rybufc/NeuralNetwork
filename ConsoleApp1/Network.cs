using System;
using System.Collections;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.Xml;
using System.IO;
using System.Linq;

namespace ConsoleApp1
{

    public class NeuralNetwork
    {
        public double LearnRate { get; set; }
        public double Momentum { get; set; }
        public List<Neuron> InputLayer { get; set; }
        public List<List<Neuron>> HiddenLayers { get; set; }
        public List<Neuron> OutputLayer { get; set; }

        public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, 
            double? learnRate = null, double? momentum= null)
        {
            LearnRate = learnRate ?? 0.3;
            Momentum = momentum ?? 0.4;

            InputLayer = new List<Neuron>();
            HiddenLayers = new List<List<Neuron>>();
            OutputLayer = new List<Neuron>();

            for (int i = 0; i < inputSize; i++)
            {
                InputLayer.Add(new Neuron());
            }

            var firstHiddenLayer = new List<Neuron>();
            for (int i = 0; i < hiddenSizes[0]; i++)
            {
                firstHiddenLayer.Add(new Neuron(InputLayer));
            }
            HiddenLayers.Add(firstHiddenLayer);

            for (int i =  1; i < hiddenSizes.Length; i++)
            {
                var hiddenLayer = new List<Neuron>();
                for (int j = 0; j < hiddenSizes[i]; j++)
                {
                    hiddenLayer.Add(new Neuron(HiddenLayers[i-1]));
                }
                HiddenLayers.Add(hiddenLayer);
            }

            for (var i = 0; i < outputSize; i++)
                OutputLayer.Add(new Neuron(HiddenLayers[HiddenLayers.Count - 1]));
        }

        public void Train(List<DataSet> dataSets, int numOfEpoch)
        {
            for (int epoch = 0; epoch < numOfEpoch; epoch++)
            {
                double maxError = 0;
                DataSet maxErrorSet = dataSets[0];
                for (int i = 0; i < dataSets.Count; i++)
                {
                    Forward(dataSets[i].Values);
//                    BackPropagate(dataSets[i].Targets);
                    var error = MSE(OutputLayer.Select(x => x.Output).ToArray(), dataSets[i].Targets);
                    if (error > maxError)
                    {
                        maxError = error;
                        maxErrorSet = dataSets[i];
                    }
                }
                Forward(maxErrorSet.Values);
                BackPropagate(maxErrorSet.Targets);
            }
        }

        public double[] Compute(double[] inputs)
        {
            Forward(inputs);
            return OutputLayer.Select(x => x.Output).ToArray();
        }

        public void BackPropagate(double[] expected)
        {
            for (int i = 0; i < OutputLayer.Count; i++)
            {
                OutputLayer[i].Delta = outDelta(OutputLayer[i], expected[i]);
            }

            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < HiddenLayers[i].Count; j++)
                {
                    HiddenLayers[i][j].Delta = hDelta(HiddenLayers[i][j]);
                }
            }

            for (int i = 0; i < InputLayer.Count; i++)
            {
                InputLayer[i].Delta = hDelta(InputLayer[i]);
            }

            for (int i = 0; i < InputLayer.Count; i++)
            {
                for (int j = 0; j < InputLayer[i].OutputSynapses.Count; j++)
                {
                    UpdateSynapseWeight(InputLayer[i].OutputSynapses[j]);
                }
            }

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                for (int j = 0; j < HiddenLayers[i].Count; j++)
                {
                    for (int k = 0; k < HiddenLayers[i][j].OutputSynapses.Count; k++)
                    {
                        UpdateSynapseWeight(HiddenLayers[i][j].OutputSynapses[k]);
                    }
                }
            }
        }

        public void Forward(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                InputLayer[i].Output = values[i];
            }

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                for (int j = 0; j < HiddenLayers[i].Count; j++)
                {
                    double result = 0;
                    for (int k = 0; k < HiddenLayers[i][j].InputSynapses.Count; k++)
                    {
                        result += HiddenLayers[i][j].InputSynapses[k].InputNeuron.Output *
                                  HiddenLayers[i][j].InputSynapses[k].Weight;
                    }
                    HiddenLayers[i][j].Input = result;
                    HiddenLayers[i][j].Output = HypTan(result);
                }
            }

            for (int i = 0; i < OutputLayer.Count; i++)
            {
                double result = 0;
                for (int j = 0; j < OutputLayer[i].InputSynapses.Count; j++)
                {
                    result += OutputLayer[i].InputSynapses[j].InputNeuron.Output *
                              OutputLayer[i].InputSynapses[j].Weight;
                }
                OutputLayer[i].Input = result;
                OutputLayer[i].Output = HypTan(result);
            }
        }

        public double HypTan(double x)
        {
            return (Math.Pow(Math.E, 2 * x) - 1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        public double HypTanDiff(double x)
        {
            return 1 - x * x;
        }

        private double outDelta(Neuron outNeuron, double expected)
        {
            return (expected - outNeuron.Output) * HypTanDiff(outNeuron.Output);
        }

        private double hDelta(Neuron hNeuron)
        {
            double result = 0;
            for (int i = 0; i < hNeuron.OutputSynapses.Count; i++)
            {
                result += hNeuron.OutputSynapses[i].OutputNeuron.Delta * hNeuron.OutputSynapses[i].Weight;
            }
            result *= HypTanDiff(hNeuron.Output);

            return result;
        }

        public void UpdateSynapseWeight(Synapse synapse)
        {
            double weightIncrement = LearnRate * synapse.Gradient() + Momentum * synapse.LastUpdate;
            synapse.LastUpdate = weightIncrement;
            synapse.Weight += weightIncrement;
        }

        public double MSE(double[] result, double[] expected)
        {
            double error = 0;
            for (int i = 0; i < result.Length; i++)
            {
                error += Math.Pow(expected[i] - result[i], 2);
            }
            error /= result.Length;

            return error;
        }
    }
}
