using System;
using System.Collections;
using System.Collections.Generic;
using System.Xml.Serialization;

namespace ConsoleApp1
{
    public class Neuron
    {
        public double Delta;
        public double Input { get; set; }
        public double Output { get; set; }
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }

        public Neuron()
        {
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
        }

        public Neuron(IEnumerable<Neuron> inputNeurons) : this()
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = new Synapse(inputNeuron, this);
                inputNeuron.OutputSynapses.Add(synapse);
                InputSynapses.Add(synapse);
            }
        }
    }
}