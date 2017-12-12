using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class Synapse
    {
        public Neuron InputNeuron { get; set; }
        public Neuron OutputNeuron { get; set; }
        public double Weight { get; set; }
        public double LastUpdate { get; set; }

        public Synapse() {}

        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Random rnd = new Random();
            LastUpdate = 0;
            Weight = rnd.NextDouble() * 2 - 1;
        }

        public double Gradient()
        {
            return InputNeuron.Output * OutputNeuron.Delta;
        }
    }
}
