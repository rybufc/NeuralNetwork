using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new NeuralNetwork(2, new []{2}, 1);
            var dataSets = new List<DataSet>
            {
                new DataSet(new[] {0.0, 0.0}, new[] {0.0}),
                new DataSet(new[] {1.0, 1.0}, new[] {0.0}),
                new DataSet(new[] {0.0, 1.0}, new[] {1.0}),
                new DataSet(new[] {1.0, 0.0}, new[] {1.0})
            };
            network.Train(dataSets, 5000);
            Console.WriteLine(network.Compute(new[] { 1.0, 0.0 })[0]);
            Console.WriteLine(network.Compute(new[] { 0.0, 1.0 })[0]);
            Console.WriteLine(network.Compute(new[] { 0.0, 0.0 })[0]);
            Console.WriteLine(network.Compute(new[] { 1.0, 1.0 })[0]);
            Console.ReadLine();
        }
    }
}
