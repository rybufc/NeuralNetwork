using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class DataSet
    {
        #region -- Properties --
        public double[] Values { get; set; }
        public double[] Targets { get; set; }
        #endregion

        #region -- Constructor --
        public DataSet(double[] values, double[] targets)
        {
            Values = values;
            Targets = targets;
        }
        #endregion
    }
}
