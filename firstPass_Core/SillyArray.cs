using System;
using System.Collections.Generic;
using System.Text;

namespace firstPass_Core {
    public class SillyArray<T> {
        public T this[int index] {
            get {
                if (this.p_Valid[index])
                    return this.p_Data[index];

                throw new ArgumentOutOfRangeException();
            }

            set {
                this.p_Data[index] = value;
                this.p_Valid[index] = true;
            }
        }

        private T[]     p_Data  = null;
        private bool[]  p_Valid = null;
    }
}
