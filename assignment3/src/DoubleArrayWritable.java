import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;

public class DoubleArrayWritable extends ArrayWritable {
    public DoubleArrayWritable() {
        super(DoubleWritable.class);
    }

    public DoubleArrayWritable(double[] values) {
        super(DoubleWritable.class, convert(values));
    }

    private static DoubleWritable[] convert(double[] values) {
        DoubleWritable[] mappedValues = new DoubleWritable[values.length];
        for (int i = 0; i < values.length; i++) {
            mappedValues[i] = new DoubleWritable(values[i]);
        }

        return mappedValues;
    }
}