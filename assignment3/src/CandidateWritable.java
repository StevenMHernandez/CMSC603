import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class CandidateWritable implements WritableComparable<CandidateWritable> {
    public double cl;
    public double distance;

    public void write(DataOutput out) throws IOException {
        out.writeDouble(cl);
        out.writeDouble(distance);
    }

    public void readFields(DataInput in) throws IOException {
        cl = in.readDouble();
        distance = in.readDouble();
    }

    @Override
    public int compareTo(CandidateWritable o) {
        return (Double.compare(this.distance, o.distance));
    }
}
