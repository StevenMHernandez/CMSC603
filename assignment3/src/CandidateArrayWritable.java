import org.apache.hadoop.io.ArrayWritable;

public class CandidateArrayWritable extends ArrayWritable {
    public CandidateArrayWritable() {
        super(CandidateWritable.class);
    }

    public CandidateArrayWritable(Candidate[] values) {
        super(CandidateWritable.class, convert(values));
    }

    private static CandidateWritable[] convert(Candidate[] values) {
        CandidateWritable[] mappedValues = new CandidateWritable[values.length];
        for (int i = 0; i < values.length; i++) {
            mappedValues[i] = new CandidateWritable();
            mappedValues[i].cl = values[i].cl;
            mappedValues[i].distance = values[i].distance;
        }

        return mappedValues;
    }
}