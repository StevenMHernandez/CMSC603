import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Arrays;
import java.util.Random;

public class KNNMapper extends Mapper<Object, Text, Object, CandidateArrayWritable> {
    private LongWritable identifierKey = new LongWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);

        identifierKey.set((new Random()).nextInt());
    }

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        long splitLength = ((FileSplit) context.getInputSplit()).getLength();

        // each call to map is only 1 sample, so we don't need to
        if (!value.toString().startsWith("@")) {
            FileSystem fs = FileSystem.get(URI.create(CONFIG.testFile), context.getConfiguration());
            InputStream in = null;
            try {
                in = fs.open(new Path(CONFIG.testFile));
                BufferedReader TS = new BufferedReader(new InputStreamReader(in));

                double[] trValue = Arrays.stream(value.toString().split(",")).mapToDouble(Double::parseDouble).toArray();

                Candidate[] candidates = knn(trValue, TS, CONFIG.k);

                CandidateArrayWritable output = new CandidateArrayWritable(candidates);

                LongWritable out = new LongWritable((long)  Math.round(context.getProgress() * splitLength));
                context.write(out, output);
            } finally {
                IOUtils.closeStream(in);
            }
        }
    }

    private double euclideanDistance(double[] x1, double[] x2) {
        double sum = 0;

        // NOTE: last element in double is consider the class
        for (int i = 0; i < x1.length - 1; i++) {
            sum += Math.pow(x1[i] - x2[i], 2);
        }

        return Math.sqrt(sum);
    }

    private Candidate[] knn(double[] x, BufferedReader TS, Integer k) throws IOException {
        double[] candidateClasses = new double[k];
        double[] candidateDistances = new double[k];

        for (int i = 0; i < k; i++) {
            candidateClasses[i] = -1;
            candidateDistances[i] = 99999999;
        }

        // For each element in TS, figure out
        while (true) {
            String l = TS.readLine();
            if (l == null) {
                break;
            }
            if (!l.startsWith("@")) {
                double[] tsValue = Arrays.stream(l.split(",")).mapToDouble(Double::parseDouble).toArray();

                double classValue = tsValue[tsValue.length - 1];
                double distance = euclideanDistance(x, tsValue);

                if (distance > 0) {
                    for (int i = 0; i < k; i++) {
                        if (candidateDistances[i] > distance) {
                            double tmpClassValue = candidateClasses[i];
                            double tmpDistance = candidateDistances[i];

                            candidateClasses[i] = classValue;
                            candidateDistances[i] = distance;

                            classValue = tmpClassValue;
                            distance = tmpDistance;
                        }
                    }
                }
            }
        }

        Candidate[] candidates = new Candidate[k];

        for (int i = 0; i < k; i++) {
            candidates[i] = new Candidate(candidateClasses[i], candidateDistances[i]);
        }

        return candidates;
    }
}
