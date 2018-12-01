import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashMap;

public class KNNReducer extends Reducer<LongWritable, CandidateArrayWritable, LongWritable, DoubleWritable> {
    @Override
    public void reduce(LongWritable key, Iterable<CandidateArrayWritable> values, Context context) throws IOException, InterruptedException {
        HashMap<Double, Integer> votes = new HashMap<>();

        values.forEach(candidateArrayWritable -> {
            Writable[] candidates = candidateArrayWritable.get();
            for (Writable w : candidates) {
                CandidateWritable c = (CandidateWritable) w;
                if (!votes.containsKey(c.cl)) {
                    votes.put(c.cl, 0);
                }

                votes.put(c.cl, votes.get(c.cl) + 1);
            }
        });

        final int[] maxVotes = {-1};
        final double[] maxVotesCl = {-1};

        votes.forEach((cl, voteCount) -> {
            if (voteCount > maxVotes[0]) {
                maxVotes[0] = voteCount;
                maxVotesCl[0] = cl;
            }
        });

        context.write(key, new DoubleWritable(maxVotesCl[0]));
    }
}