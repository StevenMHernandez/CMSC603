import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KNNMain {
    public static void main(String[] args) throws Exception {
        // SEE: https://stackoverflow.com/questions/11041253/set-hadoop-system-user-for-client-embedded-in-java-webapp
        System.setProperty("HADOOP_USER_NAME", "hadoop");

        long startTime = System.nanoTime();

        Configuration conf = new Configuration();
        // conf.setLong(FileInputFormat.SPLIT_MAXSIZE, 4096);
        Job job = Job.getInstance(conf, "Word Count Example");
        job.setMapperClass(KNNMapper.class);
        job.setReducerClass(KNNReducer.class);

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(CandidateArrayWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, new Path(CONFIG.trainFile));
        FileOutputFormat.setOutputPath(job, new Path("hdfs://hadoop-master:9000/output/knn-out_".concat(Long.toString(System.currentTimeMillis())).concat(".txt")));

        boolean out = job.waitForCompletion(true);

        long endTime = System.nanoTime();
        System.out.println("Took " + ((float) endTime - startTime) / 1000000000 + " seconds");

        System.exit(out ? 0 : 1);
    }
}

