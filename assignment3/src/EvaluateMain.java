import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;

public class EvaluateMain {
    public static void main(String[] args) throws Exception {
        System.setProperty("HADOOP_USER_NAME", "hadoop");

        String inputFile = CONFIG.testFile;
        String outputFile = "hdfs://hadoop-master:9000/output/knn-out_1543793777716.txt/part-r-00000";
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(CONFIG.testFile), conf);
        InputStream inTest = null;
        InputStream inPred = null;

        try {
            inTest = fs.open(new Path(inputFile));
            inPred = fs.open(new Path(outputFile));
            BufferedReader TS = new BufferedReader(new InputStreamReader(inTest));
            BufferedReader predictions = new BufferedReader(new InputStreamReader(inPred));

            float accuracy = butJustComputeTheAccuracyThough(TS, predictions);

            System.out.println(String.format("The KNN classifier accuracy was %.4f\n", accuracy));
        } finally {
            IOUtils.closeStream(inTest);
            IOUtils.closeStream(inPred);
        }
    }

    private static float butJustComputeTheAccuracyThough(BufferedReader TS, BufferedReader predictions) throws IOException {
        String test_l = TS.readLine();
        String pred_l = predictions.readLine();

        // ignore any useless lines starting with `@` from TS
        while (test_l.startsWith("@")) {
            test_l = TS.readLine();
        }

        int count = 0;
        int correctCount = 0;

        while (test_l != null) {
            count++;

            String[] splt = test_l.split(",");
            Double actualClass = Double.valueOf(splt[splt.length - 1]);
            Double predClass = Double.valueOf(pred_l.split("\t")[1]);

            if (actualClass.equals(predClass)) {
                correctCount++;
            }

            // afterwards, load next lines to keep the while loop going
            test_l = TS.readLine();
            pred_l = predictions.readLine();
        }

        return (float) correctCount / count;
    }
}
