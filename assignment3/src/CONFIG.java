public class CONFIG {
//    public static final String trainFile = "hdfs://hadoop-master:9000/small.arff";
     static final String trainFile = "hdfs://hadoop-master:9000/medium.arff";
//     static final String trainFile = "hdfs://hadoop-master:9000/poker.arff"; // https://sci2s.ugr.es/keel/dataset.php?cod=194

    public static final String testFile = trainFile;

    public static final Integer k = 5;
}
