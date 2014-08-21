import java.util.*;
import java.io.*;


public class PrepareData {
  private static final int THRESHOLD = 5;

  private String dataPath;
  private String corpusFile;
  private String language;
  private int vocabSize;
  private Vocab[] vocab;

  public void buildTreePath(int node, int tLeft, int tRight, int wordIndex, List <Integer> list) {
    list.add(node);
    if (tLeft >= tRight) {
      return;
    }
    int leftChild = 2 * node;
    int rightChild = 2 * node + 1;
    int tMid = (tLeft + tRight) / 2;
    if (wordIndex <= tMid) {
      buildTreePath(leftChild, tLeft, tMid, wordIndex, list);
    } else {
      buildTreePath(rightChild, tMid + 1, tRight, wordIndex, list);
    }
  }

  public void makeIntFile() throws Exception {
    System.err.print("# Making int file from " + dataPath + "/" + corpusFile + "...");
    InputReader reader = new InputReader(dataPath + "/" + corpusFile);
    PrintWriter writer = new PrintWriter(new FileWriter(new File(dataPath + "/" + corpusFile + ".int")));

    Map <String, Vocab> map = new HashMap <String, Vocab> ();
    for (int i = 0; i < vocab.length; i++) {
      map.put(vocab[i].word, vocab[i]);
    }

    for (int lineCount = 0; reader.hasNext(); ++lineCount) {
      String line = reader.nextLine();
      StringTokenizer tokenizer = new StringTokenizer(line);
      while (true) {
        String token = tokenizer.nextToken();
        writer.print(map.get(token).wordIndex);
        if (tokenizer.hasMoreTokens()) {
          writer.print(" ");
        } else {
          writer.println();
          break;
        }
      }
      if (lineCount % 100000 == 0) {
        System.err.print(".");
      }
    }

    System.err.println("done");
    writer.close();
  }

  public void run() throws Exception {
    System.err.print("# Building dictionary from file " + dataPath + "/" + corpusFile + "...");
    InputReader reader = new InputReader(dataPath + "/" + corpusFile);
    PrintWriter writer = new PrintWriter(new FileWriter(new File(dataPath + "/" + language + ".paths")));

    int totalWordsCount = 0;
    Map <String, Integer> wordCounts = new HashMap <String, Integer> ();
    for (int lineCount = 0; reader.hasNext(); ++lineCount) {
      String word = reader.next();
      Integer count = wordCounts.get(word);
      if (count == null) count = 0;
      wordCounts.put(word, count + 1);
      ++totalWordsCount;
      if (lineCount % 1000000 == 0) {
        System.err.print(".");
      }
    }

    vocabSize = 0;
    vocab = new Vocab[wordCounts.size()];
    for (String word : wordCounts.keySet()) {
      vocab[vocabSize++] = new Vocab(word, wordCounts.get(word));
    }
    Arrays.sort(vocab);

    int actualVocabSize = 0;
    for (int i = 0; i < vocabSize; i++) {
      if (vocab[i].count < THRESHOLD) {
        actualVocabSize = i;
        break;
      }
    }
    for (int i = actualVocabSize; i < vocabSize; i++) {
      vocab[i].wordIndex = actualVocabSize + 1;
    }
    vocabSize = actualVocabSize + 1;
    for (int i = 0; i < vocabSize; i++) {
      vocab[i].wordIndex = i + 1;
      buildTreePath(1, 1, vocabSize, i+1, vocab[i].path);
//      writer.print(vocab[i].word + " ");
      for (int j = 0; j < vocab[i].path.size(); j++) {
        writer.print(vocab[i].path.get(j));
        if (j < vocab[i].path.size() - 1) {
          writer.print(" ");
        } else {
          writer.println();
        }
      }
    }

    System.out.println("done");
    writer.close();

    writer = new PrintWriter(new BufferedWriter(new FileWriter(dataPath + "/" + language + ".words")));
    for (int i = 0; i < vocabSize; i++) {
      writer.println(vocab[i].word);
    }
    writer.close();

    makeIntFile();
  }


  public PrepareData(String[] args) throws Exception {
    for (int i = 0; i < args.length; i += 2) {
      if (args[i].equals("-dataPath"))          dataPath = args[i + 1];
      else if (args[i].equals("-corpusFile"))   corpusFile = args[i + 1];
      else if (args[i].equals("-language"))     language = args[i + 1];
      else throw new Exception(String.format("Unknown argument %", args[i]));
    }
  }

  public static void main(String[] args) throws Exception {
    new PrepareData(args).run();
  }
}

class Vocab implements Comparable <Vocab> {
  public String word;
  public int wordIndex;
  public int count;
  public List <Integer> path;
  
  public Vocab(String word, int count) {
    this.word = word;
    this.count = count;
    this.path = new ArrayList <Integer> ();
  }
  
  public int compareTo(Vocab other) {
    if (this.count != other.count) {
      return other.count - this.count;
    }
    return this.word.compareTo(other.word);
  }
  
  public String toString() {
    return word + " " + wordIndex + " " + count + " " + path;
  }
}

class InputReader {
  private BufferedReader reader;
  private StringTokenizer tokenizer;
  
  public InputReader(String fileName) {
    try {
      reader = new BufferedReader(new FileReader(fileName));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    tokenizer = null;
  }
  
  public InputReader(InputStream stream) {
    reader = new BufferedReader(new InputStreamReader(stream));
    tokenizer = null;
  }
  
  public boolean hasNext() {
    while (tokenizer == null || !tokenizer.hasMoreTokens()) {
      try {
        String nextLine = reader.readLine();
        if (nextLine == null) {
          return false;
        } else {
          tokenizer = new StringTokenizer(nextLine);
        }
      } catch (IOException e) {
        return false;
      }
    }
    return true;
  }
  
  public String next() {
    hasNext();
    return tokenizer.nextToken();
  }
  
  public String nextLine() {
    hasNext();
    return tokenizer.nextToken("\n\r");
  }
  
  public int nextInt() {
    return Integer.parseInt(next());
  }
  
  public long nextLong() {
    return Long.parseLong(next());
  }
  
  public double nextDouble() {
    return Double.parseDouble(next());
  }
}

