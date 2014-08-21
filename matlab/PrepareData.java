import java.util.*;
import java.io.*;

public class PrepareData {
  private static final int LOGEVERY = 1000000;
  private static final int THRESHOLD = 5;

  private String dataPath;
  private String corpusFile;
  private String language;
  private List <Vocab> vocabs;
  private List <TreeNode> tree;

  public void makeIntFile() throws Exception {
    System.err.print("# Making int file from " + dataPath + "/" + corpusFile + "...");
    InputReader reader = new InputReader(dataPath + "/" + corpusFile);
    PrintWriter writer = new PrintWriter(new FileWriter(new File(dataPath + "/" + corpusFile + ".int")));

    Map <String, Integer> map = new HashMap <String, Integer> ();
    for (int i = 0; i < vocabs.size(); i++) {
      map.put(vocabs.get(i).word, vocabs.get(i).wordIndex);
    }

    for (int lineCount = 0; reader.hasNext(); ++lineCount) {
      String line = reader.nextLine();
      StringTokenizer tokenizer = new StringTokenizer(line);
      while (true) {
        String token = tokenizer.nextToken();
        writer.print(map.get(token));
        if (tokenizer.hasMoreTokens()) {
          writer.print(" ");
        } else {
          break;
        }
      }
      writer.println();
      if (lineCount % 100000 == 0) {
        System.err.print(".");
      }
    }

    System.err.println("done");
    writer.close();
  }

  int stackCount = 0;
  public int buildTreePath(int node, int from, int to, int sum, int wordIndex) {
    if (from >= to) {
      return node;
    }
    int mid = from;
    int prefixSum = vocabs.get(from).count;
    for (int i = from+1; i <= to; i++) {
      if (prefixSum + vocabs.get(i).count <= sum / 2) {
        prefixSum += vocabs.get(i).count;
      } else {
        mid = i - 1;
        break;
      }
    }
//    System.err.println(node + " " + from + " " + to + " ---> " + sum + " | " + mid + " " + prefixSum);
    if (wordIndex <= mid) {
//      System.err.println("left");
      if (tree.get(node).left == -1) {
        tree.add(new TreeNode());
        tree.get(tree.size() - 1).parent = node;
        tree.get(node).left = tree.size() - 1;
      }
      return buildTreePath(tree.get(node).left, from, mid, prefixSum, wordIndex);
    } else {
//      System.err.println("right");
      if (tree.get(node).right == -1) {
        tree.add(new TreeNode());
        tree.get(tree.size() - 1).parent = node;
        tree.get(node).right = tree.size() - 1;
      }
      return buildTreePath(tree.get(node).right, mid+1, to, sum-prefixSum, wordIndex);
    }
  }

  public void buildDictionary() throws Exception {
    System.err.print("# Building dictionary from file " + dataPath + "/" + corpusFile + "...");
    InputReader reader = new InputReader(dataPath + "/" + corpusFile);
    PrintWriter writer = new PrintWriter(new FileWriter(new File(dataPath + "/" + language + ".paths")));

    Map <String, Integer> wordCounts = new HashMap <String, Integer> ();
    int totalWordsCount = 0;
    for (int lineCount = 0; reader.hasNext(); ++lineCount) {
      String word = reader.next();
      Integer count = wordCounts.get(word);
      if (count == null) count = 0;
      wordCounts.put(word, count + 1);
      if (lineCount % LOGEVERY == 0) {
        System.err.print(".");
      }
      ++totalWordsCount;
    }

    vocabs = new ArrayList <Vocab> ();
    for (String word : wordCounts.keySet()) {
      vocabs.add(new Vocab(word, wordCounts.get(word)));
    }
    Collections.sort(vocabs);

    tree = new ArrayList <TreeNode> ();
    tree.add(new TreeNode());
    for (int i = 0; i < vocabs.size(); i++) {
      vocabs.get(i).wordIndex = i+1;
//      System.err.println("========================");
      int currNode = buildTreePath(0, 0, vocabs.size() - 1, totalWordsCount, i);
//      System.err.println(currNode);
      List <Integer> list = new ArrayList <Integer> ();
      for (; currNode >= 0; currNode = tree.get(currNode).parent) {
        list.add(currNode);
      }
      for (int j = list.size() - 1; j >= 0; j--) {
        vocabs.get(i).path.add(list.get(j) + 1);
      }
      List <Integer> dirs = new ArrayList <Integer> ();
      for (int j = 0; j < vocabs.get(i).path.size(); j++) {
        writer.print(vocabs.get(i).path.get(j));
        if (j < vocabs.get(i).path.size() - 1) {
          writer.print(" ");
          if (vocabs.get(i).path.get(j+1) == tree.get(vocabs.get(i).path.get(j)).left) {
            dirs.add(1);
          } else {
            dirs.add(-1);
          }
        } else {
          writer.println();
        }
      }
      for (int j = 0; j < dirs.size(); j++) {
        writer.print(dirs.get(j));
        if (j < dirs.size() - 1) {
          writer.print(" ");
        } else {
          writer.println();
        }
      }
    }

    System.out.println("done");
    writer.close();

    writer = new PrintWriter(new BufferedWriter(new FileWriter(dataPath + "/" + language + ".words")));
    for (int i = 0; i < vocabs.size(); i++) {
      writer.println(vocabs.get(i).word);
    }
    writer.close();
  }

  public void run() throws Exception {
    buildDictionary();
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

class TreeNode {
  public int parent, left, right;

  public TreeNode() {
    this.parent = this.left = this.right = -1;
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
//    if (this.count != other.count) {
//      return other.count - this.count;
//    }
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

