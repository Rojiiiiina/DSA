// question1.
// a.

// 
// public class MinimumCostMatchingOutfits { 

  

//     public static void main(String[] args) { 

//         int N = 3; 

//         int[][] price = {{14, 4, 11}, {11, 14, 3}, {14, 2, 10}}; 

         

//         int minCost = calculateMinimumCost(N, price); 

//         System.out.println("Minimum cost: " + minCost); 

//     } 

     

//     public static int calculateMinimumCost(int N, int[][] price) { 

//         // Initialize minimum costs for each color for the first person 

//         int minCost1 = price[0][0]; 

//         int minCost2 = price[0][1]; 

//         int minCost3 = price[0][2]; 

         

//         // Iterate through the remaining people 

//         for (int i = 1; i < N; i++) { 

//             // Calculate the minimum cost for the current person's clothing in each color 

//             int currentMinCost1 = Math.min(minCost2, minCost3) + price[i][0]; 

//             int currentMinCost2 = Math.min(minCost1, minCost3) + price[i][1]; 

//             int currentMinCost3 = Math.min(minCost1, minCost2) + price[i][2]; 

             

//             // Update the minimum costs for each color for the current person 

//             minCost1 = currentMinCost1; 

//             minCost2 = currentMinCost2; 

//             minCost3 = currentMinCost3; 

//         } 

         

//         // Return the overall minimum cost among the three colors for the last person 

//         return Math.min(minCost1, Math.min(minCost2, minCost3)); 

//     } 

// } 

// 1.b.
// public class MinCoinsForRiders { 

  

//     public static void main(String[] args) { 

//         int[] ratings = {1, 0, 2}; 

//         int minCoins = distributeCoins(ratings); 

//         System.out.println("Minimum number of coins required: " + minCoins); 

//     } 

     

//     public static int distributeCoins(int[] ratings) { 

//         int n = ratings.length; 

         

//         // Initialize arrays to store the number of coins each rider receives and 

//         // initialize all riders with 1 coin 

//         int[] coins = new int[n]; 

//         Arrays.fill(coins, 1); 

         

//         // Traverse the ratings from left to right to ensure higher-rated riders receive more coins 

//         for (int i = 1; i < n; i++) { 

//             if (ratings[i] > ratings[i - 1]) { 

//                 coins[i] = coins[i - 1] + 1; // Give more coins to higher-rated rider 

//             } 

//         } 

         

//         // Traverse the ratings from right to left to make sure the neighbors of higher-rated 

//         // riders receive appropriate coins as well 

//         for (int i = n - 2; i >= 0; i--) { 

//             if (ratings[i] > ratings[i + 1]) { 

//                 coins[i] = Math.max(coins[i], coins[i + 1] + 1); // Update if necessary 

//             } 

//         } 

         

//         // Calculate the total number of coins required 

//         int totalCoins = 0; 

//         for (int coinCount : coins) { 

//             totalCoins += coinCount; 

//         } 

         

//         return totalCoins; 

//     } 

// } 

// 2.a.
// public class LongestDecreasingSubsequence { 

  

//     public static void main(String[] args) { 

//         int[] nums = {8, 5, 4, 2, 1, 4, 3, 4, 3, 1, 15}; 

//         int k = 3; 

//         int longestLength = findLongestSubsequence(nums, k); 

//         System.out.println("Length of the longest subsequence: " + longestLength); 

//     } 

     

//     public static int findLongestSubsequence(int[] nums, int k) { 

//         int n = nums.length; 

         

//         // Create a dp array to store the length of the longest subsequence ending at each index 

//         int[] dp = new int[n]; 

//         dp[0] = 1; // The longest subsequence ending at the first index is 1 

         

//         int longestLength = 1; // Initialize the length of the longest subsequence 

         

//         // Traverse the array to find the longest subsequence for each index 

//         for (int i = 1; i < n; i++) { 

//             dp[i] = 1; // Minimum length of a subsequence is 1 

             

//             // Check all previous elements to find a valid element to extend the subsequence 

//             for (int j = 0; j < i; j++) { 

//                 if (nums[j] - nums[i] <= k) { 

//                     dp[i] = Math.max(dp[i], dp[j] + 1); // Update the longest subsequence length 

//                 } 

//             } 

             

//             longestLength = Math.max(longestLength, dp[i]); // Update the overall longest length 

//         } 

         

//         return longestLength; 

//     } 

// } 

// 2.b.
// import java.util.*; 

  

// public class RandomPortGenerator { 

//     private Set<Integer> blacklist; 

//     private Random random; 

//     private int whitelistedCount; 

  

//     public RandomPortGenerator(int k, int[] blacklisted_ports) { 

//         blacklist = new HashSet<>(); 

//         for (int port : blacklisted_ports) { 

//             blacklist.add(port); 

//         } 

//         random = new Random(); 

//         whitelistedCount = k - blacklist.size(); 

//     } 

  

//     public int get() { 

//         int randomPort = random.nextInt(whitelistedCount); 

//         if (blacklist.contains(randomPort)) { 

//             return getRandomWhitelistedPort(); 

//         } else { 

//             return randomPort; 

//         } 

//     } 

  

//     private int getRandomWhitelistedPort() { 

//         int candidate = random.nextInt(whitelistedCount); 

//         while (blacklist.contains(candidate)) { 

//             candidate = (candidate + 1) % whitelistedCount; 

//         } 

//         return candidate; 

//     } 

  

//     public static void main(String[] args) { 

//         int[] blacklisted_ports = {2, 3, 5}; 

//         RandomPortGenerator generator = new RandomPortGenerator(7, blacklisted_ports); 

  

//         List<Integer> outputs = new ArrayList<>(); 

//         outputs.add(null); 

//         for (int i = 0; i < 5; i++) { 

//             outputs.add(generator.get()); 

//         } 

  

//         System.out.println(outputs); 

//     } 

// } 

// 3.a.
// public class MaximumPointsFromTargets { 

  

//     public static void main(String[] args) { 

//         int[] a = {3, 1, 5, 8}; 

//         int maxPoints = getMaxPoints(a); 

//         System.out.println("Maximum points: " + maxPoints); 

//     } 

     

//     public static int getMaxPoints(int[] a) { 

//         int n = a.length; 

         

//         // Create a new array with padded 1's at the beginning and end 

//         int[] targets = new int[n + 2]; 

//         targets[0] = 1; 

//         targets[n + 1] = 1; 

//         System.arraycopy(a, 0, targets, 1, n); 

         

//         // Create a dp array to store the maximum points for shooting targets in each subarray 

//         int[][] dp = new int[n + 2][n + 2]; 

         

//         // Iterate over subarray lengths from 1 to n 

//         for (int len = 1; len <= n; len++) { 

//             // Iterate over subarray starting positions 

//             for (int i = 1; i + len - 1 <= n; i++) { 

//                 int j = i + len - 1; // Ending position of subarray 

//                 // Iterate over all possible split positions within the subarray 

//                 for (int k = i; k <= j; k++) { 

//                     dp[i][j] = Math.max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + targets[i - 1] * targets[k] * targets[j + 1]); 

//                 } 

//             } 

//         } 

         

//         return dp[1][n]; // Maximum points for the entire array 

//     } 

// } 

//   3.b.
//   public class BellmanFordWithMaxHeap { 

  

//     public static class Edge { 

//         int source; 

//         int destination; 

//         int weight; 

  

//         public Edge(int source, int destination, int weight) { 

//             this.source = source; 

//             this.destination = destination; 

//             this.weight = weight; 

//         } 

//     } 

  

//     public static int[] bellmanFord(int n, List<Edge> edges, int source) { 

//         int[] distance = new int[n]; 

//         Arrays.fill(distance, Integer.MAX_VALUE); 

//         distance[source] = 0; 

  

//         PriorityQueue<Edge> maxHeap = new PriorityQueue<>(Collections.reverseOrder()); 

//         maxHeap.add(new Edge(source, source, 0)); 

  

//         for (int i = 0; i < n - 1; i++) { 

//             while (!maxHeap.isEmpty()) { 

//                 Edge edge = maxHeap.poll(); 

//                 int u = edge.source; 

//                 int v = edge.destination; 

//                 int w = edge.weight; 

  

//                 if (distance[u] + w < distance[v]) { 

//                     distance[v] = distance[u] + w; 

//                 } 

//             } 

  

//             for (Edge edge : edges) { 

//                 maxHeap.add(edge); 

//             } 

//         } 

  

//         return distance; 

//     } 

  

//     public static void main(String[] args) { 

//         int n = 5; 

//         List<Edge> edges = new ArrayList<>(); 

//         edges.add(new Edge(0, 1, -1)); 

//         edges.add(new Edge(0, 2, 4)); 

//         edges.add(new Edge(1, 2, 3)); 

//         edges.add(new Edge(1, 3, 2)); 

//         edges.add(new Edge(1, 4, 2)); 

//         edges.add(new Edge(3, 2, 5)); 

//         edges.add(new Edge(3, 1, 1)); 

//         edges.add(new Edge(4, 3, -3)); 

  

//         int source = 0; 

//         int[] distance = bellmanFord(n, edges, source); 

  

//         System.out.println("Shortest distances from source " + source + ":"); 

//         for (int i = 0; i < n; i++) { 

//             System.out.println("Vertex " + i + ": " + distance[i]); 

//         } 

//     } 

// } 
// 4.a.
// import java.util.*; 

  

// public class MinimumStepsToCompleteTasks { 

//     public int minSteps(int n, int[][] prerequisites) { 

//         List<List<Integer>> graph = new ArrayList<>(); 

//         for (int i = 0; i <= n; i++) { 

//             graph.add(new ArrayList<>()); 

//         } 

  

//         int[] inDegree = new int[n + 1]; 

  

//         for (int[] prerequisite : prerequisites) { 

//             int x = prerequisite[0]; 

//             int y = prerequisite[1]; 

//             graph.get(x).add(y); 

//             inDegree[y]++; 

//         } 

  

//         Queue<Integer> queue = new LinkedList<>(); 

//         for (int i = 1; i <= n; i++) { 

//             if (inDegree[i] == 0) { 

//                 queue.offer(i); 

//             } 

//         } 

  

//         int steps = 0; 

//         while (!queue.isEmpty()) { 

//             int size = queue.size(); 

//             for (int i = 0; i < size; i++) { 

//                 int node = queue.poll(); 

//                 for (int neighbor : graph.get(node)) { 

//                     if (--inDegree[neighbor] == 0) { 

//                         queue.offer(neighbor); 

//                     } 

//                 } 

//             } 

//             steps++; 

//         } 

  

//         return steps == n ? steps : -1; 

//     } 

  

//     public static void main(String[] args) { 

//         MinimumStepsToCompleteTasks solution = new MinimumStepsToCompleteTasks(); 

  

//         int n = 3; 

//         int[][] prerequisites = {{1, 3}, {2, 3}}; 

//         int minSteps = solution.minSteps(n, prerequisites); 

  

//         System.out.println("Minimum steps needed: " + minSteps); // Output: 2 

//     } 

// }  
// 4.b.
// class TreeNode { 

//     int val; 

//     TreeNode left; 

//     TreeNode right; 

     

//     TreeNode(int val) { 

//         this.val = val; 

//     } 

// } 

  

// public class BrothersInBinaryTree { 

//     private int depthX = -1; 

//     private int depthY = -1; 

//     private TreeNode parentX = null; 

//     private TreeNode parentY = null; 

  

//     public boolean areBrothers(TreeNode root, int x, int y) { 

//         findDepthAndParent(root, x, y, null, 0); 

//         return depthX == depthY && parentX != parentY; 

//     } 

  

//     private void findDepthAndParent(TreeNode node, int x, int y, TreeNode parent, int depth) { 

//         if (node == null) { 

//             return; 

//         } 

         

//         if (node.val == x) { 

//             depthX = depth; 

//             parentX = parent; 

//         } 

         

//         if (node.val == y) { 

//             depthY = depth; 

//             parentY = parent; 

//         } 

         

//         findDepthAndParent(node.left, x, y, node, depth + 1); 

//         findDepthAndParent(node.right, x, y, node, depth + 1); 

//     } 

  

//     public static void main(String[] args) { 

//         BrothersInBinaryTree solution = new BrothersInBinaryTree(); 

         

//         // Create the binary tree: [1,2,3,4] 

//         TreeNode root = new TreeNode(1); 

//         root.left = new TreeNode(2); 

//         root.right = new TreeNode(3); 

//         root.left.left = new TreeNode(4); 

  

//         int x = 4; 

//         int y = 3; 

//         boolean result = solution.areBrothers(root, x, y); 

         

//         System.out.println("Are nodes " + x + " and " + y + " brothers? " + result); // Output: false 

//     } 

// } 
// 5.a.
// import java.util.Random; 

  

// public class HillClimbing { 

  

//     public static double function(double x) { 

//         // Example function: f(x) = -x^2 + 4x 

//         return -x * x + 4 * x; 

//     } 

  

//     public static double hillClimbing(double initialX, double stepSize, int maxIterations) { 

//         double currentX = initialX; 

//         double currentValue = function(currentX); 

  

//         for (int i = 0; i < maxIterations; i++) { 

//             double newX = currentX + stepSize; 

//             double newValue = function(newX); 

  

//             if (newValue > currentValue) { 

//                 currentX = newX; 

//                 currentValue = newValue; 

//             } else { 

//                 stepSize *= -0.5; // Reduce step size if no improvement is found 

//             } 

//         } 

  

//         return currentX; 

//     } 

  

//     public static void main(String[] args) { 

//         double initialX = 0.0; // Starting point 

//         double stepSize = 0.1; // Initial step size 

//         int maxIterations = 100; // Maximum number of iterations 

  

//         double result = hillClimbing(initialX, stepSize, maxIterations); 

//         System.out.println("Maximum value found at x = " + result); 

//         System.out.println("Maximum value = " + function(result)); 

//     } 

// } 

// 5.b.
// import java.util.*; 

  

// public class ReorientConnections { 

//     public int minReorder(int n, int[][] connections) { 

//         List<List<Integer>> graph = new ArrayList<>(); 

//         for (int i = 0; i < n; i++) { 

//             graph.add(new ArrayList<>()); 

//         } 

         

//         for (int[] connection : connections) { 

//             graph.get(connection[0]).add(connection[1]); // original direction 

//             graph.get(connection[1]).add(-connection[0]); // reverse direction 

//         } 

         

//         boolean[] visited = new boolean[n]; 

//         return dfs(0, graph, visited); 

//     } 

     

//     private int dfs(int node, List<List<Integer>> graph, boolean[] visited) { 

//         visited[node] = true; 

//         int count = 0; // count of edges that need to be reversed 

         

//         for (int neighbor : graph.get(node)) { 

//             if (!visited[Math.abs(neighbor)]) { 

//                 if (neighbor > 0) { // original direction 

//                     count += dfs(neighbor, graph, visited); 

//                 } else { // reverse direction 

//                     count++; 

//                     count += dfs(-neighbor, graph, visited); 

//                 } 

//             } 

//         } 

         

//         return count; 

//     } 

  

//     public static void main(String[] args) { 

//         ReorientConnections solution = new ReorientConnections(); 

  

//         int n = 6; 

//         int[][] connections = {{0, 1}, {1, 3}, {2, 3}, {4, 0}, {4, 5}}; 

//         int minChanges = solution.minReorder(n, connections); 

  

//         System.out.println("Minimum changes required: " + minChanges); // Output: 3 

//     } 

// } 

// 6.
// import java.util.Arrays; 

// import java.util.concurrent.RecursiveAction; 

  

// public class ParallelMergeSort { 

  

//     public static void parallelMergeSort(int[] arr, int threadCount) { 

//         ParallelMergeSortTask task = new ParallelMergeSortTask(arr, 0, arr.length - 1, threadCount); 

//         task.invoke(); 

//     } 

  

//     public static void merge(int[] arr, int left, int mid, int right) { 

//         int[] leftArray = Arrays.copyOfRange(arr, left, mid + 1); 

//         int[] rightArray = Arrays.copyOfRange(arr, mid + 1, right + 1); 

  

//         int i = 0, j = 0, k = left; 

  

//         while (i < leftArray.length && j < rightArray.length) { 

//             if (leftArray[i] <= rightArray[j]) { 

//                 arr[k++] = leftArray[i++]; 

//             } else { 

//                 arr[k++] = rightArray[j++]; 

//             } 

//         } 

  

//         while (i < leftArray.length) { 

//             arr[k++] = leftArray[i++]; 

//         } 

  

//         while (j < rightArray.length) { 

//             arr[k++] = rightArray[j++]; 

//         } 

//     } 

  

//     static class ParallelMergeSortTask extends RecursiveAction { 

//         private int[] arr; 

//         private int left, right; 

//         private int threadCount; 

  

//         public ParallelMergeSortTask(int[] arr, int left, int right, int threadCount) { 

//             this.arr = arr; 

//             this.left = left; 

//             this.right = right; 

//             this.threadCount = threadCount; 

//         } 

  

//         @Override 

//         protected void compute() { 

//             if (left < right) { 

//                 if (threadCount > 1) { 

//                     int mid = (left + right) / 2; 

//                     ParallelMergeSortTask leftTask = new ParallelMergeSortTask(arr, left, mid, threadCount / 2); 

//                     ParallelMergeSortTask rightTask = new ParallelMergeSortTask(arr, mid + 1, right, threadCount / 2); 

  

//                     invokeAll(leftTask, rightTask); 

  

//                     merge(arr, left, mid, right); 

//                 } else { 

//                     Arrays.sort(arr, left, right + 1); 

//                 } 

//             } 

//         } 

//     } 

  

//     public static void main(String[] args) { 

//         int[] arr = {5, 3, 8, 1, 2, 9, 4, 7, 6}; 

//         int threadCount = 4; // Set the number of threads to use 

  

//         System.out.println("Original array: " + Arrays.toString(arr)); 

  

//         parallelMergeSort(arr, threadCount); 

  

//         System.out.println("Sorted array: " + Arrays.toString(arr)); 

//     } 

// } 

// 7.
// import javafx.application.Application; 

// import javafx.scene.Scene; 

// import javafx.scene.canvas.Canvas; 

// import javafx.scene.canvas.GraphicsContext; 

// import javafx.scene.control.*; 

// import javafx.scene.image.Image; 

// import javafx.scene.layout.BorderPane; 

// import javafx.scene.layout.VBox; 

// import javafx.stage.Stage; 

 
 

// public class SocialNetworkGraphApp extends Application { 

//     public static void main(String[] args) { 

//         launch(args); 

//     } 

 
 

//     @Override 

//     public void start(Stage primaryStage) { 

//         BorderPane root = new BorderPane(); 

 
 

//         // Canvas for drawing the graph 

//         Canvas canvas = new Canvas(800, 600); 

//         GraphicsContext gc = canvas.getGraphicsContext2D(); 

 
 

//         // Toolbar with buttons 

//         VBox toolbar = new VBox(); 

//         Button selectModeButton = new Button("Select Mode"); 

//         Button addNodeButton = new Button("Add Node"); 

//         Button addEdgeButton = new Button("Add Edge"); 

//         toolbar.getChildren().addAll(selectModeButton, addNodeButton, addEdgeButton); 

 
 

//         root.setLeft(toolbar); 

//         root.setCenter(canvas); 

 
 

//         Scene scene = new Scene(root, 1000, 800); 

//         primaryStage.setTitle("Social Network Graph"); 

//         primaryStage.setScene(scene); 

//         primaryStage.show(); 

//     } 

 
 

//     // You can add methods for loading data from a file, drawing nodes and edges, handling interactions, etc. 

 
 

//     public class Node { 

//         private String userName; 

//         private Image profilePicture; 

//         private int followersCount; 

 
 

//         // Constructor, getters, and setters 

//     } 

 
 

//     public class Edge { 

//         private Node source; 

//         private Node target; 

//         private int connectionStrength; 

 
 

//         // Constructor, getters, and setters 

//     } 

// } 
