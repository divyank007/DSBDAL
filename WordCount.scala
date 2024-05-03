// This opens up an interactive shell
// Create a new input1.txt file
val input = sc.textFile("passage.txt")

// Create a new RDD by splitting the input RDD on the basis of space
val words = input.flatMap(x => x.split(" "))

// Create a new RDD by mapping each word to a tuple of (word, 1)
val counts = words.map(x => (x, 1))

// Create a new RDD by reducing the tuples by key
val reducedCounts = counts.reduceByKey((x, y) => x + y)

// Save the RDD to a file
reducedCounts.saveAsTextFile("output.txt")

// Print the contents of the file
reducedCounts.foreach(println)


*****************************************************************************************************
// Load the text file
val data = sc.textFile("input.txt")
data.collect

// Split each line into words
val splitdata = data.flatMap(line => line.split(" "))
splitdata.collect

// Map each word to a tuple (word, 1)
val mapdata = splitdata.map(word => (word, 1))
mapdata.collect

// Reduce by key to get word counts
val reducedata = mapdata.reduceByKey(_ + _)
reducedata.collect

println()
// Print the word frequencies directly in a clean format
reducedata.collect.foreach { case (word, count) =>
  println(s"$word\t$count")
}

********************************************************************************************************
// Split each line into words, considering both spaces and punctuation marks
val splitdata = data.flatMap(line => line.split("[\\s\\p{Punct}]+"))
splitdata.collect