// Databricks notebook source
// Movie Summary Search Engine

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover
import scala.collection.mutable.WrappedArray

// COMMAND ----------

// Load files for metadata (movie titles), summaries, and user created search file.
val metadata = spark.read.option("sep", "\t").csv("/FileStore/tables/movie_metadata-ab497.tsv")
val summaries = sc.textFile("/FileStore/tables/plot_summaries.txt").map(_.toLowerCase)map(x => (x.split("	")(0), x.split("	")(1).split("""\W+""")))
val search_file = sc.textFile("/FileStore/tables/search.txt").collect()

val n = summaries.count() // Total number of summaries (documents). Will be used in tf-df calculation.

summaries.take(5) // Preview of data.

// COMMAND ----------

// Remove stop words from summaries. 
// Vectors RDD will represnt the document vectors to be used when calculating cosine similarity.
val df = summaries.toDF("id", "summary")
val remover = new StopWordsRemover().setInputCol("summary").setOutputCol("text")
val newDF = remover.transform(df).drop("summary")
val vectors = newDF.rdd.map(row => (row(0).toString, row(1).asInstanceOf[WrappedArray[String]].toArray))
vectors.take(10)

// COMMAND ----------

// Split arrays of words to individial words. 
// This creates an idf with (key, value) where key is the term and value is the document id.
val words = vectors.flatMap{ case(str, arr) => arr.map( (s) => (s, str))}
words.take(10)

// COMMAND ----------

// Get term frequencies using MapReduce: count the number of occurences of each term in each document.
val tf = words.map(x => (x, 1)).reduceByKey((x,y) => x + y).map{ case((term, id), count) => (term, (id, count))}
tf.take(10)

// COMMAND ----------

// Get document frequencies using MapReduce: count the number of occurecnes of each term in the whole corpus.
val df = words.groupByKey().map{case(term, arr) => (term, arr.toArray.distinct.count(t=>true))}
df.take(10)

// COMMAND ----------

// Method used to calculate tf-df.
// w = tf * log(n/ni)
// n = total number of documents 
// ni = number of documents that contain term
// tf = number of times term occurs in each document
def calc_tfidf(tf: Double, df: Double) : Double = {
  val idf: Double = n/df 
  val tfidf = tf * (math.log10(idf)/math.log10(2))  // w = tf * log(n/ni)
  return tfidf
}

// COMMAND ----------

// Get tf-idf for every term and document.
val tf_idf = df.join(tf).map{ case(term, (df, (id, tf))) => (term, id, calc_tfidf(tf, df))}
tf_idf.take(10)

// COMMAND ----------

// Method used to calculate cosine similarity using bag of words approach.
// sim(d1, d2) = (d1.d2) / (|d1||d2|)
// d1.d2 = dot product of d1 and d2 (number of same words)
// |d1| = total number of distinct words in d1 
def cosine_similarity (query: Array[String], document: Array[String]) : Double = {
  var num = query.toSet.intersect(document.toSet).count(x => true)   // Get intersection of query and document and count number of same words in each.
  return num / (math.sqrt(query.distinct.length) * math.sqrt(document.distinct.length))
}

// COMMAND ----------

// Method for searching given a single term using tf-df.
def search_word (search_term: String) : Array[org.apache.spark.sql.Row] = {
  val matches = tf_idf.filter{ case(term, id, tfidf) => term == search_term }   // Filter tf-idf to get instances that contain search term.
   .map{case(term,id,tfidf)=>(id,tfidf)}.sortBy(-_._2)  // Sort by descending tf-df value.
   .map{ case(id, tfidf) => id}.take(10)  // Get the top 10 results.
  val results = matches.flatMap(x => metadata.where($"_c0" === x).select("_c2").collect) // Get movie names of top 10.
  return results
}

// COMMAND ----------

// Method for searching given multiple terms using cosine similarity.
def search_phrase (query: String) : Array[org.apache.spark.sql.Row] = {
  var q = query.split(" ")
  var cos = vectors.map{ case(id, terms) => (id, cosine_similarity(q, terms))} //  // Calculate cosine similarity between query and each document.
    .sortBy(-_._2) .map{ case(id, cos) => id}.take(10) // Sort by descending similarity and get top 10 results.
  val results = cos.flatMap(x => metadata.where($"_c0" === x).select("_c2").collect) // Get movie names of top 10.
  return results
}

// COMMAND ----------

// Overall search function that calls appropriate search method based on input type.
def search(query: String): Array[org.apache.spark.sql.Row] = {
  if(query.contains(" ")) search_phrase(query)
  else search_word(query)
}

// COMMAND ----------

// Search database using search file.
val results = search_file.map(x => (x, search(x)))

// COMMAND ----------

// Display results for each search term or phrase.
results.foreach{case(query, result) => print(query + ": " + result.mkString(", ") + "\n\n")}
