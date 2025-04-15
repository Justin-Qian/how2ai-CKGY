from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

class ConceptCluster:
    def __init__(self, id, keywords):
        """
        Represents a group of semantically related concepts.

        :param id: Unique identifier for the cluster
        :param keywords: A list of terms associated with this concept
        """
        self.id = id  # Unique identifier for the cluster
        self.keywords = set(keywords)  # Set of terms in this cluster
        self.embeddings = {kw: model.encode(kw) for kw in keywords}  # Precompute embeddings for each keyword
        self.score = 0  # Knowledge score: positive indicates "known", negative indicates "not known"
        self.history = []  # Record of updates: (doc_id, delta, highlight)

    def add_keyword(self, keyword):
        """
        Adds a new keyword to the cluster and computes its embedding.

        :param keyword: The new keyword to add
        """
        self.keywords.add(keyword)
        self.embeddings[keyword] = model.encode(keyword)

class MentalKnowledgeGraph:
    def __init__(self):
        """
        Represents a mental knowledge graph that tracks user understanding of concepts.
        """
        self.clusters = []  # List of ConceptCluster objects

    def calculate_similarity(self, embedding1, embedding2):
        """
        Computes the semantic similarity between two embeddings.

        :param embedding1: First embedding
        :param embedding2: Second embedding
        :return: Semantic similarity score (0 to 1)
        """
        return util.cos_sim(embedding1, embedding2).item()

    def find_closest_cluster(self, highlight, threshold=0.4):
        """
        Finds the closest matching cluster for a given highlight based on semantic similarity.

        :param highlight: The user-highlighted text span
        :param threshold: Similarity threshold to consider a match
        :return: Closest matching ConceptCluster or None if no match is found
        """
        best_match = None
        best_similarity = 0
        best_keyword = None
        highlight_embedding = model.encode(highlight)

        for cluster in self.clusters:
            for keyword in cluster.keywords:
                similarity = self.calculate_similarity(highlight_embedding, cluster.embeddings[keyword])
                if similarity > best_similarity:
                    best_match = cluster
                    best_similarity = similarity
                    best_keyword = keyword  # Store the best matching keyword for debugging
        print(f"Best match similarity for '{highlight}': {best_similarity} with keyword '{best_keyword}'")
        return best_match if best_similarity >= threshold else None

    def update(self, doc_id, highlight, tag):
        """
        Updates the scores of concept clusters based on user feedback.
        If no matching cluster is found, creates a new cluster.

        :param doc_id: The document ID being processed
        :param highlight: The highlighted text from the user
        :param tag: Either "know" or "notknow", representing user understanding
        :param threshold: Similarity threshold for matching clusters
        """
        delta = 1 if tag == "know" else -1  # Positive for "know", negative for "notknow"
        matched_cluster = self.find_closest_cluster(highlight)

        if matched_cluster:
            # Update the existing cluster
            matched_cluster.score += delta
            matched_cluster.history.append((doc_id, delta, highlight))
            matched_cluster.add_keyword(highlight)  # Add the new word to the cluster
        else:
            # Create a new cluster
            new_cluster_id = len(self.clusters) + 1
            new_cluster = ConceptCluster(id=new_cluster_id, keywords=[highlight])
            new_cluster.score += delta
            new_cluster.history.append((doc_id, delta, highlight))
            self.clusters.append(new_cluster)


# Initialize the mental model
mkg = MentalKnowledgeGraph()

# Define concept clusters
mkg.clusters.append(ConceptCluster(
    id=1, keywords=["search", "loop through"] # "traverse" "scan" is a synonym for "search" and "loop through"
))
mkg.clusters.append(ConceptCluster(
    id=2, keywords=["arrange", "reorder"] # "sort" "order" is a synonym for "arrange" and "reorder"
))
mkg.clusters.append(ConceptCluster(
    id=3, keywords=["construct", "build", "initialize", "set up"]
))

# Update the model with user input
mkg.update(doc_id=1, highlight="traverse", tag="notknow")
mkg.update(doc_id=2, highlight="scan", tag="know")
mkg.update(doc_id=3, highlight="sort", tag="notknow")
mkg.update(doc_id=4, highlight="order", tag="know")

# Print current cluster status
for c in mkg.clusters:
    print(f"[{c.id}] Score: {c.score} | Keywords: {c.keywords} ")
