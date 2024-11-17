import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from py2neo import Graph
import re
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans
import openai

# Load environment variables
load_dotenv()

# Set up API and database credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Global variable for Neo4j driver
driver = None

# Function to initialize Neo4j connection
def init_neo4j():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("Neo4j connection initialized.")

# Function to close Neo4j connection
def close_neo4j_connection():
    global driver
    if driver:
        driver.close()
        driver = None
        print("Neo4j connection closed.")

# Function to parse extracted topics from OpenAI response
def parse_extracted_topics(response_text):
    # Split the response into lines
    lines = response_text.strip().split('\n')
    topics = []

    for line in lines:
        # Match lines that start with a number and a period
        match = re.match(r'^\d+\.\s*(.*)', line)
        if match:
            topic = match.group(1).strip()
            topics.append(topic)
        else:
            # Handle bullet points or other formats
            match = re.match(r'^[\-\*]\s*(.*)', line)
            if match:
                topic = match.group(1).strip()
                topics.append(topic)
    return topics

# Function to filter topics to remove unnecessary words and improve relevance
def filter_extracted_topics(topics):
    # Define a set of stopwords and generic terms to be removed
    stopwords = {
        "to", "in", "and", "of", "for", "on", "with", "a", "an", "is", "the", "from",
        "this", "these", "related", "include", "providing", "creating", "based", "pursuing",
        "cover", "main", "topics", "introduction", "concepts", "management", "systems",
        "data", "learning", "studies", "case", "applications", "roadmap"
    }
    filtered_topics = []

    for topic in topics:
        # Remove punctuation and convert to lowercase
        topic_clean = re.sub(r'[^\w\s]', '', topic).strip().lower()
        if topic_clean not in stopwords and len(topic_clean) > 1:
            filtered_topics.append(topic_clean.title())  # Convert back to title case

    # Remove duplicates and return the list
    return list(set(filtered_topics))

# Function to validate topics with Neo4j
def validate_topics_with_neo4j(graph, topics):
    valid_topics = []
    for topic in topics:
        # Clean the topic to remove special characters and convert to lowercase
        topic_clean = re.sub(r'[^\w\s]', '', topic).lower()
        keywords = topic_clean.split()
        # Build a case-insensitive query
        query_conditions = []
        params = {}
        for i, kw in enumerate(keywords):
            param_kw = f"kw{i}"
            query_conditions.append(f"toLower(c.title) CONTAINS ${param_kw} OR toLower(c.description) CONTAINS ${param_kw}")
            params[param_kw] = kw
        query = f"""
            MATCH (c:Course)
            WHERE {" OR ".join(query_conditions)}
            RETURN COUNT(c) > 0 AS topic_exists
        """
        result = graph.run(query, **params).data()
        if result and result[0].get('topic_exists', False):
            valid_topics.append(topic_clean)  # Use the cleaned topic

    if not valid_topics:
        print("No topics validated, using original extracted topics as a fallback.")
        valid_topics = [re.sub(r'[^\w\s]', '', topic).lower() for topic in topics]

    return valid_topics

# Function to get embeddings using OpenAI API
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Main function to generate the learning roadmap
def generate_learning_roadmap(query):
    try:
        # Initialize the graph connection
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Extract relevant topics dynamically using OpenAI API
        system_prompt = "Extract the main topics for creating a learning roadmap from the following query. Focus on broad technical subjects or other fields as mentioned in the query. Return the topics as a numbered list, one topic per line."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )

        response_text = response['choices'][0]['message']['content']
        print(f"OpenAI response:\n{response_text}")

        # Parse the extracted topics properly
        extracted_topics = parse_extracted_topics(response_text)
        print(f"Extracted topics: {extracted_topics}")

        # Filter the topics to remove unnecessary words
        filtered_topics = filter_extracted_topics(extracted_topics)
        print(f"Filtered core topics: {filtered_topics}")

        # Validate topics with Neo4j
        core_topics = validate_topics_with_neo4j(graph, filtered_topics)
        print(f"Validated core topics: {core_topics}")

        if not core_topics:
            print("No valid topics found after validation.")
            return None, None

        # Build the Cypher query to fetch courses
        course_query_conditions = []
        params = {}
        param_counter = 0

        for topic in core_topics:
            param_name = f'kw{param_counter}'
            course_query_conditions.append(
                f"(toLower(c.title) CONTAINS ${param_name} OR toLower(c.description) CONTAINS ${param_name})"
            )
            params[param_name] = topic.lower()
            param_counter += 1

        courses_query = f"""
            MATCH (c:Course)
            WHERE {" OR ".join(course_query_conditions)}
            RETURN DISTINCT c
            LIMIT 100
        """

        result = graph.run(courses_query, **params)

        courses = []
        course_dict = {}
        added_course_ids = set()

        for record in result:
            course_node = record['c']
            course_id = str(course_node['id']).strip()

            if course_id in added_course_ids:
                continue  # Skip if already processed

            course_info = {
                'id': course_id,
                'title': course_node['title'],
                'description': course_node.get('description', ''),
                'prerequisites': []
            }
            courses.append(course_info)
            course_dict[course_id] = course_info
            added_course_ids.add(course_id)

        if not courses:
            print("No courses found or no valid data to render")
            return None, None

        # Retrieve prerequisites for each course
        for course in courses:
            course_id = course['id']
            prereq_query = """
                MATCH (c:Course {id: $course_id})-[:REQUIRES]->(prereq:Course)
                RETURN prereq.id AS prereq_id
            """
            prereq_result = graph.run(prereq_query, course_id=course_id)
            prereq_ids = [str(record['prereq_id']).strip() for record in prereq_result if str(record['prereq_id']).strip() in added_course_ids]
            course['prerequisites'] = prereq_ids

        # Compute embeddings for courses
        for course in courses:
            text = course['title'] + " " + course['description']
            course['embedding'] = get_embedding(text)

        # Check if there are enough courses to perform clustering
        if len(courses) >= 2:
            embeddings = [course['embedding'] for course in courses]
            embeddings_array = np.array(embeddings)

            num_clusters = min(len(courses), 5)  # Adjust the number of clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(embeddings_array)

            # Assign cluster labels
            for idx, course in enumerate(courses):
                course['cluster'] = int(kmeans.labels_[idx])
        else:
            # Assign default cluster label when not enough courses
            for course in courses:
                course['cluster'] = 0

        # Generate cluster labels using OpenAI
        cluster_labels = {}
        for cluster_id in set(course['cluster'] for course in courses):
            cluster_courses = [course for course in courses if course['cluster'] == cluster_id]
            cluster_texts = " ".join(course['title'] + " " + course['description'] for course in cluster_courses)
            prompt = f"Provide a concise label or title that summarizes the following topics:\n\n{cluster_texts}\n\nLabel:"
            
            # Use ChatCompletion API with gpt-3.5-turbo
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise labels for groups of topics."},
                {"role": "user", "content": prompt}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=20,
                temperature=0.5
            )
            
            label = response['choices'][0]['message']['content'].strip()
            cluster_labels[cluster_id] = label

        # Compute cluster centroids
        cluster_embeddings = {}
        for cluster_id in set(course['cluster'] for course in courses):
            cluster_courses = [course for course in courses if course['cluster'] == cluster_id]
            embeddings = [course['embedding'] for course in cluster_courses]
            centroid = np.mean(embeddings, axis=0)
            cluster_embeddings[cluster_id] = centroid

        # Calculate similarities between clusters
        cluster_similarities = []
        cluster_ids = list(cluster_embeddings.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster_id_a = cluster_ids[i]
                cluster_id_b = cluster_ids[j]
                centroid_a = cluster_embeddings[cluster_id_a]
                centroid_b = cluster_embeddings[cluster_id_b]
                similarity = cosine_similarity(centroid_a, centroid_b)
                cluster_similarities.append((cluster_id_a, cluster_id_b, similarity))

        # Create the network graph using PyVis
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#f0f0f0",
            font_color="black"
        )

        # Adjust layout settings
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.005,
                    "springLength": 230,
                    "springConstant": 0.18
                },
                "minVelocity": 0.75,
                "timestep": 0.35,
                "adaptiveTimestep": true
            }
        }
        """)

        # Add cluster label nodes
        for cluster_id, label in cluster_labels.items():
            cluster_node_id = f"cluster_{cluster_id}"
            net.add_node(
                cluster_node_id,
                label=label,
                shape='box',
                color='lightblue',
                font={'size': 20, 'color': 'black'},
                level=0  # Position at the top
            )
            cluster_labels[cluster_id] = cluster_node_id  # Store the node ID for connections

        # Add course nodes and connect them to cluster label nodes
        for course in courses:
            net.add_node(
                course['id'],
                label=course['title'],
                title=course['description'],
                group=course['cluster']
            )
            # Connect course node to cluster label node
            cluster_node_id = cluster_labels[course['cluster']]
            net.add_edge(cluster_node_id, course['id'], color='grey', hidden=True)  # Hidden edge to influence positioning

        # Add edges based on prerequisites
        for course in courses:
            for prereq_id in course['prerequisites']:
                if prereq_id in course_dict:
                    net.add_edge(prereq_id, course['id'])

        # Similarity threshold for course connections
        similarity_threshold = 0.85  # Adjust as needed

        # Add edges based on similarity within clusters
        if len(courses) >= 2:
            for cluster_id in set(course['cluster'] for course in courses):
                cluster_courses = [course for course in courses if course['cluster'] == cluster_id]
                for i in range(len(cluster_courses)):
                    for j in range(i + 1, len(cluster_courses)):
                        course_a = cluster_courses[i]
                        course_b = cluster_courses[j]
                        similarity = cosine_similarity(course_a['embedding'], course_b['embedding'])
                        if similarity > similarity_threshold:
                            # Add an edge between similar courses
                            net.add_edge(
                                course_a['id'],
                                course_b['id'],
                                color='blue',
                                title=f"Similarity: {similarity:.2f}"
                            )

        # Threshold for cluster similarity
        cluster_similarity_threshold = 0.5  # Adjust as needed

        # Add edges between cluster label nodes
        for cluster_id_a, cluster_id_b, similarity in cluster_similarities:
            if similarity > cluster_similarity_threshold:
                node_id_a = cluster_labels[cluster_id_a]
                node_id_b = cluster_labels[cluster_id_b]
                net.add_edge(
                    node_id_a,
                    node_id_b,
                    color='orange',
                    width=2,
                    title=f"Cluster Similarity: {similarity:.2f}"
                )

        # Identify and highlight start and end nodes
        prereq_ids = set()
        for course in courses:
            prereq_ids.update(course['prerequisites'])

        start_nodes = [course['id'] for course in courses if not course['prerequisites']]
        end_nodes = [course['id'] for course in courses if course['id'] not in prereq_ids]

        for node_id in start_nodes:
            net.get_node(node_id)['color'] = 'green'

        for node_id in end_nodes:
            net.get_node(node_id)['color'] = 'red'

        # Generate the network graph HTML
        html = net.generate_html()
        print("Generated network graph HTML.")

        return courses, html

    except Exception as e:
        print(f"Error occurred during roadmap generation: {str(e)}")
        return None, None

# Main function to run the app
def main():
    st.title("Learning Roadmap Generator")

    # Knowledge Graph Section
    st.header("Knowledge Graph")
    user_query = st.text_input("Enter your learning goal:", "List all courses for learning database management systems.")
    if user_query:
        courses, html = generate_learning_roadmap(user_query)
        if courses and html:
            # Display the network graph in Streamlit
            components.html(html, height=750, scrolling=True)
        else:
            st.write("No courses found for your query.")

if __name__ == "__main__":
    init_neo4j()
    main()
    close_neo4j_connection()
