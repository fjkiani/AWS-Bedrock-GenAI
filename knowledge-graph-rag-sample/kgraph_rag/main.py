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
import time
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from datetime import datetime
import json

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
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        text = text.replace("\n", " ")
        response = openai.Embedding.create(
            input=[text],
            model=model,
            timeout=30  # Add timeout
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise  # Re-raise for retry

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Add connection retry decorator
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_neo4j_connection():
    try:
        print("Attempting to connect to Neo4j...")
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Test the connection
        graph.run("MATCH (n) RETURN n LIMIT 1")
        print("Successfully connected to Neo4j")
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {str(e)}")
        raise

# Main function to generate the learning roadmap
def generate_learning_roadmap(query):
    try:
        print("Starting learning roadmap generation...")
        
        # Try to establish Neo4j connection with retry logic
        try:
            graph = get_neo4j_connection()
        except RetryError:
            print("Failed to connect to Neo4j after multiple attempts")
            return None, None

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

        # Replace the current course fetching logic with optimized sequence
        result = get_optimized_course_sequence(graph, core_topics)
        
        # Process the Neo4j results into proper course objects
        courses = []
        for record in result:
            try:
                course_data = record['c']  # This is the Neo4j node
                course = {
                    'id': str(course_data.get('id', '')),
                    'title': str(course_data.get('title', 'Untitled')),
                    'description': str(course_data.get('description', 'No description available')),
                    'difficulty_level': int(record.get('difficulty_level', 1)),
                    'prereq_depth': int(record.get('prereq_depth', 0)),
                    'prereq_count': int(record.get('prereq_count', 0)),
                    'prerequisites': [str(p.get('id', '')) for p in record.get('prerequisites', [])],
                    'related_courses': [str(r.get('id', '')) for r in record.get('related_courses', [])]
                }
                courses.append(course)
                print(f"Processed course: {course['title']}")  # Debug print
            except Exception as e:
                print(f"Error processing course record: {e}")
                continue

        if not courses:
            print("No courses were processed successfully")
            return None, None

        # Compute embeddings for courses
        for course in courses:
            text = f"{course['title']} {course['description']}"
            course['embedding'] = get_embedding(text)

        # Perform clustering if we have enough courses
        if len(courses) >= 2:
            embeddings = [course['embedding'] for course in courses]
            embeddings_array = np.array(embeddings)
            
            num_clusters = min(len(courses), 5)  # Maximum of 5 clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(embeddings_array)
            
            # Assign cluster labels
            for idx, course in enumerate(courses):
                course['cluster'] = int(kmeans.labels_[idx])

            # Generate cluster labels using OpenAI
            cluster_labels = {}
            for cluster_id in set(course['cluster'] for course in courses):
                cluster_courses = [course for course in courses if course['cluster'] == cluster_id]
                cluster_texts = " ".join(f"{course['title']} {course['description']}" for course in cluster_courses)
                
                messages = [
                    {"role": "system", "content": "Generate a short, descriptive label (2-4 words) for a group of related courses."},
                    {"role": "user", "content": f"Topics: {cluster_texts}"}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=20,
                    temperature=0.3
                )
                
                label = response['choices'][0]['message']['content'].strip()
                cluster_labels[cluster_id] = label

        # Compute cluster centroids and similarities
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

        # Adjust layout settings for better cluster visualization
        net.set_options("""
        {
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
            },
            "nodes": {
                "font": {
                    "size": 16
                }
            },
            "edges": {
                "color": {
                    "inherit": false
                },
                "smooth": {
                    "enabled": true,
                    "type": "continuous"
                }
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
                level=0
            )
            cluster_labels[cluster_id] = cluster_node_id

        # Add course nodes and connect them to cluster labels
        for course in courses:
            net.add_node(
                course['id'],
                label=course['title'],
                title=course['description'],
                group=course['cluster'],
                shape='dot',
                size=25
            )
            # Connect to cluster
            cluster_node_id = cluster_labels[course['cluster']]
            net.add_edge(cluster_node_id, course['id'], color='grey', hidden=True)

        # Create course dictionary for prerequisite lookup
        course_dict = {course['id']: course for course in courses}

        # Add prerequisite edges
        for course in courses:
            for prereq_id in course['prerequisites']:
                if prereq_id in course_dict:
                    net.add_edge(
                        prereq_id, 
                        course['id'],
                        color='#666666',
                        arrows={'to': {'enabled': True}}
                    )

        # Add similarity edges within clusters
        similarity_threshold = 0.85
        for cluster_id in set(course['cluster'] for course in courses):
            cluster_courses = [course for course in courses if course['cluster'] == cluster_id]
            for i in range(len(cluster_courses)):
                for j in range(i + 1, len(cluster_courses)):
                    course_a = cluster_courses[i]
                    course_b = cluster_courses[j]
                    similarity = cosine_similarity(course_a['embedding'], course_b['embedding'])
                    if similarity > similarity_threshold:
                        net.add_edge(
                            course_a['id'],
                            course_b['id'],
                            color='blue',
                            title=f"Similarity: {similarity:.2f}",
                            width=1
                        )

        # Add edges between clusters
        cluster_similarity_threshold = 0.5
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

        # Highlight start and end nodes
        prereq_ids = set()
        for course in courses:
            prereq_ids.update(course['prerequisites'])

        start_nodes = [course['id'] for course in courses if not course['prerequisites']]
        end_nodes = [course['id'] for course in courses if course['id'] not in prereq_ids]

        for node_id in start_nodes:
            net.get_node(node_id)['color'] = 'green'

        for node_id in end_nodes:
            net.get_node(node_id)['color'] = 'red'

        # Generate the HTML
        try:
            html = net.generate_html()
            print(f"Generated network visualization with {len(net.nodes)} nodes and {len(net.edges)} edges")
        except Exception as e:
            print(f"Error generating visualization: {e}")
            html = ""

        return courses, html

    except Exception as e:
        print(f"Error in generate_learning_roadmap: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None
    finally:
        # Ensure we close the connection even if there's an error
        if 'graph' in locals():
            try:
                graph.close()
                print("Neo4j connection closed properly")
            except:
                print("Error while closing Neo4j connection")

def initialize_session_state():
    if 'course_progress' not in st.session_state:
        st.session_state.course_progress = {}

# Main function to run the app
def main():
    st.title("Learning Roadmap Generator")
    
    # Initialize session state
    initialize_session_state()
    
    # Knowledge Graph Section
    st.header("Knowledge Graph")
    user_query = st.text_input("Enter your learning goal:", "What courses do I need to learn AI?")
    
    if user_query:
        with st.spinner('Generating your learning roadmap...'):
            try:
                print("Starting roadmap generation...")
                courses, html = generate_learning_roadmap(user_query)
                
                if courses and html:
                    # Display the graph first
                    st.markdown("### 🔍 Course Graph")
                    components.html(html, height=750, scrolling=True)
                    
                    # Then show the course list
                    st.markdown("### 📚 Learning Path")
                    
                    # Debug courses data
                    print("\nDEBUG - Courses data:")
                    print(f"Number of courses: {len(courses)}")
                    if courses:
                        print("First course structure:", courses[0])
                    
                    # Sort courses safely
                    try:
                        sorted_courses = sorted(
                            courses,
                            key=lambda x: (
                                x.get('difficulty_level', 1),
                                x.get('prereq_depth', 0),
                                len(x.get('prerequisites', [])),
                                x.get('title', '')
                            )
                        )
                    except Exception as e:
                        print(f"Sorting error: {e}")
                        sorted_courses = courses
                    
                    # Display courses
                    for i, course in enumerate(sorted_courses, 1):
                        try:
                            with st.expander(f"Step {i}: {course.get('title', 'Untitled Course')}", expanded=False):
                                # Course Description
                                st.write("**Description:**")
                                st.write(course.get('description', 'No description available'))
                                
                                # Prerequisites
                                prereqs = course.get('prerequisites', [])
                                if prereqs:
                                    st.write("**Prerequisites:**")
                                    for prereq_id in prereqs:
                                        prereq = next(
                                            (c for c in courses if c.get('id') == prereq_id),
                                            {'title': 'Unknown Course'}
                                        )
                                        st.markdown(f"- {prereq.get('title', 'Unknown Course')}")
                                
                                # Course Details
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    difficulty = int(course.get('difficulty_level', 1))
                                    st.write("**Difficulty Level:**", "⭐" * difficulty)
                                    st.write("**Prerequisites Count:**", len(prereqs))
                                
                                with col2:
                                    course_id = str(course.get('id', ''))
                                    current_status = st.session_state.course_progress.get(
                                        course_id, {}).get('status', 'Not Started')
                                    
                                    status = st.selectbox(
                                        "Status",
                                        options=['Not Started', 'In Progress', 'Completed'],
                                        key=f"status_{course_id}",
                                        index=['Not Started', 'In Progress', 'Completed'].index(current_status)
                                    )
                                    
                                    if st.button("Save Progress", key=f"save_{course_id}"):
                                        st.session_state.course_progress[course_id] = {
                                            'status': status,
                                            'title': course.get('title', 'Untitled Course'),
                                            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        st.success(f"Progress saved! Course marked as {status}")
                        except Exception as e:
                            st.error(f"Error displaying course: {str(e)}")
                            continue
                    
                    # Sidebar Progress Overview
                    with st.sidebar:
                        st.markdown("### Your Progress")
                        completed = sum(1 for c in st.session_state.course_progress.values() 
                                     if c.get('status') == 'Completed')
                        in_progress = sum(1 for c in st.session_state.course_progress.values() 
                                        if c.get('status') == 'In Progress')
                        st.metric("Completed Courses", completed)
                        st.metric("Courses In Progress", in_progress)
                
                else:
                    st.warning("No courses found for your query. Please try different search terms.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(f"Error details: {e}")
                import traceback
                print(traceback.format_exc())

def get_optimized_course_sequence(graph, core_topics):
    print("\nExecuting optimized course sequence query...")
    print(f"Core topics: {core_topics}")
    
    # Update difficulty query with more specific levels
    difficulty_query = """
    MATCH (c:Course)
    SET c.difficulty = 
    CASE 
        WHEN c.title CONTAINS 'Introduction' OR 
             c.title CONTAINS 'Fundamentals' OR 
             c.title CONTAINS 'Basics'
        THEN 1
        WHEN c.title CONTAINS 'Statistical' OR
             c.title CONTAINS 'Neural Networks' OR
             c.title CONTAINS 'Deep Learning'
        THEN 3
        WHEN c.title CONTAINS 'Advanced' OR
             c.title CONTAINS 'Generative' OR
             c.title CONTAINS 'Control'
        THEN 4
        ELSE 2
    END
    """
    graph.run(difficulty_query)
    
    # Build the topic conditions
    topic_conditions = []
    params = {}
    for i, topic in enumerate(core_topics):
        param_name = f'kw{i}'
        topic_conditions.append(
            f"(toLower(c.title) CONTAINS ${param_name} OR toLower(c.description) CONTAINS ${param_name})"
        )
        params[param_name] = topic.lower()

    # Updated query with proper aggregation handling
    query = f"""
    MATCH (c:Course)
    WHERE {" OR ".join(topic_conditions)}
    
    // Find explicit prerequisites
    OPTIONAL MATCH path=(c)-[r:REQUIRES*]->(prereq:Course)
    WITH c, 
         collect(DISTINCT prereq) as prerequisites,
         max(length(path)) as prereq_depth
    
    // Find related courses
    OPTIONAL MATCH (c)-[sim:SIMILAR_TO]-(related:Course)
    WHERE related.difficulty <= coalesce(c.difficulty, 1)
    
    RETURN 
        c,
        prerequisites,
        collect(DISTINCT related) as related_courses,
        coalesce(c.difficulty, 1) as difficulty_level,
        coalesce(prereq_depth, 0) as prereq_depth,
        size(prerequisites) as prereq_count
    ORDER BY difficulty_level ASC, prereq_depth ASC, prereq_count ASC, c.title ASC
    """
    
    return graph.run(query, **params)

if __name__ == "__main__":
    try:
        init_neo4j()
        main()
    finally:
        close_neo4j_connection()
