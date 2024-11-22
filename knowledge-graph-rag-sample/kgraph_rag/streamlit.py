import streamlit as st
import os
from dotenv import load_dotenv
from main import generate_learning_roadmap, init_neo4j, close_neo4j_connection
import json
from datetime import datetime

# Load environment variables
load_dotenv()

def save_progress_to_file():
    """Save progress data to a JSON file"""
    with open('course_progress.json', 'w') as f:
        json.dump(st.session_state.course_progress, f)

def load_progress_from_file():
    """Load progress data from JSON file"""
    try:
        with open('course_progress.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def display_course_sequence(courses):
    # Deduplicate courses based on title
    seen_titles = set()
    unique_courses = []
    for course in courses:
        if course['title'] not in seen_titles:
            seen_titles.add(course['title'])
            unique_courses.append(course)
    
    # Use the deduplicated list for display
    st.markdown("### 📚 Learning Path")
    
    # Group courses by difficulty level
    level_names = {
        1: "🔵 Foundational Courses (Start Here)",
        2: "🟡 Intermediate Courses",
        3: "🔴 Advanced Courses"
    }
    
    # Sort and group courses using the deduplicated list
    sorted_courses = sorted(unique_courses, key=lambda x: (x.get('level', 1), len(x.get('prerequisites', []))))
    current_level = None
    
    for course in sorted_courses:
        level = course.get('level', 1)
        
        # Show level header when it changes
        if level != current_level:
            st.markdown(f"### {level_names.get(level, 'Other Courses')}")
            current_level = level
        
        # Create course card
        with st.expander(f"📘 {course['title']}", expanded=False):
            # Show prerequisites if any
            prereqs = course.get('prerequisites', [])
            if prereqs:
                st.markdown("**Prerequisites:**")
                for prereq in prereqs:
                    st.markdown(f"- ✓ {prereq['title']}")
                st.markdown("---")
            
            # Course description
            st.markdown("**Course Description:**")
            st.write(course.get('description', 'No description available'))
            
            # Progress tracking
            col1, col2 = st.columns([3,1])
            with col1:
                current_status = st.session_state.course_progress.get(
                    course['id'], {}).get('status', 'Not Started')
                status = st.selectbox(
                    "Status",
                    options=['Not Started', 'In Progress', 'Completed'],
                    key=f"status_{course['id']}",
                    index=['Not Started', 'In Progress', 'Completed'].index(current_status)
                )
            
            with col2:
                if st.button("Save", key=f"save_{course['id']}"):
                    st.session_state.course_progress[course['id']] = {
                        'status': status,
                        'title': course['title'],
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_progress_to_file()
                    st.success("✓")
            
            # Show estimated completion time or other metadata
            col1, col2 = st.columns(2)
            with col1:
                # Get difficulty_level from course object, fallback to 1 if not found
                difficulty_level = course.get('difficulty_level', 1)
                st.markdown(f"**Difficulty Level:** {'⭐' * int(difficulty_level)}")
            with col2:
                st.markdown(f"**Prerequisites Count:** {len(prereqs)}")

    # Add a progress summary at the bottom
    st.markdown("---")
    st.markdown("### 📊 Progress Summary")
    col1, col2, col3 = st.columns(3)
    
    total_courses = len(courses)
    completed = sum(1 for c in st.session_state.course_progress.values() 
                   if c.get('status') == 'Completed')
    in_progress = sum(1 for c in st.session_state.course_progress.values() 
                     if c.get('status') == 'In Progress')
    
    with col1:
        st.metric("Total Courses", total_courses)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("In Progress", in_progress)
    
    # Add progress bar
    progress = (completed + in_progress * 0.5) / total_courses
    st.progress(progress)

# Initialize session state
if 'course_progress' not in st.session_state:
    st.session_state.course_progress = load_progress_from_file()
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# Streamlit Title and Inputs
st.title("The Brain")
st.markdown("""
Welcome to the clustering and catagorizing phase - here all topics are extracted and relationships are made between entities based on similarities 
Type in any learning goal or subject, and we'll generate a personalized learning graph for you!
""")

# User Query Input
user_query = st.text_input("Enter your learning goal (e.g., 'What courses do I need to learn AI?'):", 
                          value=st.session_state.current_query)

if st.button("Generate Roadmap") or st.session_state.current_query:
    if user_query:
        st.session_state.current_query = user_query
        init_neo4j()
        
        try:
            st.write("Generating your personalized learning roadmap... Please wait.")
            courses, html_content = generate_learning_roadmap(user_query)
            
            if courses and html_content:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    display_course_sequence(courses)
                
                with col2:
                    st.markdown("### 🔍 Course Graph")
                    st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Show progress overview in sidebar
                with st.sidebar:
                    st.markdown("### Your Progress")
                    completed = sum(1 for status in st.session_state.course_progress.values() 
                                  if status.get('status') == 'Completed')
                    in_progress = sum(1 for status in st.session_state.course_progress.values() 
                                    if status.get('status') == 'In Progress')
                    st.metric("Completed Courses", completed)
                    st.metric("Courses In Progress", in_progress)
                
            else:
                st.warning("No courses found for your query.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        close_neo4j_connection()
    else:
        st.warning("Please enter a learning goal.")

def get_optimized_course_sequence(graph, core_topics):
    print("\nExecuting optimized course sequence query...")
    print(f"Core topics: {core_topics}")
    
    topic_conditions = []
    params = {}
    for i, topic in enumerate(core_topics):
        param_name = f'kw{i}'
        topic_conditions.append(
            f"(toLower(c.title) CONTAINS ${param_name} OR toLower(c.description) CONTAINS ${param_name})"
        )
        params[param_name] = topic.lower()

    # Updated query to better handle course relationships
    query = f"""
    MATCH (c:Course)
    WHERE {" OR ".join(topic_conditions)}
    WITH DISTINCT c
    
    // Find all prerequisite paths
    OPTIONAL MATCH prereq_paths = (c)-[:REQUIRES|PREREQUISITE_FOR*1..]->(prereq:Course)
    WITH c, collect(distinct prereq) as prerequisites, 
         CASE 
             WHEN count(prereq_paths) > 0 THEN max(length(prereq_paths))
             ELSE 0 
         END as prereq_depth
    
    // Find related courses
    OPTIONAL MATCH (c)-[:SIMILAR_TO|RELATED_TO]-(related:Course)
    WHERE related.id IN [p IN prerequisites | p.id]
    
    // Calculate difficulty level based on prerequisites
    WITH c, prerequisites, prereq_depth, collect(distinct related) as related_courses,
         CASE 
             WHEN size(prerequisites) = 0 THEN 1
             WHEN size(prerequisites) <= 2 THEN 2
             ELSE 3
         END as difficulty_level
    
    RETURN c,
           prerequisites,
           related_courses,
           prereq_depth,
           difficulty_level
    ORDER BY difficulty_level ASC, prereq_depth ASC, size(prerequisites) ASC
    """
    
    # Debug the query results
    result = graph.run(query, **params)
    results_list = list(result)
    print(f"\nFound {len(results_list)} courses")
    print("Course sequence:")
    for record in results_list:
        print(f"\nCourse: {record['c']['title']}")
        print(f"Prereq depth: {record['prereq_depth']}")
        print(f"Difficulty level: {record['difficulty_level']}")
        print(f"Prerequisites: {[p['title'] for p in record['prerequisites']]}")
        print(f"Related courses: {[r['title'] for r in record['related_courses']]}")
    
    return graph.run(query, **params)

st.markdown("""
    <style>
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
