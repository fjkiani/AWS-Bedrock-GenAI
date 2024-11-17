import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Define the PeterPortal API base URL
PETERPORTAL_API_BASE = "https://api.peterportal.org/rest/v0"

# Function to fetch course data with a limit on results
def fetch_course_data(department, limit=None, chunk_size=100):
    """
    Fetch course data from the API in chunks.
    If limit is set to None, it will fetch all available courses.
    """
    url = f"{PETERPORTAL_API_BASE}/courses/all"
    all_courses = []
    offset = 0

    while True:
        # Request courses in chunks
        params = {"department": department, "limit": chunk_size, "offset": offset}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error fetching courses: {response.status_code}")
            break

        data = response.json()
        if not data:
            break  # No more data to fetch

        # Add chunk of courses to all_courses
        all_courses.extend(data)
        print(f"Fetched {len(data)} courses, total: {len(all_courses)}")

        # Check if we reached the limit, if set
        if limit and len(all_courses) >= limit:
            all_courses = all_courses[:limit]
            break

        # Increase offset for the next chunk
        offset += chunk_size

    return all_courses

# Test example
def test_fetch_course_data():
    # Define test parameters
    department = "COMPSCI"
    limit = 50  # Modify limit to test different number of courses

    print(f"Testing fetch_course_data with department: {department} and limit: {limit}")
    courses = fetch_course_data(department, limit=limit)

    # Print the number of courses retrieved and a sample of the data
    if courses:
        print(f"Total courses retrieved: {len(courses)}")
        print("Sample courses:")
        for course in courses[:5]:  # Print first 5 courses as a sample
            print(course)
    else:
        print("No courses retrieved.")

# Run the test function
if __name__ == "__main__":
    test_fetch_course_data()
