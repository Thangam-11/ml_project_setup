from setuptools import setup, find_packages

# Function to read requirements.txt
def get_requirements(file_path="requirements.txt"):
    with open(file_path) as f:
        return f.read().splitlines()

setup(
    name="ml_project_setup",      # ✅ project/package name
    version="0.0.1",                      # ✅ version
    author="thangarasu",                  # ✅ your name
    author_email="thangamani1128@gmail.com",
    packages=find_packages(),             # ✅ automatically finds Python packages
    install_requires=get_requirements(),  # ✅ reads dependencies from requirements.txt
    python_requires=">=3.10",             # ✅ minimum Python version required
)
