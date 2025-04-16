from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()


## edit below variables as per your requirements
REPO_NAME = "Multiple-Diseases-System-Using-Machine-Learning"
AUTHOR_USER_NAME = "Techexpvaishnavi"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS =["streamlit","numpy","scikit-learn","Pillow","streamlit-option-menu"]


setup(
    name=SRC_REPO,
    version="0.0.1",
    author = AUTHOR_USER_NAME,
    description = " A small package for Multiple Diseases Prediction System ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="vg309587@gmail.com",
    package=[SRC_REPO],
    python_requires=">=3.8",
    install_requires=LIST_OF_REQUIREMENTS

)