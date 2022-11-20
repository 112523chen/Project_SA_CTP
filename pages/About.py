import streamlit as st

st.set_page_config(
    layout="centered"
)

tab1, tab2, tab3 = st.tabs(["FAQ", "Technologies", "Data"])

with tab1:
    st.header("Frequency Asked Question")

    purpose = st.expander("What is the purpose of this project?")
    purpose.write("""
    Create a machine learning model that predicts the emotions of a section of text using data from a labeled dataset.
    """)

    accuracy = st.expander("What is the accuracy of our model?")
    accuracy.write("""
    The current demo's model has an accuracy score of 88%
    """)

    takeaways = st.expander("What are some key takeaways from this project?")
    takeaways.write(
        """
        - Sometimes using techniques like Stemming aren't good as generalizations aren't helpful for computers to understand language
        """
    )

with tab2:
    st.header("Technologies Used In This Project")
    col1, col2, col3 = st.columns(3)
    technologies_badges = [
        "![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)",
        "![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)",
        "![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)",
        "![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)",
        "![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)",
        "![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)"
    ]
    for idx, tech in enumerate(technologies_badges):
        if idx % 3 == 0:
            col1.write(tech)
        elif idx % 3 == 1:
            col2.write(tech)
        else:
            col3.write(tech)

with tab3:
    st.header("Datasets Used In This Project")
    st.write(
        """
        [Emotion Dataset for Emotion Recognition Tasks](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)

        > Dataset that contains a section of text with labels for the text that describes the emotion of the text
        """
    )