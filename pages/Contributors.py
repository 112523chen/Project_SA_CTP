import streamlit as st

st.set_page_config(
    layout="wide"
)

st.header("Contributors")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        # Victor Wu
        [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/victor-wu-/)
        [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/WuVi5054)
        """
    )

with c2:
    st.markdown(
        """
        # Alex Chen
        [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alex-chen-112523chen/)
        [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/112523chen)
        """
    )

with c3:
    st.markdown(
        """
        # Richard Yeung
        [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yeung-richard/)
        [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ryrichard)
        """
    )