import streamlit as st
from streamlit_option_menu import option_menu
import requests 
from streamlit_lottie import st_lottie
# Setting Page Config, must remain at top
st.set_page_config(page_title="My Portfolio ", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Creating Functions for differet Sections
# Home Function --------------------------------------------------------------------------------------------------------------
def Home():
    st.markdown("""
    ## Hi, I'm Sagar Puthalapattu 👋
    # A Data Science And AI / ML Engineering 
    By day, I'm diving deep into **Data Structures and Algorithms (DSA)**, and by night, I'm exploring the vast worlds of Data Science, including **Data Analysis**, **Machine Learning**, and **Deep Learning**. My journey also includes working on cutting-edge *Large Language Models*, bringing innovative ideas to fruition. My technical arsenal includes **Python**, **SQL**, **Streamlit**, **pandas**, **numpy**, **matplotlib**, **Plotly**, and **seaborn**, to name a few. 
    
    My portfolio is a treasure trove of diverse projects, from building pathfinder visualizers and Gemini API clones to developing interactive data visualization tools. Whether it's crafting complex algorithms or designing captivating user interfaces, I'm your go-to person for all things development. When I'm not coding, you can find me exploring the latest tech trends or honing my problem-solving skills. I'm always eager to learn and grow, and I'm excited about where my passion for Data Science and development will take me next. If you're looking to bring your ideas to life, let's collaborate and create something extraordinary together!

    📂 Here, you will find a collection of my **projects**, **achievements**, and **insights** into the world of data science and artificial intelligence. 🌐 Explore and feel free to reach out if you have any questions or collaboration ideas!

    📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/sagar-puthalapattu/) || [GitHub](https://github.com/sagarputhalapattu) ||  [Email](mailto:sagarputhalapattu@gmail)
    """)
    
    st.write("Here's my resume:")
    with open("F:/Resume/portfoilo/Puthalapattu Sagar Resume A.docx", "rb") as file:
        btn = st.download_button(
            label="Download Resume",
            data=file,
            file_name="Puthalapattu Sagar .docx",
            mime="application/pdf"
        )
    lottie_url = "https://lottie.host/a5c4040b-e424-4634-a7e6-660c8ce12c32/LEkIf9Zzl4.json"
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, height=300, key="home")
    else:
        st.error("Failed to load Lottie animation.")
    

# Main container for description
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/afcc1a9b-2645-4d65-bc8b-e49625ff20cf/KDd9FRoCai.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="HOME")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Data Science Internship at Cognifyz Technologies")
            st.write("#### Duration: August 2024 – September 2024")
            st.write("""
                    - Focused on **data analysis** and developing insights to align with **business analyst responsibilities**.
                    - Collaborated with cross-functional teams to derive insights from data and present actionable recommendations.
                    - Contributed to several data-driven tasks and projects during the internship.
                    """)
            if st.button("Know More 🍓🥝"):
                with st.expander("### Key Tasks and Responsibilities", expanded=False):
                    st.write("""
                            - **Analyzing Ratings and Cuisines**: Explored relationships between restaurant cuisines and ratings to derive actionable insights.
                            - **Predictive Modeling**: Built models to estimate aggregate ratings for businesses.
                            - **Geographic Insights**: Examined correlations between restaurant locations and customer ratings.
                            - **Data Visualization**: Created visualizations (e.g., histograms, bar plots) to present findings effectively.
                            - **Drawing Conclusions**: Formulated insights into customer preferences and cuisine popularity.
                            """)
                with st.expander("### Technologies and Tools Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Python** for data manipulation and analysis.")
                    st.write("- **Pandas, NumPy** for data processing.")
                    st.write("- **Matplotlib, Seaborn** for data visualization.")
                with col2:
                    st.write("- **Power BI, Tableau** for creating interactive dashboards.")
                    st.write("- **Scikit-learn** for predictive modeling.")
                    st.write("- **SQL** for querying datasets.")
                with st.expander("### Key Learnings and Takeaways", expanded=False):
                    st.write("""
                            - Enhanced problem-solving and data interpretation skills.
                            - Gained expertise in presenting data insights to stakeholders.
                            - Improved knowledge in customer behavior analytics and industry-specific challenges.
                            """)
                with st.expander("### Portfolio Link", expanded=False):
                    st.write("[Explore My Internship Projects](https://github.com/sagarputhalapattu/Cognifyz_internship)")
                st.write("---")
                st.write("#### Have questions about this experience? Feel free to reach out!")


# Projects Function --------------------------------------------------------------------------------------------------------------
def Projects():
    
    # Project 1 Container
    with st.container():
        st.header("My AI / ML Projects")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/161a3757-2d51-4d71-8b3f-391b7c5761e3/BWiTvPC4r3.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project1")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            # Enter Descritipion
            st.write("### Yolo RealTime Object Detection")
            st.write("""
                    - Developed a real- time Object detection system using the Yolo Algorithm.
                    - Leveraged deep learning and advanced techniques for high-speed and accurate detection.
                    - Received Positive feedback for creating an efficient and User-friendly solution.
                    """)
            if st.button("Know More ➡️"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("- Python")
                        st.write("- OpenCV")
                        st.write("- Flask / FastAPI")
                    with col2:
                        st.write("- TensorFlow / Pytorch")
                        st.write("- Dockers")
                        st.write("- YOLO(v4/v5)")
                    with col3:
                        st.write("- streamlit")
                        st.write("- Pandas")
                        st.write("- Numpy")
                        
                with st.expander("### Project Link", expanded = False):
                    st.write("[Yolo RealTime Object Detection repo](https://github.com/sagarputhalapattu/projects/tree/main/files)")
                    
                with st.expander("Features", expanded = False):
                    st.write("""
                            - **Real-Time Detection** : Detect multiple Objects in real-time using Yolo model.
                            - **Data Processing**: Enhanced Preprocessing and augmentation for improved accuracy.
                            - **User-Friendly Interface**:Seamless interaction through a web-based Platform.
                            - **Advanced Deployment**:Integrated Docker for Scalable deployment Options.
                            """)
    
    # Project 2 Container
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/63045f0c-21c3-4feb-a341-a5506dd657e0/UW7N5jUnLr.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project2")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Customer Churn model Using BERT and CI/CD Integration ")
            st.write("""
                    - Developed a classification model to predict Customer Churn using Bert accuracy by 25%.
                    - Integrated the solution with CI/CD pipelines for automated deployment and monitoring.
                    - Enhanced efficiency by leveraging cloud platforms for scalability and performance
                    """)
            if st.button("Know More 🚀"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- **BERT** fro NLP-based Churn prediction.")
                        st.write("- **Azure DevOps** for CI/CD pipeline.")
                        st.write("- **Databricks** for model training and experimentation")
                    with col2:
                        st.write("- **Python** for implementation.")
                        st.write("- **Flask** for API deployment")
                        st.write("- **Docker** for Containerization")
                
                with st.expander("### Project Link", expanded = False):
                    st.write("[Customer Churn Model Using BERT and CI/CD Integration's Repo](https://github.com/sagarputhalapattu/projects/tree/main/Customer_Churn_Model_using_BERT_CICD_Integration)")
                    
                with st.expander("Features", expanded = False):
                    st.write("""
                            - **Churn Prediction**:Leverage Bert to analyze textual Customer feedback and predict churn probability.
                            - **Attendance Deployment**:Integrated CI/CD pipeline for Seamless model Updates.
                            - **Cloud scalability**: Hosted and trained model on Databricks for efficient resource utilization.
                            - **Real-Time API**: Deployed using Flask for live prediction and monitoring.
                            """)
    
    # Project 3 Container
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/04bae7dc-8363-4b31-9ad5-3a612605696a/E3BLSwSgXu.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project3")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Q&A system for finance documents Using RAG(Retrieval-Augmented Generation)")
            st.write("""
                    - Developed a Q&A system to Extract answer from Finance Document Using a Retrieval-Augmented Generation (RAG)Model.
                    - Integrated Azure service for Document Storage , retrieval , and model deployment.
                    - Utilized the RAG Architecture to Combine Information Retrieval and Text generation for accurate for Context-aware Answer.
                    """)
            if st.button("Know More 🎂"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- **Azure** for Cloud Services,Document storage and Deployment")
                        st.write("- **Retrieval-Augmented Generation(RAG)**")
                        st.write("- **Flask**")
                    with col2:
                        st.write("- **Python**")
                        st.write("- **Transformers**")
                        st.write("- **HuggingFace**")
                
                with st.expander("### Project Link", expanded = False):
                    st.write("[Q&A system for finance documents Using RAG(Retrieval-Augmented Generation](https://github.com/sagarputhalapattu/projects/tree/main/LangChan_LLM_RAG)")
                
                with st.expander("Features", expanded = False):
                    st.write("""
                            - **Answer Extraction**: Utilized RAG Model to extract relevant answer from given context.
                            - **Contextualized Answer**: Generated answer considering the context provided by the user.
                            - **Real-time API**: Deployed using Flask for live prediction and monitoring.
                            - **Accurate Predictions**: Combines document retrieval and natural language generation for contextually relevant answers.
                            """)
    # Project 4 Container
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/2ba6554c-8a54-456d-b6c4-b70f155468c2/DX7lA91qJG.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project4")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Google Gemini Project")
            st.write("""
                    - Engineered a fully functional Google Gemini clone, achieving 100% feature parity and functionality replication .
                    - Crafted a pixel-perfect UI, meticulously adhering to Gemini's established design patterns and user experience guidelines .
                    - Architected the front-end using React.js, seamlessly integrating Google Gemini API for powerful LLM capabilities .
                    - Optimized performance, reducing initial load time by 20% compared to the original Gemini interface .
                    """)
            if st.button("Know More 🗯️"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- Open API")
                        st.write("- PYTHON")
                        st.write("- LLM")
                    with col2:
                        st.write("- SQL")
                
                with st.expander("### Project Link", expanded = False):
                    st.write("[View The Project on This Repo](https://github.com/sagarputhalapattu/Gen-AI-Projects-)")
    # project 5
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url =  "https://lottie.host/ae310169-459b-423c-a029-ed38c74d2de6/evSon5tkxW.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st.lottie(lottie_animation,speed=1,height=300 , key ="project11")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Leaf Disease Classification Using Computer Vision")
            st.write("""
                    - Developed a computer vision model to classify leaf diseases from images.
                    - Enhanced agricultural productivity by identifying diseases early, enabling timely intervention.
                    - Leveraged Convolutional Neural Networks (CNNs) for feature extraction and classification.
                    """)
            if st.button("Know More 🍀"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **TensorFlow** for building and training CNN models.")
                    st.write("- **Keras** for model architecture and layers.")
                    st.write("- **OpenCV** for image preprocessing and augmentation.")
                with col2:
                    st.write("- **Python** for implementation.")
                    st.write("- **Flask API** for deployment.")
                    st.write("- **Streamlit** for interactive visualization.")
                with st.expander("### Project Link", expanded=False):
                    st.write("[Access the Leaf Disease Classification Project](https://github.com/sagarputhalapattu/projects/tree/main/Leaf%20data%20set)")
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Disease Classification**: Accurately classifies various leaf diseases such as bacterial blight, rust, and mildew.
                            - **Image Augmentation**: Improved model robustness using techniques like rotation, flipping, and zoom.
                            - **Real-Time Prediction**: Integrated with an API for live predictions on user-uploaded images.
                            - **Visualization**: Heat-maps for visualizing model focus on diseased areas.
                            """)
                with st.expander("### Model Performance", expanded=False):
                    st.write("""
                            - **Accuracy**: Achieved 95% validation accuracy on the test dataset.
                            - **Metrics Used**: Precision, Recall, F1-Score, Confusion Matrix.
                            """)
                    st.write("---")
                    st.write("#### Want to learn more about this project? Feel free to reach out!")
# 6
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url =  "https://lottie.host/0db5db6e-48ab-4ec8-8605-7280cc9f57a6/hwWWwUWyQi.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st.lottie(lottie_animation,speed=1,height=300 , key ="project12")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Algorithmic Trading Model")
            st.write("""
                    - Developed a model for algorithmic trading to automate financial transactions based on predictive analytics.
                    - Utilized financial data and machine learning techniques to identify profitable trading strategies.
                    - Designed for robust performance under real-time market conditions.
                    """)
            if st.button("Know More 💹"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Python** for implementation.")
                    st.write("- **Pandas** for data preprocessing and manipulation.")
                    st.write("- **Scikit-learn** for building predictive models.")
                with col2:
                    st.write("- **Matplotlib/Seaborn** for data visualization.")
                    st.write("- **Backtrader** for backtesting trading strategies.")
                    st.write("- **Streamlit** for creating interactive dashboards.")
                with st.expander("### Project Link", expanded=False):
                    st.write("[Access the Algorithmic Trading Model](https://github.com/sagarputhalapattu/projects/blob/main/Algorithmic_Trading_Model_.ipynb)")
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Predictive Analytics**: Utilized machine learning to forecast stock prices and trends.
                            - **Automated Trading**: Implemented logic for executing trades based on model outputs.
                            - **Backtesting**: Evaluated strategies on historical data to ensure reliability and robustness.
                            - **Interactive Dashboard**: Visualized trading performance and strategy insights in real-time.
                            """)
                    st.write("---")
                    st.write("#### Want to learn more about this project? Feel free to reach out!")
    #7
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url =  "https://lottie.host/2f1fe33b-5511-4f96-97ca-1d9ba3e165b7/WKLZLja6Vd.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st.lottie(lottie_animation,speed=1,height=300 , key ="project13")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Portfolio Risk Management Project")
            st.write("""
                    - Developed a portfolio risk management system to assess and mitigate financial risks.
                    - Analyzed asset allocations and calculated risk metrics to optimize portfolios.
                    - Integrated machine learning techniques to predict and minimize investment risks.
                    """)
            if st.button("Know More 🤖"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Python** for implementing risk management algorithms.")
                    st.write("- **Pandas, NumPy** for financial computations.")
                    st.write("- **Matplotlib, Plotly** for interactive visualizations.")
                with col2:
                    st.write("- **Scikit-learn** for predictive analytics.")
                    st.write("- **Streamlit** for building an interactive dashboard.")
                    st.write("- **Jupyter Notebooks** for exploratory data analysis.")
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Risk Assessment**: Calculate risk-adjusted returns using Sharpe and Sortino ratios.
                            - **Portfolio Optimization**: Employ Monte Carlo simulations for optimal asset allocation.
                            - **Interactive Dashboards**: Visualize portfolio performance and risk metrics.
                            """)
                with st.expander("### Project Link", expanded=False):
                    st.write("[View the Portfolio Risk Management Project](https://your-project-link.com)")
                st.write("---")
                st.write("#### Interested in this project? Reach out for more details!")


            
    # projects of the Data Analyst 
    with st.container():
        st.write("---")
        st.write("##")
        st.header("Data Analytics Projects")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url =  "https://lottie.host/86f714c0-41df-4d4a-8804-1922af591f4d/lrfcPjv4m2.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st.lottie(lottie_animation,speed=1,height=300 , key ="project5")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### IPL 2024 Data Analytics Project ")
            st.write("""
                    - Conducted a Comprehensive analysis of IPL 2024 matches,focusing on match statistics,player performance and team strategies.
                    - Integrated data Visualization for insights and trend analysis.
                    - Utilized machine learning algorithms to identify patterns and trends in the data.
                    - Generated insights and recommendations to improve Bat or Ball performance.
                    """)
            if st.button("Know More 🏏"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- Python")
                        st.write("- Pandas")
                        st.write("- Matplotlib")
                    with col2:
                        st.write("- Scikit-learn")
                        st.write("- Statistics analysis")
                        st.write("- Data Visualization")
                
                with st.expander("### Project Link", expanded = False):
                    st.write("[Explore the IPL2024-Dataset](https://github.com/sagarputhalapattu/projects/tree/main/ipl%202024)")
                with st.expander("### Features",expanded=False):
                    st.write("""
                            - **Match Analysis**: Insights into team rankings , player stats , and match outcomes.
                            - **Player Analysis**: Insights into individual players' performances , Trends and prediction based on player stats.
                            - **Team Analysis**: Insights into team-level performance, Analysis of team performance under various conditions.
                            - **Data Visualization**: Visual representations of the data.
                            """)
    # 1
    with st.container():
        st.write("---")
        st.write("##")
        img_col,text_col = st.columns((1,2))
        with img_col:
            lottie_url = "https://lottie.host/4df30870-cd21-475d-925b-aed188cdfd84/6lmL3rWmee.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st.lottie(lottie_animation,speed = 1, height =300,key="project6")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### UCI Bank Marketing Analysis")
            st.write("""
                    - Conducted a detailed analysis of the UCI Bank Marketing Dataset to understand factors influencing term deposit subscriptions.
                    - Applied clustering and feature analysis to identify key attributes impacting client decisions.
                    - Generated actionable insights for optimizing marketing strategies.
                    """)
            if st.button("Know More🏦"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- **Pandas**")
                        st.write("- **Matplotlib**")
                        st.write("- **Scikit-learn**")
                    with col2:
                        st.write("- k-means clustering")
                        st.write("- **Pythons**")
                        st.write("- **Numpy**")
                
                with st.expander("### Dataset Used ", expanded = False):
                    st.write("[Access the UCI bank marketing ](https://github.com/sagarputhalapattu/projects/tree/main/UCI-Bank-Marketing-Analysis-main)")
                with st.expander("### Features",expanded=False):
                    st.write("""
                            - **Clustering Analysis**: Identified patterns is customer segmentation.
                            - **Feature Importance**: Highlighted key factors influencing term deposit subscription.
                            - **Actionable Insights**:Provide data-driven recommendation for marketing campaigns.
                            - **Interactive Dashboards**: Visualized customer trends and segmentation result.
                            """)
    #2
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/be6d55f4-8f2f-4357-b7d3-17af381944ba/4P0UceHfbc.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project7")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Demand Forecasting for a Retail Store")
            st.write("""
                    - Developed a demand forecasting model to predict product demand using time series analysis and machine learning techniques.
                    - The goal is to optimize stock levels, improve supply chain efficiency, and reduce overstocking/understocking.
                    - This model helps the retail store improve inventory management by predicting future demand trends based on historical data.
                    """)
            if st.button("Know More 🙂"):
                # Expander for technologies used
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("- **Time Series Analysis** for demand prediction.")
                        st.write("- **ARIMA** and **Prophet** for forecasting.")
                        st.write("- **Python** for implementation.")
                    with col2:
                        st.write("- **Scikit-learn** for machine learning models.")
                        st.write("- **Pandas** and **Numpy** for data processing.")
                        st.write("- **Matplotlib/Seaborn** for data visualization.")
                    # Expander for project link
                with st.expander("### Project Link", expanded=False):
                    st.write("[Explore the Demand Forecasting Project](https://github.com/sagarputhalapattu/projects/blob/main/Demand%20Forcasting%20for%20a%20Retail%20Store.ipynb)")
                    # Expander for features
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Demand Prediction**: Uses historical sales data to forecast demand for future time periods.
                            - **Inventory Optimization**: Helps the store determine the best stock levels based on predicted demand.
                            - **Visualization**: Visualizes sales trends and predictions using interactive charts.
                            """)
                    # Footer or additional sections
                    st.write("---")
                    st.write("#### Want to learn more or collaborate on this project? Feel free to reach out!")
                    
    # 3
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/4dd01d2f-a68f-4993-96e8-7d69cfaf450f/fj9nHBnkv6.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project8")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Credit Fraud Detection with Imbalanced Datasets")
            st.write("""
                    - Addressed the challenge of detecting fraud in imbalanced datasets using machine learning models.
                    - Applied oversampling techniques such as **SMOTE** and ensemble models to improve classification performance.
                    - This project aims to help financial institutions accurately identify fraudulent transactions despite the data imbalance.
                    """)
            if st.button("Know More 💳"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Ensemble Models** (e.g., Random Forest) for better fraud detection.")
                    st.write("- **SMOTE** for oversampling the minority class (fraudulent transactions).")
                    st.write("- **Python** for implementation.")
                with col2:
                    st.write("- **Pandas** for data processing.")
                    st.write("- **Scikit-learn** for machine learning model building.")
                    st.write("- **Matplotlib/Seaborn** for visualizing model performance.")
                # Expander for project link
                with st.expander("### Project Link", expanded=False):
                    st.write("[Explore the Credit Fraud Detection Project](https://github.com/sagarputhalapattu/projects/tree/main/Credit%20Fraud%20Detection%20with%20Imbalanced%20Datasets)")
                    # Expander for features
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Fraudulent Transaction Detection**: Uses classification algorithms to identify fraud.
                            - **Imbalanced Data Handling**: Utilizes SMOTE to balance the class distribution.
                            - **Model Performance**: Evaluates models using metrics like precision, recall, and F1-score due to the imbalanced dataset.
                            """)
                        # Footer or additional sections
                    st.write("---")
                    st.write("#### Want to learn more or collaborate on this project? Feel free to reach out!")
                    
    # 4
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/d7c1276a-c4c0-4bb7-aee7-960ccc08ee46/Hco6MJoSR3.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project9")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Telecom Customer Churn Prediction")
            st.write("""
                    - Used machine learning algorithms to predict customer churn in a telecom company.
                    - The model focuses on improving customer retention strategies by identifying at-risk customers based on historical data.
                    - Features such as customer demographics, usage patterns, and subscription details were used to build the model.
                    """)
            if st.button("Know More📱"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Logistic Regression** for churn prediction.")
                    st.write("- **Random Forest** for better performance.")
                    st.write("- **Python** for implementation.")
                with col2:
                    st.write("- **Pandas** for data manipulation.")
                    st.write("- **Matplotlib/Seaborn** for visualizing churn patterns.")
                    st.write("- **Scikit-learn** for modeling.")
                with st.expander("### Project Link", expanded=False):
                    st.write("[Explore the Telecom Churn Prediction Project](https://github.com/sagarputhalapattu/projects/tree/main/Telecom_Customer_Churn_Prediction_updated_05_09)")
                with st.expander("### Features", expanded=False):
                    st.write("""
                            - **Churn Prediction**: Identifies which customers are most likely to leave the telecom service.
                            - **Customer Segmentation**: Segments customers based on their risk of churn.
                            - **Retention Strategies**: Helps the company create personalized offers to retain at-risk customers.
                        """)
                st.write("---")
                st.write("#### Want to learn more or collaborate on this project? Feel free to reach out!")
    # 5
    with st.container():
        st.write("---")
        st.write("##")
        img_col, text_col = st.columns((1, 2))
        with img_col:
            lottie_url = "https://lottie.host/41d2037f-547f-4099-aae3-e5aab3e4191e/Ea0Nbdh32e.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="project10")
            else:
                st.error("Failed to load Lottie animation.")
        with text_col:
            st.write("### Visa Dataset")
            st.write("""
                    - Analyzed transaction data to detect fraudulent activities using machine learning techniques.
                    - Applied classification models to identify transactions that could potentially be fraudulent.
                    - The goal was to help banks and financial institutions reduce fraud and improve security measures.
                    """)
            if st.button("Know More 🛩️"):
                with st.expander("### Technologies Used", expanded=False):
                    col1, col2 = st.columns(2)
                with col1:
                    st.write("- **Random Forest Classifier** for fraud detection.")
                    st.write("- **SMOTE** for handling imbalanced datasets.")
                    st.write("- **Python** for implementation.")
                with col2:
                    st.write("- **Pandas** for data manipulation.")
                    st.write("- **Scikit-learn** for model building.")
                    st.write("- **Matplotlib/Seaborn** for visualizing fraud patterns.")
                with st.expander("### Project Link", expanded=False):
                    st.write("[Explore the Visa Dataset Project](https://github.com/sagarputhalapattu/projects/tree/main/Visa%20Dataset)")
                with st.expander("### Features", expanded=False):
                    st.write("""
                             - **Fraud Detection**: Predicts fraudulent transactions based on historical data.
                            - **Imbalanced Dataset Handling**: Utilizes techniques like SMOTE to balance the dataset.
                            - **Transaction Analysis**: Analyzes transaction details such as amounts, merchant info, and time stamps.
                            """)
                    st.write("---")
                    st.write("#### Want to learn more or collaborate on this project? Feel free to reach out!")


    
# Achievements Function --------------------------------------------------------------------------------------------------------------
def Achievements():
    with st.container():
        st.write("# 🏆 My Achievements")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                        - 🧠 Solved 300+ questions on LeetCode. I'm on a roll!
                        - Focused on **data analysis** and developing insights to align with **business analyst responsibilities** 
                    """)
            st.markdown("""
                        I believe that every achievement, big or small, is a stepping stone towards success. I am always eager to learn and grow, and I am excited to see what the future holds for me. 🚀
                        """)
        with col2:
            lottie_url = "https://lottie.host/a7990eb1-5152-4cd1-a7fd-e0c592683b97/KOvlwaJ4uL.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="achievements")
            else:
                st.error("Failed to load Lottie animation.")
        st.write("---")
        st.write("# 📚 I Like To Talk About")
        st.markdown("""
                    - **Data Science**
                    - **Machine Learning**
                    - **Deep Learning**
                    - **Artificial Intelligence**
                    - **Computer Vision**
                    - **Natural Language Processing**
                    - **Data Structures & Algorithms**
                    - **Self-Improvement** 
                    """)
        st.write("---")
        
        with st.container():
            st.write("## 📜 **Certifications**")
            
            # Certification 1: Azure AI Engineer Associate
        with st.expander("### 🎓 **Microsoft Certified: Azure AI Engineer Associate**", expanded=False):
            st.write("""
                     - **Issued by**: Microsoft
                     - **Skills Acquired**:
                    - Building and managing AI solutions using Microsoft Azure.
                    - Integrating AI models into applications and deploying them at scale.
                    """)
            # Certification 2: Data Analyst by Cisco
        with st.expander("### 🎓 **Data Analyst Certification**", expanded=False):
            st.write("""
                    - **Issued by**: Cisco
                    - **Skills Acquired**:
                    - Data analysis and visualization techniques.
                    - Proficiency in statistical tools and business insights.
                    """)
            # Certification 3: Generative AI by Great Learning
        with st.expander("### 🎓 **Generative AI Certification**", expanded=False):
            st.write("""
                    - **Issued by**: Great Learning
                    - **Skills Acquired**:
                        - Fundamentals of generative AI models.
                        - Applications of generative AI in industries like gaming and healthcare.
                    """)
            st.write("[🔗 For more details, feel free to reach out!](https://github.com/sagarputhalapattu/Certification)")


# Skills Function --------------------------------------------------------------------------------------------------------------
def Skills():
    
    with st.container():
        st.write("### 💼 My Skills")
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                    - **Languages**: Python, SQL, CSS, HTML, JavaScript
                    - **Libraries**: Pandas, Numpy, Matplotlib, Plotly, Seaborn
                    - **Frameworks**: Streamlit, Bootstrap CSS
                    - **Tools**: Git, Docker, Jupyter Notebook, VS Code
                    - **Databases**: MySQL Server, Vector Databasess
                    - **Machine Learning**: Scikit-learn & Machine Learining Algorithms
                    - **Deep Learning**: Tensorflow, Keras, OpenCV, NLP & it's Libraries 
                    - **Soft Skills**: Teamwork, Communication, Problem-Solving, Time Management, Adaptability
                    - **Others**: Data Structures & Algorithms, Computer Vision, Natural Language Processing
                    - **Testing**: Post-man , MLOPS 
                    - **Visualization**: Microsoft PowerBi ,Tableau , statistical analysis
                    - **Business Skills**: Finance, Marketing, Sales
                    
                """)
        with col2:
            lottie_url = "https://lottie.host/e0f72cc9-db24-4eac-a85e-19bb821629f9/mumuEVoTHL.json"
            lottie_animation = load_lottieurl(lottie_url)
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, height=300, key="skills")
            else:
                st.error("Failed to load Lottie animation.")

# Setting Sidebar Main Menu ------------------------------------------------------------------------------------------------------------
with st.sidebar:
    selected_page = option_menu(
        "Navigate Here", 
        ["Home", "Projects", "Achievements","Skills"],
        icons = ['house', 'braces', 'trophy', 'code'],
        menu_icon = "cast",
        default_index = 0,
        )
    


# Displaying Selected Page --------------------------------------------------------------------------------------------------------------

if selected_page == "Home":
    Home()
elif selected_page == "Projects":
    Projects()
elif selected_page == "Achievements":
    Achievements()
elif selected_page == "Skills":
    Skills()
    


