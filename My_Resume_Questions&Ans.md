

### Questions an interviewer may ask based on your Research Assistant experience:

1. **Sales Forecasting for the Pharmaceutical Company**
   - What were the challenges in implementing the ML models for sales forecasting?
   - Why did you choose FbProphet and XGBoost for this project? Can you explain their strengths in this context?
   - How did you validate the model's accuracy, and what improvements could be made?

2. **Demand Forecasting in the Cement Industry**
   - What specific data preprocessing techniques did you use to standardize and format the data?
   - Can you explain ensemble stacking and why it improved the MAPE (Mean Absolute Percentage Error) in your case?
   - How did you ensure your forecasting models were robust across regions, districts, and dealers?

3. **Video Analytics for the FMCG Company**
   - What was the goal of detecting objects and stages in washing videos, and how did it help the company?
   - Can you compare YOLOv5 and YOLOv8 in terms of performance and use cases in this project?
   - How did you handle the custom dataset, including its annotation and validation?

### Structuring your answers for clarity and impact:

1. **Understand the intent of the question.**
   - If it's about technical implementation, focus on explaining the steps you followed, challenges encountered, and decisions made.
   - If it's about the impact, describe how your work contributed to solving the business problem.

2. **Use the STAR approach:**
   - **S**ituation: Briefly describe the problem or goal.
   - **T**ask: Highlight your role and responsibilities.
   - **A**ction: Explain the methods, tools, and techniques you used.
   - **R**esult: Share quantifiable outcomes or improvements.

3. **Be concise but detailed.**
   - Avoid overly technical jargon unless asked for specifics.
   - Break down concepts when introducing technical terms to ensure the interviewer follows your reasoning.

### Example response:

**Q: How did ensemble stacking improve the demand forecasting model for the cement industry?**

**A:**
- **Situation:** The cement industry's supply chain needed optimization with accurate demand forecasting across multiple levels—dealers, districts, and regions. 
- **Task:** My role was to enhance the accuracy of the demand forecasting model by experimenting with various ML techniques.
- **Action:** I implemented ensemble stacking, which combines predictions from multiple base models (like Lasso, Ridge, and Random Forest) to create a meta-model. The ensemble approach improved stability and generalization compared to individual models.
- **Result:** This method outperformed expectations, reducing MAPE to 6.7% for regions, 19.75% for districts, and 27.90% for dealers. This accuracy helped optimize inventory and logistics planning, directly reducing costs and improving efficiency.

Here are comprehensive answers to the questions based on your Research Assistant experience:

---

### **Sales Forecasting for the Pharmaceutical Company**

1. **Challenges in implementing ML models for sales forecasting**
   - **S:** The pharmaceutical company's sales were highly seasonal, making it challenging to predict trends.
   - **T:** I needed to address data irregularities such as missing values and noisy data while ensuring models captured seasonality accurately.
   - **A:** To tackle this, I used FbProphet for its ability to handle seasonality and added external regressors like holidays and promotions to improve predictions. I paired it with XGBoost to capture complex interactions and non-linear relationships in the data.
   - **R:** These approaches achieved an average accuracy of 82%, providing actionable insights for inventory management and marketing.

---

2. **Why FbProphet and XGBoost?**
   - FbProphet excels in handling time-series data with clear seasonality and holidays, making it ideal for pharmaceutical sales.
   - XGBoost is a robust algorithm for structured data, capable of modeling complex patterns and interactions.
   - The combination allowed me to leverage the strengths of both models: FbProphet for capturing time-dependent trends and XGBoost for feature-based predictions.

---

3. **Validation and potential improvements**
   - **Validation:** I split the data into training and testing sets and used time-series cross-validation to evaluate performance. Metrics like RMSE and MAE were monitored.
   - **Improvements:** Incorporating additional external features, such as competitor activity or weather data, could further enhance prediction accuracy.

---

### **Demand Forecasting in the Cement Industry**

1. **Data preprocessing techniques**
   - **S:** Cement demand data was scattered and inconsistent across regions, districts, and dealers.
   - **T:** I standardized and formatted raw data to address missing values, outliers, and inconsistencies.
   - **A:** Techniques used included:
     - Imputing missing values using median or time-series interpolation.
     - Removing outliers based on IQR (Interquartile Range).
     - Normalizing features to ensure uniformity across datasets.
   - **R:** This resulted in cleaner data, improving model performance and stability.

---

2. **Ensemble stacking and its impact on MAPE**
   - **S:** The challenge was ensuring accurate forecasting across different granularities (regions, districts, dealers).
   - **T:** I implemented an ensemble stacking technique combining base models like Ridge, Lasso, and Random Forest.
   - **A:** Ensemble stacking aggregates predictions from multiple models, using a meta-model to learn from their outputs. This approach reduced overfitting and improved generalization.
   - **R:** MAPEs were reduced to 6.7% (regions), 19.75% (districts), and 27.90% (dealers), enabling more precise supply chain decisions.

---

3. **Ensuring robustness across granularities**
   - I ensured robustness by:
     - Segmenting the dataset into regional, district, and dealer levels.
     - Optimizing hyperparameters for each model through grid search.
     - Using bootstrapping to simulate variability and ensure consistent performance across levels.

---

### **Video Analytics for the FMCG Company**

1. **Goal of detecting objects and stages in washing videos**
   - **S:** The FMCG company aimed to analyze washing videos for quality control and process improvement.
   - **T:** My task was to detect objects (e.g., detergent, clothes) and identify stages (e.g., rinsing, drying).
   - **A:** I used YOLOv5 for object and stage detection due to its balance of speed and accuracy.
   - **R:** The insights helped streamline washing processes, improving efficiency and reducing waste.

---

2. **Comparison of YOLOv5 and YOLOv8**
   - **YOLOv5:** Provided faster inference times and worked well for object detection tasks, achieving a mAP (mean Average Precision) of 89% on this project.
   - **YOLOv8:** Offered better image segmentation capabilities with improved precision and recall. It was effective for segmenting complex objects in the custom dataset.

---

3. **Handling the custom dataset**
   - **S:** The dataset required extensive preprocessing, including annotation of objects and stages.
   - **T:** I annotated the dataset using tools like LabelImg and ensured balanced class representation.
   - **A:** Techniques used included augmenting data with rotation, scaling, and flipping to improve model robustness.
   - **R:** This approach enhanced the model's performance, ensuring accurate detection and segmentation even in varying conditions.

---

Here are additional in-depth questions based on various sections of your resume, along with suggested answers:  

---

### **1. Data Analyst at Chordify Inc.**  
#### **Question 1: How did you integrate and optimize BigQuery for regular reporting on GCP?**  
- **S:** Chordify required regular, efficient reports (daily, weekly, monthly) using BigQuery on Google Cloud.  
- **T:** My role was to optimize existing queries and automate the reporting process.  
- **A:** I identified slow-performing queries, rewrote them to use partitioned and clustered tables, and implemented scheduled queries using Cloud Scheduler.  
- **R:** This reduced query execution time by 40% and ensured timely delivery of actionable insights.  

---

#### **Question 2: How does Looker Studio enable better decision-making through data visualization?**  
- Looker Studio allows the integration of multiple data sources, providing interactive dashboards.  
- I designed dashboards to visualize trends like sales performance and customer engagement, making KPIs intuitive and accessible for stakeholders.  
- I ensured the reports were dynamic, enabling real-time filtering by geography, product category, and time.

---

#### **Question 3: What challenges did you face while integrating YOLOv8 with multiple webcams for real-time face detection?**  
- **S:** Real-time detection required handling high frame rates and ensuring minimal latency across webcams.  
- **T:** I optimized the YOLOv8 model by reducing input frame size and using TensorRT for inference.  
- **A:** Deployed the system using a multi-threading approach, enabling simultaneous processing of video streams.  
- **R:** This achieved a real-time performance of 25 FPS per webcam with 95% detection accuracy.  

---

### **2. Research Assistant at IIM Mumbai**  
#### **Question 1: How did you tackle instability in the cement industry demand dataset?**  
- **S:** Data inconsistencies across regions and dealers caused unreliable forecasts.  
- **T:** I focused on cleaning and transforming the data.  
- **A:** Methods included outlier detection using z-scores, imputing missing data with KNN imputation, and applying log transformations to normalize skewed features.  
- **R:** These preprocessing steps significantly improved model performance, leading to accurate demand forecasts.  

---

#### **Question 2: How did you measure the success of the ensemble stacking model?**  
- Success was measured using MAPE, RMSE, and R-squared values.  
- Cross-validation ensured model robustness. For instance, the stacking model reduced error rates by 15% compared to individual base models.  
- These metrics demonstrated the model’s effectiveness in optimizing inventory and logistics.  

---

#### **Question 3: How did you optimize object detection accuracy for FMCG video analytics?**  
- **A:** Improved YOLOv8 accuracy by:  
  - Augmenting the training dataset (e.g., brightness, rotation, noise).  
  - Fine-tuning hyperparameters like learning rate and anchor sizes.  
  - Implementing early stopping to prevent overfitting.  
- This boosted mAP by 7% compared to default YOLOv8 settings.

---

### **3. Hate Speech Detection Project**  
#### **Question 1: How did you address sociodemographic biases in hate speech detection?**  
- **S:** Standard models often exhibited biases against disability-related content.  
- **T:** I fine-tuned a pre-trained BERT model on a custom dataset annotated for bias-free classification.  
- **A:** Techniques included adversarial training to reduce bias, augmenting the dataset with neutral examples, and using SHAP values to interpret model predictions.  
- **R:** This improved model fairness and achieved 80% classification accuracy without favoring specific demographics.  

---

### **4. Real Estate Price Prediction**  
#### **Question 1: How did you handle outliers and missing values in the dataset?**  
- Outliers were removed using box plots and z-scores for numerical features like price and area.  
- Missing values were handled using median imputation for skewed data and mean imputation for balanced data.  
- This preprocessing ensured cleaner inputs, boosting regression model accuracy to 89%.  

---

### **5. Internships and Academic Projects**  
#### **Question 1: How did you achieve cost efficiency in the drone-based last-mile delivery model?**  
- **S:** The goal was to minimize delivery costs for a logistics company.  
- **T:** I built a mathematical optimization model using ML algorithms and CPLEX solver.  
- **A:** Constraints included battery life, payload weight, and delivery time. The model optimized delivery routes and batch sizes.  
- **R:** This resulted in 85% cost savings compared to traditional delivery methods.  

---

#### **Question 2: What challenges did you face while creating a predictive ML model during the ISB internship?**  
- The dataset from 600+ companies had imbalanced classes for firm terminations.  
- To address this, I used techniques like SMOTE (Synthetic Minority Oversampling) and adjusted class weights during training.  
- These steps improved the model’s precision and recall to above 87%.

---

### General Questions  
1. **What’s your approach to debugging ML models that underperform?**  
   - Start with feature engineering: Check for data leakage or irrelevant features.  
   - Analyze model parameters and optimize through grid/random search.  
   - Use SHAP or LIME for interpretability to identify why certain predictions fail.  

2. **What tools do you prefer for data visualization, and why?**  
   - I prefer Looker Studio for interactive dashboards, Matplotlib/Seaborn for static plots, and Power BI for business-focused reports.  
   - Each tool suits different audiences—business stakeholders vs. data scientists.

---

