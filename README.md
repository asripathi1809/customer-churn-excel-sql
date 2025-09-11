# customer-churn-excel-sql
**Customer Churn Analysis — Telecom Industry**

**Overview:**

Customer churn is a critical revenue threat in the telecom industry, with even small losses of subscribers costing companies billions annually. Understanding which customer segments are at highest risk and quantifying the financial impact of retention efforts can help companies reduce churn, boost loyalty, and drive growth.

To tackle this, I developed a comprehensive churn prediction system using the IBM Telecom Customer Churn dataset (7,000+ customers). By combining SQL-driven data exploration, advanced feature engineering, and machine learning models, I identified high-risk segments and derived actionable strategies to improve retention and revenue.

**Problem Statement:**

**Challenge:** Millions lost annually as customers leave telecom providers

**Impact:** Revenue loss, decreased customer lifetime value, and increased acquisition costs

**Goal:** Identify high-risk customer segments and quantify retention impact using predictive analytics

**Data & Tools Used:**

**Dataset:** IBM Telecom Customer Churn Dataset (7,000+ customer records)

**Tools:** Python (data cleaning, modeling), Scikit-learn (ML algorithms), Excel/CSV (visualization), SQL (data querying)

**Approach & Process:**

**Data Cleaning:** Handled missing values, converted types, encoded categorical variables

**Exploratory Data Analysis:** Visualized churn patterns highlighting key risk factors

**Feature Engineering:** Created tenure segments and billing cycle features to improve prediction accuracy

**Modeling:** Built and compared Logistic Regression, Random Forest, XGBoost, and Decision Trees with 5-fold cross-validation

**Validation:** Used confusion matrices and precision/recall metrics to address class imbalance

**Key Insights:**

Customers with tenure **less than 12 months** show significantly higher churn

Month-to-month contract users have **elevated churn risk** compared to longer contracts

Customers paying via **electronic checks churn more, possibly due to payment system issues**

**Actionable Recommendations:**

Design targeted retention campaigns for customers with **less than 1 year tenure**

Focus on **month-to-month subscribers with special offers or loyalty programs**

**Improve payment system reliability or provide alternative payment options for electronic check users**

**Business Impact:**

High-risk customers identified: 896

ROI: 7.1x on retention efforts

Potential savings: Millions by reducing churn and improving customer lifetime value

**Challenges:**

Extracting impactful SQL queries balancing business KPIs and modeling needs

Encoding categorical variables and handling class imbalance to ensure unbiased model evaluation

**What’s Next?**

Incorporate customer support interaction and complaint data for enhanced prediction

Experiment with ensemble and deep learning models to improve accuracy and robustness

**Repository & Dashboard:**

Explore the full code, notebooks, and dashboards here:
https://github.com/asripathi1809

**Conclusion:**

This project leverages data science to solve a crucial telecom business problem by identifying churn risks early and enabling targeted retention strategies. The outcome is a data-driven system that not only saves millions but also strengthens customer relationships and competitive advantage—showcasing the future of telecom success.
