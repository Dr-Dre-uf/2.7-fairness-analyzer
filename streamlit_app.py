import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="Module 2: Sex-Specific Modeling", layout="wide")

@st.cache_data
def build_eicu_data():
    """Loads and merges the raw eICU data."""
    # Load patient information
    patient_cols = ['patientunitstayid', 'hospitalid', 'gender', 'age', 'ethnicity', 'admissionheight', 'admissionweight', 'dischargeweight',
                'hospitaladmitsource', 'hospitaldischargelocation', 'hospitaldischargestatus', 'unittype', 'uniquepid', 'unitvisitnumber',
                'patienthealthsystemstayid', 'hospitaldischargeyear']
    df = pd.read_csv('https://www.dropbox.com/scl/fi/qld4pvo6vlptm41av3y2e/patient.csv?rlkey=gry21fvb3u3dytz7i5jcujcu9&dl=1',
                      usecols=patient_cols)
    
    # Clean Age for numerical filtering
    df['age'] = df['age'].replace({'> 89': 90})
    df['age'] = pd.to_numeric(df['age'], errors='coerce') 
    
    df = df.sort_values(by=['uniquepid', 'patienthealthsystemstayid', 'unitvisitnumber'])
    df = df.groupby('patienthealthsystemstayid').first().reset_index()
    df = df.drop(columns=['unitvisitnumber'])

    # Load hospital information
    hospital = pd.read_csv('https://www.dropbox.com/scl/fi/5sdjsbrjxk0hlbmpb4csi/hospital.csv?rlkey=rmlrhg3m9sm3hj2s6rykrg5w2&st=329j4gwk&dl=1')
    with pd.option_context('future.no_silent_downcasting', True):
        hospital = hospital.replace({'teachingstatus': {'f': 0, 't':1}})

    df = df.merge(hospital, on='hospitalid', how='left')

    # Load labs
    labcols = ['patientunitstayid', 'labname', 'labresult']
    labnames = ['BUN', 'creatinine', 'sodium', 'Hct', 'wbc', 'glucose', 'potassium', 'Hgb', 'chloride', 'platelets',
                'RBC', 'calcium', 'MCV', 'MCHC', 'bicarbonate', 'MCH', 'RDW', 'albumin']
    labs = pd.read_csv('https://www.dropbox.com/scl/fi/qaxtx330hicc5u61siehn/lab.csv?rlkey=xs9oxpl5istkbuh5s80oyxwwi&st=ydfrjxkh&dl=1',
                        usecols=labcols)
    labs['labname'] = labs['labname'].replace({
        'WBC x 1000': 'wbc',
        'platelets x 1000': 'platelets'
    })
    labs = labs[labcols]
    labs = labs[labs['labname'].isin(labnames)]
    labs = labs.pivot_table(columns=['labname'], values=['labresult'], aggfunc='mean', index='patientunitstayid')
    labs.columns = list(labs.columns.droplevel(0))
    labs = labs.reset_index()
    labnames = ['lab_' + c.lower() for c in labnames]
    labs.columns = ['patientunitstayid'] + labnames

    df = df.merge(labs, on='patientunitstayid', how='left')

    # Renaming
    df = df.rename(columns={
        'uniquepid': 'patient_id',
        'patienthealthsystemstayid': 'admission_id',
        'hospitaldischargeyear': 'admission_year',
        'hospitalid': 'hospital_id',
        'admissionheight': 'height',
        'admissionweight': 'weight_admission',
        'dischargeweight': 'weight_discharge',
        'region': 'hospital_region',
        'teachingstatus': 'hospital_teaching',
        'numbedscategory': 'hospital_beds',
        'hospitaladmitsource': 'admission_source',
        'hospitaldischargelocation': 'discharge_location'
    })

    df['in_hospital_mortality'] = df['hospitaldischargestatus'].map(lambda status: {'Alive': 0, 'Expired': 1}[status]
                                                                        if pd.notnull(status) else status)
    df = df.drop(columns=['hospitaldischargestatus'])

    df_cols = ['patient_id', 'admission_id', 'admission_year', 'age', 'gender', 'ethnicity', 'height', 'weight_admission',
               'weight_discharge', 'admission_source', 'hospital_id', 'hospital_region', 'hospital_teaching',
               'hospital_beds'] + labnames + ['discharge_location', 'in_hospital_mortality']
    df = df[df_cols]
    df = df.sort_values(by=['admission_year', 'patient_id', 'admission_id'])
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def get_processed_data(df_raw):
    """Runs the preprocessing steps (One-Hot Encoding & Imputation)."""
    # Drop IDs for processing
    x = df_raw.drop(columns=['patient_id', 'hospital_id', 'admission_id', 'admission_year', 'weight_discharge', 'discharge_location'])
    
    numerical_features = ['age', 'height','weight_admission', 'hospital_teaching', 'lab_bun','lab_creatinine','lab_sodium',
                      'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride','lab_platelets','lab_rbc','lab_calcium','lab_mcv',
                      'lab_mchc','lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']

    categorical_features = ['ethnicity', 'admission_source', 'hospital_region', 'hospital_beds']
    
    x[numerical_features] = x[numerical_features].apply(pd.to_numeric, errors='coerce', axis=1)
    x[numerical_features] = x[numerical_features].fillna(x[numerical_features].mean())
    
    # Standard preprocessing
    x = pd.get_dummies(x, columns=categorical_features, dtype='int')
    x['in_hospital_mortality'] = x['in_hospital_mortality'].fillna(0)
    
    return x

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Module Navigation")
page = st.sidebar.radio("Go to:", 
    ["1. Data Processing", 
     "2. Exploratory Analysis",
     "3. Univariate Analysis",
     "4. Multivariate Analysis"])

st.sidebar.info("Use the menu above to follow the module structure.")

# --- 3. PAGE LOGIC ---

# === 1. DATA PROCESSING ===
if page == "1. Data Processing":
    st.title("Data Processing")
    
    st.markdown("""
    ### Removing Irrelevant Columns
    One of the first logical steps is to remove the columns that contain no informational value. Some columns are unique random numeric identifiers (e.g., patient_id) with no discernible meaning. 
    
    We'll also remove weight_discharge and discharge_location because they will not be used as inputs.
    
    ### Fast Preprocessing
    We will save time with a heavy-handed approach:
    1. One-hot encode all categorical variables.
    2. Impute missing values in each numerical column with the mean.
    """)

    # Initialize state
    if 'proc_run' not in st.session_state:
        st.session_state.proc_run = False
        
    with st.spinner("Loading Raw Data..."):
        df_raw = build_eicu_data()

    if st.button("Run Preprocessing Pipeline", help="Clean the data, impute missing values, and prepare it for modeling."):
        st.session_state.proc_run = True

    if st.session_state.proc_run:
        with st.spinner("Processing..."):
            df_processed = get_processed_data(df_raw)
            
        st.success("Preprocessing Complete!")
        st.write(f"Processed Data Shape: {df_processed.shape}")
        st.dataframe(df_processed.head())
        
        csv = df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Processed CSV", 
            csv, 
            "eicu_processed.csv", 
            "text/csv", 
            key='download-csv-processed'
        )

# === 2. EXPLORATORY ANALYSIS ===
elif page == "2. Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.markdown("""
    ### Visualize the sex-specific patterns
    Before jumping into modeling, it is important to ask: "Do males and females behave differently in this data?" In many clinical datasets, combining all patients into a single analysis can blur important differences. Let’s split the lens and take a closer look.
    """)

    with st.spinner("Preparing Data..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

    # --- VISUALIZATION ---
    st.subheader("Interactive Statistics")
    available_vars = ['age', 'height','weight_admission', 'lab_bun','lab_creatinine','lab_sodium',
                      'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride',
                      'lab_platelets','lab_rbc','lab_calcium','lab_mcv','lab_mchc',
                      'lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']
    
    selected_variables = st.multiselect(
        "Select Variables to Compare:", 
        available_vars, 
        default=['lab_bun','lab_creatinine','lab_sodium'],
        help="Select variables to compare mean values between men and women, split by survival."
    )

    if selected_variables:
        df_male = df_processed[df_processed['gender'] == "Male"]
        df_female = df_processed[df_processed['gender'] == "Female"]

        female_survived = [df_female.loc[df_female['in_hospital_mortality'] == 0][str(i)].mean() for i in selected_variables]
        female_dead = [df_female.loc[df_female['in_hospital_mortality'] == 1][str(i)].mean() for i in selected_variables]
        male_survived = [df_male.loc[df_male['in_hospital_mortality'] == 0][str(i)].mean() for i in selected_variables]
        male_dead = [df_male.loc[df_male['in_hospital_mortality'] == 1][str(i)].mean() for i in selected_variables]

        Year = ['Survival'] * len(female_survived) + ['In hospital mortality'] * len(female_dead)
        Female = female_survived + female_dead
        Male = male_survived + male_dead
        
        df_plot = pd.DataFrame({'index': selected_variables * 2, 'Year': Year, 'Female': Female, 'Male': Male})
        df_plot.set_index(['Year', 'index'], inplace=True)
        df0 = df_plot.reorder_levels(['index', 'Year']).sort_index().unstack(level=-1)

        colors = plt.cm.Paired.colors
        fig, ax = plt.subplots(figsize=(10, 6))
        (df0['Female'] + df0['Male']).plot(kind='barh', color=[colors[3], colors[2]], rot=0, ax=ax)
        df0['Male'].plot(kind='barh', color=[colors[5], colors[4]], rot=0, ax=ax)
        ax.legend([f'{val} ({context})' for val, context in df0.columns])
        ax.set_title("Mean Values by Mortality Status (Stacked by Sex)")
        st.pyplot(fig)

    # --- QUESTION 1 ---
    st.divider()
    st.subheader("Question 1")
    st.markdown("Why is it important to analyze clinical data separately for males and females before modeling?")
    
    q1_options = {
        "A": "To reduce the number of observations",
        "B": "To make the dataset more complex",
        "C": "To identify sex-specific patterns that might be masked in pooled data",
        "D": "To apply the same model to both groups without changes"
    }
    
    q1_choice = st.radio("Select Answer:", list(q1_options.keys()), format_func=lambda x: f"{x}) {q1_options[x]}", key="q1")
    
    if st.button("Submit Question 1"):
        if q1_choice == "C":
            st.success("Correct! Clinical data often contains meaningful differences between males and females. When we analyze the entire dataset as a single group, these differences can get averaged out or hidden.")
        else:
            st.error("Try again.")

# === 3. UNIVARIATE ANALYSIS ===
elif page == "3. Univariate Analysis":
    st.title("Sex-specific Association Models")
    st.markdown("""
    We will now perform a univariate analysis using odds ratios separately for females and males. 
    
    By calculating odds ratios within each sex, we can uncover whether a variable (like elevated lactate) has differential predictive power for mortality in women versus men.
    """)

    with st.spinner("Preparing Data & Models..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

        df_male = df_processed[df_processed['gender'] == "Male"]
        df_female = df_processed[df_processed['gender'] == "Female"]

        train_dat_female, test_dat_female = train_test_split(df_female, test_size=0.2, random_state=2025)
        train_dat_male, test_dat_male = train_test_split(df_male, test_size=0.2, random_state=2025)
        train_dat_full, test_dat_full = train_test_split(df_processed, test_size=0.2, random_state=2025)

    all_vars = ['age', 'height','weight_admission', 'lab_bun','lab_creatinine','lab_sodium',
                'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride','lab_platelets','lab_rbc','lab_calcium','lab_mcv',
                'lab_mchc','lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']
    
    st.subheader("Calculate Odds Ratios")
    selected_variables = st.multiselect("Select Variables:", all_vars, default=all_vars[:10])

    if st.button("Run Regression Models"):
        vals_male, vals_female, vals_full = [], [], []
        
        progress_bar = st.progress(0)
        for i, univ_analysis in enumerate(selected_variables):
            progress_bar.progress((i + 1) / len(selected_variables))
            
            # Male
            reg_male = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_male).fit(disp=0)
            vals_male.append(np.exp(reg_male.params).iloc[1]) # Get OR for the variable
            
            # Female
            reg_female = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_female).fit(disp=0)
            vals_female.append(np.exp(reg_female.params).iloc[1])
            
            # Full
            reg_full = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_full).fit(disp=0)
            vals_full.append(np.exp(reg_full.params).iloc[1])

        progress_bar.empty()

        fig, ax = plt.subplots(figsize=(12, len(selected_variables) * 0.6 + 2))
        bar_width = 0.25
        r1 = np.arange(len(selected_variables))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        ax.barh(r1, vals_female, bar_width, label='Female', color='tab:orange', alpha=0.8)
        ax.barh(r2, vals_male, bar_width, label='Male', color='tab:blue', alpha=0.8)
        ax.barh(r3, vals_full, bar_width, label='All Cohort', color='tab:green', alpha=0.8)
        
        ax.set_xlabel('Odds Ratio (OR)')
        ax.set_yticks([r + bar_width for r in range(len(selected_variables))])
        ax.set_yticklabels(selected_variables)
        ax.set_title('Univariate Analysis: Odds Ratios')
        ax.legend()
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)
        st.pyplot(fig)

# === 4. MULTIVARIATE ANALYSIS ===
elif page == "4. Multivariate Analysis":
    st.title("Performance Evaluation by Sex")
    st.markdown("""
    Now we will evaluate how well our models perform not just overall, but within each sex. 
    We use the AUROC (Area Under the Receiver Operating Characteristic Curve) to measure discriminative ability.
    """)

    # Data Loading
    with st.spinner("Preparing Data..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

    # Filter Logic
    st.markdown("#### 1. Population Filter (Optional)")
    enable_filter = st.checkbox("Filter Patient Population?")
    df_model_input = df_processed.copy()
    
    if enable_filter:
        numeric_options = ['age', 'weight_admission', 'height', 'lab_creatinine']
        filter_col = st.selectbox("Select Feature:", numeric_options)
        min_val, max_val = float(df_processed[filter_col].min()), float(df_processed[filter_col].max())
        range_vals = st.slider(f"Range for {filter_col}:", min_val, max_val, (min_val, max_val))
        df_model_input = df_processed[(df_processed[filter_col] >= range_vals[0]) & (df_processed[filter_col] <= range_vals[1])]

    # Variable Selection
    st.markdown("#### 2. Model Features")
    all_vars = ['age', 'height','weight_admission', 'lab_bun','lab_creatinine','lab_sodium',
                'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride','lab_platelets','lab_rbc','lab_calcium','lab_mcv',
                'lab_mchc','lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']
    selected_predictors = st.multiselect("Choose predictors:", all_vars, default=all_vars)

    if st.button("Train & Evaluate Models"):
        if not selected_predictors:
            st.error("Select at least one predictor.")
            st.stop()
            
        df_male = df_model_input[df_model_input['gender'] == "Male"]
        df_female = df_model_input[df_model_input['gender'] == "Female"]
        
        # Train/Test Split
        train_dat_female, test_dat_female = train_test_split(df_female, test_size=0.2, random_state=2025)
        train_dat_male, test_dat_male = train_test_split(df_male, test_size=0.2, random_state=2025)
        train_dat, test_dat = train_test_split(df_model_input, test_size=0.2, random_state=2025)

        vals_male, vals_female, vals_full = [], [], []

        try:
            # Male
            reg_male = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_male).fit(disp=0)
            vals_male.append(roc_auc_score(test_dat_male['in_hospital_mortality'], reg_male.predict(test_dat_male[selected_predictors])))
            
            # Female
            reg_female = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_female).fit(disp=0)
            vals_female.append(roc_auc_score(test_dat_female['in_hospital_mortality'], reg_female.predict(test_dat_female[selected_predictors])))
            
            # Full
            reg = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat).fit(disp=0)
            vals_full.append(roc_auc_score(test_dat['in_hospital_mortality'], reg.predict(test_dat[selected_predictors])))
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            r = np.arange(1)
            width = 0.25
            
            plt.bar(r, vals_female, width, label='Female', color='tab:orange')
            plt.bar(r + width, vals_male, width, label='Male', color='tab:blue')
            plt.bar(r + 2*width, vals_full, width, label='All Cohort', color='tab:green')
            
            plt.ylabel('AUROC')
            plt.title('Multivariate Analysis Performance')
            plt.xticks([])
            plt.ylim(0.5, 1.0)
            plt.legend(loc='lower right')
            
            # Add labels
            plt.text(r - 0.05, vals_female[0] + 0.01, f"{vals_female[0]:.3f}")
            plt.text(r + width - 0.05, vals_male[0] + 0.01, f"{vals_male[0]:.3f}")
            plt.text(r + 2*width - 0.05, vals_full[0] + 0.01, f"{vals_full[0]:.3f}")
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error training model: {e}")

    # --- QUESTION 2 ---
    st.divider()
    st.subheader("Question 2")
    st.markdown("When comparing the outcomes of these sex-specific models, what are we most interested in identifying?")
    
    q2_options = {
        "A": "The variables with the highest missing values",
        "B": "The computational time for each model",
        "C": "Any performance gaps or patterns that differ between sexes",
        "D": "Whether the models have the same coefficients"
    }
    
    q2_choice = st.radio("Select Answer:", list(q2_options.keys()), format_func=lambda x: f"{x}) {q2_options[x]}", key="q2")
    
    if st.button("Submit Question 2"):
        if q2_choice == "C":
            st.success("Correct! Our main goal is to see if the model performs differently across the two groups. Are predictions more accurate for one sex than the other? These performance differences can point to underlying biological, clinical, or systemic factors.")
        else:
            st.error("Try again.")