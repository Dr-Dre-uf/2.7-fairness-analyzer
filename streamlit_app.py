import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="Module 2 Microskill 7", layout="wide")

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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Activity:", 
    ["1. Data Inspection & Stats",
     "2. Data Preprocessing", 
     "3. Gender Bias Analysis",
     "4. Univariate Analysis",
     "5. Multivariate Analysis"])

st.sidebar.info("Use the menu above to navigate through the interactive programs.")

# --- 3. PAGE LOGIC ---

# === ACTIVITY 1: DATA INSPECTION & STATS ===
if page == "1. Data Inspection & Stats":
    st.title("Activity 1: Data Inspection & Stats")
    
    # *** ADDED INSTRUCTION HERE ***
    st.info("👈 **Navigation Tip:** Open the sidebar on the left to access other activities (Data Processing, Gender Bias Analysis, Univariate & Multivariate Analysis).")
    
    with st.spinner("Downloading and Loading Data..."):
        df_raw = build_eicu_data()
        df_cleaned = df_raw.drop(columns=['patient_id', 'hospital_id', 'admission_id', 'admission_year', 'weight_discharge', 'discharge_location'])

    st.markdown("Analyze mortality rates across different demographic groups with optional range filtering.")
    
    # Range Filtering Logic
    col_grp, col_filt = st.columns([1, 2])
    
    with col_grp:
        group_option = st.selectbox(
            "Select Group to Analyze:",
            ['ethnicity', 'gender', 'hospital_region', 'admission_source'],
            help="Choose a demographic or hospital characteristic to split the patient data by. The table below will update to show mortality rates for each subgroup."
        )

    df_analysis = df_cleaned.copy() 

    with col_filt:
        enable_filter = st.toggle("Filter by Numerical Range?", help="Toggle this on to restrict the analysis to a specific subset of patients (e.g., only older patients).")
        
        if enable_filter:
            numeric_options = ['age', 'weight_admission', 'height', 'lab_creatinine', 'lab_wbc', 'lab_glucose']
            filter_col = st.selectbox("Select Feature for Range:", numeric_options, help="Choose the numerical variable you want to use as a filter.")
            
            df_analysis[filter_col] = pd.to_numeric(df_analysis[filter_col], errors='coerce')
            min_val = float(df_analysis[filter_col].min())
            max_val = float(df_analysis[filter_col].max())
            
            if pd.isna(min_val) or pd.isna(max_val):
                st.warning("Selected column has no valid data.")
            else:
                range_vals = st.slider(
                    f"Select {filter_col} Range:",
                    min_value=min_val, max_value=max_val, value=(min_val, max_val),
                    help=f"Drag the sliders to exclude patients outside this {filter_col} range."
                )
                df_analysis = df_analysis[
                    (df_analysis[filter_col] >= range_vals[0]) & 
                    (df_analysis[filter_col] <= range_vals[1])
                ]
                st.caption(f"Showing {len(df_analysis)} patients in this range.")

    st.divider()

    if group_option:
        stats = df_analysis.groupby(group_option)['in_hospital_mortality'].agg(['count', 'mean']).reset_index()
        stats.columns = [group_option.capitalize(), 'Patient Count', 'Mortality Rate']
        stats['Mortality Rate Display'] = stats['Mortality Rate'].apply(lambda x: f"{x:.2%}")
        stats = stats.sort_values('Patient Count', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Statistics by {group_option.capitalize()}")
            st.dataframe(stats[[group_option.capitalize(), 'Patient Count', 'Mortality Rate Display']], hide_index=True)
        with col2:
            st.subheader(f"Mortality Rate Comparison")
            st.bar_chart(stats.set_index(group_option.capitalize())['Mortality Rate'])

# === ACTIVITY 2: DATA PROCESSING ===
elif page == "2. Data Preprocessing":
    st.title("Activity 2: Data Processing")
    
    if 'proc_run' not in st.session_state:
        st.session_state.proc_run = False
        
    with st.spinner("Preparing raw data..."):
        df_raw = build_eicu_data()
    
    st.write("### Preprocessing Workflow")
    st.markdown("""
    This module will transform the raw clinical data into a machine-learning ready format:
    1. **Drop IDs**: Remove identifiers like `patient_id` and `admission_id`.
    2. **Impute Missing**: Fill missing numerical values with the mean.
    3. **One-Hot Encoding**: Convert categorical variables (e.g., Ethnicity) into binary columns.
    """)
    
    if st.button("Run Fast Preprocessing", help="Click to execute the cleaning pipeline. This replaces missing data and converts text labels into numbers."):
        st.session_state.proc_run = True

    if st.session_state.proc_run:
        with st.spinner("Processing..."):
            df_processed = get_processed_data(df_raw)
            
        st.success("Preprocessing Complete!")
        st.subheader("Processed Data Preview")
        st.dataframe(df_processed.head())
        
        csv = df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Processed CSV", 
            csv, 
            "eicu_processed.csv", 
            "text/csv", 
            key='download-csv-processed',
            help="Download the cleaned dataset to your computer for further analysis."
        )

# === ACTIVITY 3: GENDER BIAS ANALYSIS ===
elif page == "3. Gender Bias Analysis":
    st.title("Activity 3: Gender Bias Analysis")
    st.markdown("Comparing mean lab values between Male and Female patients, split by mortality status.")

    with st.spinner("Preparing Data..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)

    available_vars = ['age', 'height','weight_admission', 'lab_bun','lab_creatinine','lab_sodium',
                      'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride',
                      'lab_platelets','lab_rbc','lab_calcium','lab_mcv','lab_mchc',
                      'lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']
    
    selected_variables = st.multiselect(
        "Select Variables to Compare:", 
        available_vars, 
        default=['lab_bun','lab_creatinine','lab_sodium'],
        help="Select one or more clinical variables. The chart below will show the average value of these variables for men vs. women, separated by whether they survived."
    )

    if not selected_variables:
        st.warning("Please select at least one variable.")
        st.stop()

    if 'gender' not in df_processed.columns:
        df_processed['gender'] = df_raw['gender']

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

    st.write("---")
    colors = plt.cm.Paired.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    (df0['Female'] + df0['Male']).plot(kind='barh', color=[colors[3], colors[2]], rot=0, ax=ax)
    df0['Male'].plot(kind='barh', color=[colors[5], colors[4]], rot=0, ax=ax)

    ax.legend([f'{val} ({context})' for val, context in df0.columns])
    ax.set_xlabel("Mean Value")
    ax.set_title("Mean Values by Mortality Status (Stacked by Sex)")

    st.pyplot(fig)

# === ACTIVITY 4: UNIVARIATE ANALYSIS ===
elif page == "4. Univariate Analysis":
    st.title("Activity 4: Univariate Analysis (Odds Ratios)")
    st.markdown("Fits a Logistic Regression for each variable individually to determine its Odds Ratio (OR).")

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
    
    selected_variables = st.multiselect(
        "Select Variables to Model:",
        all_vars,
        default=all_vars[:10],
        help="Choose the variables to analyze. The app will build a separate logistic regression model for each variable selected to see how strongly it predicts mortality."
    )

    if st.button("Run Regression Models", help="Click to calculate Odds Ratios for the selected variables."):
        if not selected_variables:
            st.warning("Please select at least one variable.")
            st.stop()

        vals_male, vals_female, vals_full = [], [], []
        
        progress_bar = st.progress(0)
        
        for i, univ_analysis in enumerate(selected_variables):
            progress_bar.progress((i + 1) / len(selected_variables))
            
            # Male
            reg_male = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_male).fit(disp=0)
            conf_male = np.exp(reg_male.conf_int())
            conf_male['OR'] = np.exp(reg_male.params)
            vals_male.append(conf_male.loc[str(univ_analysis), 'OR'])
            
            # Female
            reg_female = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_female).fit(disp=0)
            conf_female = np.exp(reg_female.conf_int())
            conf_female['OR'] = np.exp(reg_female.params)
            vals_female.append(conf_female.loc[str(univ_analysis), 'OR'])
            
            # Full
            reg_full = smf.logit(f'in_hospital_mortality ~ {univ_analysis}', data=train_dat_full).fit(disp=0)
            conf_full = np.exp(reg_full.conf_int())
            conf_full['OR'] = np.exp(reg_full.params)
            vals_full.append(conf_full.loc[str(univ_analysis), 'OR'])

        progress_bar.empty()

        st.subheader("Odds Ratios by Gender")
        bar_width = 0.25
        r1 = np.arange(len(selected_variables))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        fig, ax = plt.subplots(figsize=(12, len(selected_variables) * 0.6 + 2))
        ax.barh(r1, vals_female, bar_width, label='Female', color='tab:orange', alpha=0.8)
        ax.barh(r2, vals_male, bar_width, label='Male', color='tab:blue', alpha=0.8)
        ax.barh(r3, vals_full, bar_width, label='All Cohort', color='tab:green', alpha=0.8)
        
        ax.set_xlabel('Odds Ratio (OR)')
        ax.set_yticks([r + bar_width for r in range(len(selected_variables))])
        ax.set_yticklabels(selected_variables)
        ax.set_title('Univariate Analysis: Predictors of Mortality')
        ax.legend()
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)

        st.pyplot(fig)

# === ACTIVITY 5: MULTIVARIATE ANALYSIS ===
elif page == "5. Multivariate Analysis":
    st.title("Activity 5: Multivariate Analysis (AUROC)")
    st.markdown("""
    This model considers **multiple variables simultaneously** to predict mortality. 
    We compare the Area Under the Receiver Operating Characteristic (AUROC) curve.
    """)

    # 1. Load Data
    with st.spinner("Preparing Data..."):
        df_raw = build_eicu_data()
        df_processed = get_processed_data(df_raw)
        
        # Ensure gender exists
        if 'gender' not in df_processed.columns:
            df_processed['gender'] = df_raw['gender']

    # 2. Interactive Range Filter
    st.markdown("### 1. Filter Data Population")
    st.caption("Restrict the model to a specific patient range (e.g., Age > 60).")
    
    numeric_options = ['age', 'weight_admission', 'height', 'lab_creatinine', 'lab_wbc', 'lab_glucose']
    filter_col = st.selectbox(
        "Select Feature to Filter By:", 
        numeric_options,
        help="Select a variable to limit the patient population. For example, select 'age' to only study older patients."
    )
    
    # Calculate bounds
    min_val = float(df_processed[filter_col].min())
    max_val = float(df_processed[filter_col].max())
    
    # Range Slider
    range_vals = st.slider(
        f"Select {filter_col} Range:",
        min_value=min_val, max_value=max_val, value=(min_val, max_val),
        help=f"Adjust the sliders to exclude patients outside the desired {filter_col} range."
    )
    
    # Apply Filter
    df_model_input = df_processed[
        (df_processed[filter_col] >= range_vals[0]) & 
        (df_processed[filter_col] <= range_vals[1])
    ]
    
    st.write(f"**Patients in selected range:** {len(df_model_input)} (Original: {len(df_processed)})")

    # 3. Interactive Variable Selection
    st.markdown("### 2. Select Model Features")
    all_vars = ['age', 'height','weight_admission', 'lab_bun','lab_creatinine','lab_sodium',
                'lab_hct','lab_wbc','lab_glucose','lab_potassium','lab_hgb','lab_chloride','lab_platelets','lab_rbc','lab_calcium','lab_mcv',
                'lab_mchc','lab_bicarbonate','lab_mch','lab_rdw','lab_albumin']
    
    selected_predictors = st.multiselect(
        "Choose predictors for the multivariate model:",
        all_vars,
        default=all_vars,
        help="Choose which variables the model should use to predict mortality. All selected variables will be used in a single logistic regression model."
    )

    # 4. Run Analysis
    if st.button("Train Multivariate Models", help="Click to train the models and compare their accuracy (AUROC) for men vs. women."):
        if not selected_predictors:
            st.error("Please select at least one predictor.")
            st.stop()
            
        if len(df_model_input) < 100:
            st.error("Too few patients remaining after filtering. Please widen the range.")
            st.stop()

        # Split Data (Based on filtered input)
        df_male = df_model_input[df_model_input['gender'] == "Male"]
        df_female = df_model_input[df_model_input['gender'] == "Female"]

        train_dat_female, test_dat_female = train_test_split(df_female, test_size=0.2, random_state=2025)
        train_dat_male, test_dat_male = train_test_split(df_male, test_size=0.2, random_state=2025)
        train_dat, test_dat = train_test_split(df_model_input, test_size=0.2, random_state=2025)

        vals_male = []
        vals_female = []
        vals_full = []
        
        with st.spinner("Training Models..."):
            try:
                # --- Male Model ---
                reg_male = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_male).fit(disp=0)
                roc_male = roc_auc_score(test_dat_male['in_hospital_mortality'], reg_male.predict(test_dat_male[selected_predictors]))
                vals_male.append(roc_male)

                # --- Female Model ---
                reg_female = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat_female).fit(disp=0)
                roc_female = roc_auc_score(test_dat_female['in_hospital_mortality'], reg_female.predict(test_dat_female[selected_predictors]))
                vals_female.append(roc_female)

                # --- Full Model ---
                reg = smf.logit(f"in_hospital_mortality ~ " + " + ".join(selected_predictors), data=train_dat).fit(disp=0)
                roc_full = roc_auc_score(test_dat['in_hospital_mortality'], reg.predict(test_dat[selected_predictors]))
                vals_full.append(roc_full)
            except Exception as e:
                st.error(f"Model failed to converge. Try selecting fewer variables or more patients. Error: {e}")
                st.stop()

        # 5. Plotting
        st.subheader("Model Performance (AUROC)")
        
        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        
        r1 = np.arange(len(vals_full))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        plt.bar(r1, vals_female, bar_width, label='Female', color='tab:orange', capsize=5)
        plt.bar(r2, vals_male, bar_width, label='Male', color='tab:blue', capsize=5)
        plt.bar(r3, vals_full, bar_width, label='All cohort', color='tab:green', capsize=5)

        plt.ylabel('AUROC')
        plt.title('Multivariate Analysis Performance')
        plt.xticks([r + bar_width for r in range(len(vals_full))], ['Logistic Regression'])
        plt.ylim(0.5, 1.0)
        plt.legend(loc='lower right')
        
        for i, v in enumerate(vals_female):
            plt.text(r1[i] - 0.05, v + 0.01, f"{v:.3f}")
        for i, v in enumerate(vals_male):
            plt.text(r2[i] - 0.05, v + 0.01, f"{v:.3f}")
        for i, v in enumerate(vals_full):
            plt.text(r3[i] - 0.05, v + 0.01, f"{v:.3f}")

        st.pyplot(fig)