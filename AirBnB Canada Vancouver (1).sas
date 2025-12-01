/* ================================
   AIRBNB PRICE PREDICTION ANALYSIS
   Complete Analysis with All Requirements
   ================================ */

/* ================================
   0) Setup
   ================================ */
ods graphics on;
options nonotes;                         
%let seed = 123;
%let path = /home/u64128363/DSCI _519/Exam;   

/* Utility: RMSE from a dataset with columns: actual=, pred= */
%macro rmse(data=, actual=, pred=, outds=);
  %if not %sysfunc(exist(&data)) %then %do;
    %put ERROR: Dataset &data does not exist;
    %return;
  %end;
  
  proc sql noprint;
    select sqrt(mean(((&pred)-(&actual))**2)) format=8.4 into :_rmse_
    from &data;
  quit;
  %if %superq(outds) ne %then %do;
    data &outds; 
      length model $32 metric $8 value 8.;
      model="&_model_"; 
      metric="RMSE"; 
      value=&_rmse_;
    run;
  %end;
  %put NOTE: RMSE(&data) = &_rmse_;
%mend;

/* ================================
   1) Import & Initial Exploration
   ================================ */
proc import datafile="&path./listings.xlsx"
    out=work.airbnb_raw dbms=xlsx replace;
    getnames=yes;
run;

proc contents data=work.airbnb_raw order=varnum; 
    title "Data Structure - All Variables";
run;

/* Initial data overview */
proc means data=work.airbnb_raw n nmiss min max mean;
    title "Initial Data Overview - Key Variables";
run;

/* Geographic visualization */
ods graphics / reset width=6.4in height=5in;

proc sgmap plotdata=work.airbnb_raw;
    openstreetmap;
    scatter x=longitude y=latitude / 
        group=neighbourhood_cleansed
        markerattrs=(symbol=circlefilled size=3)
        transparency=0.3;
    title "Airbnb Listings Geographic Distribution";
run;

ods graphics / reset;

/* ================================
   2) Dependent Variable Investigation
   ================================ */

/* Extract numeric price from currency format */
data work.price_check;
    set work.airbnb_raw;
    length price_char $32;
    price_char = compress(strip(put(price, 32.)), '$,');
    price_num = input(price_char, best32.);
    if price_num > 0;
    keep id price price_num;
run;

/* Initial distribution analysis */
proc univariate data=work.price_check plot;
    var price_num;
    histogram price_num / normal(mu=est sigma=est);
    inset n mean std min max / position=ne;
    title "Price Distribution - Before Transformation";
    title2 "Checking for Skewness and Outliers";
run;

/* ================================
   3) Categorical Variable Analysis (BEFORE Collapsing)
   ================================ */

title "Property Type - Initial Distribution (Before Collapsing)";
proc freq data=work.airbnb_raw order=freq nlevels;
    tables property_type / nocum out=work.prop_freq;
run;

/* Show top categories */
proc print data=work.prop_freq(obs=20);
    title "Top 20 Property Types by Frequency";
    var property_type count percent;
    format percent 6.2;
run;

title "Room Type - Initial Distribution";  
proc freq data=work.airbnb_raw order=freq nlevels;
    tables room_type / nocum;
run;

title "Neighborhood - Initial Distribution";
proc freq data=work.airbnb_raw order=freq nlevels;
    tables neighbourhood_cleansed / nocum out=work.neigh_freq;
run;

/* Show neighborhood counts */
proc print data=work.neigh_freq(obs=15);
    title "Top 15 Neighborhoods by Count";
    var neighbourhood_cleansed count percent;
    format percent 6.2;
run;

/* ================================
   4) Data Cleaning & Feature Engineering
   ================================ */

/* RATIONALE FOR CATEGORICAL COLLAPSING:
   - property_type: Has 50+ levels, many with <10 observations
   - Action: Collapse to top 9 most frequent + "Other" 
   - neighbourhood_cleansed: High cardinality (20+ neighborhoods)
   - Action: Excluded (geographic info captured by lat/long)
*/

data work.airbnb;
    set work.airbnb_raw;

    /* --- PRICE: '$1,234.00' -> 1234; log transform --- */
    length price_char $32;
    price_char = compress(strip(put(price, 32.)), '$,');
    price_num = input(price_char, best32.);
    if price_num > 0 then log_price = log(price_num);

    /* --- BATHROOMS: '1.5 shared baths' -> 1.5 --- */
    if not missing(bathrooms_text) then 
        bathrooms_num = input(scan(strip(bathrooms_text), 1, ' '), best32.);
    else bathrooms_num = .;

    /* --- BOOLEAN CONVERSIONS: t/f -> 1/0 --- */
    has_avail = (strip(has_availability) = 't');
    instant = (strip(instant_bookable) = 't');

    /* --- PERCENTAGE CONVERSIONS: '92%' -> 0.92 --- */
    if not missing(host_response_rate) then 
        host_resp_rate = input(compress(host_response_rate, '%'), best32.) / 100;
    else host_resp_rate = .;
    
    if not missing(host_acceptance_rate) then 
        host_acc_rate = input(compress(host_acceptance_rate, '%'), best32.) / 100;
    else host_acc_rate = .;

    /* --- SIMPLE NUMERIC ASSIGNMENTS (variables are already numeric) --- */
    accommodates_n = accommodates;
    bedrooms_n = bedrooms;
    beds_n = beds;
    number_of_reviews_n = number_of_reviews;
    number_of_reviews_ltm_n = number_of_reviews_ltm;
    number_of_reviews_l30d_n = number_of_reviews_l30d;
    reviews_per_month_n = reviews_per_month;

    review_scores_rating_n = review_scores_rating;
    review_scores_accuracy_n = review_scores_accuracy;
    review_scores_cleanliness_n = review_scores_cleanliness;
    review_scores_checkin_n = review_scores_checkin;
    review_scores_communication_n = review_scores_communication;
    review_scores_location_n = review_scores_location;
    review_scores_value_n = review_scores_value;

    availability_30_n = availability_30;
    availability_60_n = availability_60;
    availability_90_n = availability_90;
    availability_365_n = availability_365;
    minimum_nights_n = minimum_nights;
    maximum_nights_n = maximum_nights;

    /* --- COLLAPSE PROPERTY_TYPE to manageable categories --- */
    length prop_grp $40;
    prop_grp = 'Other';
    if property_type = 'Entire rental unit' then prop_grp = 'Entire rental unit';
    else if property_type = 'Entire home' then prop_grp = 'Entire home';
    else if property_type = 'Private room in home' then prop_grp = 'Private room in home';
    else if property_type = 'Entire condo' then prop_grp = 'Entire condo';
    else if property_type = 'Entire guest suite' then prop_grp = 'Entire guest suite';
    else if property_type = 'Private room in rental unit' then prop_grp = 'Private room in rental unit';
    else if property_type = 'Entire guesthouse' then prop_grp = 'Entire guesthouse';
    else if property_type = 'Entire loft' then prop_grp = 'Entire loft';
    else if property_type = 'Entire townhouse' then prop_grp = 'Entire townhouse';

    /* --- ROOM TYPE DUMMIES (keep one out as reference) --- */
    room_entire = (room_type = 'Entire home/apt');
    room_priv = (room_type = 'Private room');
    room_shared = (room_type = 'Shared room');
    room_hotel = (room_type = 'Hotel room');

    /* --- PROPERTY GROUP DUMMIES ('Other' is reference) --- */
    prop_ent_rent_unit = (prop_grp = 'Entire rental unit');
    prop_ent_home = (prop_grp = 'Entire home');
    prop_priv_home = (prop_grp = 'Private room in home');
    prop_ent_condo = (prop_grp = 'Entire condo');
    prop_ent_guest_st = (prop_grp = 'Entire guest suite');
    prop_priv_rent_un = (prop_grp = 'Private room in rental unit');
    prop_ent_guesthouse = (prop_grp = 'Entire guesthouse');
    prop_ent_loft = (prop_grp = 'Entire loft');
    prop_ent_townhouse = (prop_grp = 'Entire townhouse');

    /* Keep only modeling fields */
    keep id price_num log_price bathrooms_num
         accommodates_n bedrooms_n beds_n
         number_of_reviews_n number_of_reviews_ltm_n number_of_reviews_l30d_n
         reviews_per_month_n
         review_scores_rating_n review_scores_accuracy_n review_scores_cleanliness_n
         review_scores_checkin_n review_scores_communication_n review_scores_location_n
         review_scores_value_n
         minimum_nights_n maximum_nights_n
         availability_30_n availability_60_n availability_90_n availability_365_n
         has_avail instant host_resp_rate host_acc_rate
         room_entire room_priv room_shared room_hotel
         prop_ent_rent_unit prop_ent_home prop_priv_home prop_ent_condo
         prop_ent_guest_st prop_priv_rent_un prop_ent_guesthouse
         prop_ent_loft prop_ent_townhouse;
run;

/* ================================
   5) Post-Collapsing Verification
   ================================ */

title "Property Groups After Collapsing - Verify Balanced Counts";
proc freq data=work.airbnb order=freq;
    tables prop_ent_rent_unit prop_ent_home prop_priv_home prop_ent_condo
           prop_ent_guest_st prop_priv_rent_un prop_ent_guesthouse
           prop_ent_loft prop_ent_townhouse / nocum;
run;

title "Room Type Distribution - Final";
proc freq data=work.airbnb order=freq;
    tables room_entire room_priv room_shared room_hotel / nocum;
run;

/* ================================
   6) Check Transformed Price Distribution
   ================================ */

proc univariate data=work.airbnb plot;
    var log_price;
    histogram log_price / normal(mu=est sigma=est);
    inset n mean std skewness kurtosis / position=ne;
    title "Log-Transformed Price Distribution";
    title2 "Checking Normality After Transformation";
run;

/* ================================
   7) Outlier Removal Using Percentiles
   ================================ */

proc univariate data=work.airbnb noprint;
    var price_num;
    output out=work.price_bounds pctlpts=1 99 pctlpre=price_;
run;

data _null_;
    set work.price_bounds;
    call symputx('lower_bound', price_1);
    call symputx('upper_bound', price_99);
run;

%put NOTE: Removing outliers outside 1st-99th percentile range;
%put NOTE: Price bounds: $&lower_bound to $&upper_bound;

data work.airbnb_clean;
    set work.airbnb;
    if missing(log_price) then delete;
    /* Remove extreme outliers */
    if price_num >= &lower_bound and price_num <= &upper_bound then output;
run;

/* Verify outlier removal */
proc univariate data=work.airbnb_clean plot;
    var log_price;
    histogram log_price / normal(mu=est sigma=est);
    inset n mean std skewness kurtosis / position=ne;
    title "Final Log Price Distribution - After Outlier Removal";
run;

/* ================================
   8) Missing Value Analysis
   ================================ */

proc means data=work.airbnb_clean n nmiss;
    var accommodates_n bedrooms_n beds_n bathrooms_num 
        review_scores_rating_n host_resp_rate host_acc_rate;
    title "Missing Value Analysis - Key Predictors";
run;

/* ================================
   9) Predictor Investigation - Multicollinearity
   ================================ */

%let interval_vars = accommodates_n bedrooms_n beds_n bathrooms_num 
                     number_of_reviews_n reviews_per_month_n 
                     review_scores_rating_n review_scores_accuracy_n 
                     review_scores_cleanliness_n review_scores_checkin_n 
                     review_scores_communication_n review_scores_location_n 
                     review_scores_value_n minimum_nights_n maximum_nights_n 
                     availability_30_n availability_60_n availability_90_n 
                     availability_365_n host_resp_rate host_acc_rate;

%let nominal_vars = room_entire room_priv room_shared 
                    prop_ent_rent_unit prop_ent_home prop_priv_home 
                    prop_ent_condo prop_ent_guest_st prop_priv_rent_un 
                    prop_ent_guesthouse prop_ent_loft prop_ent_townhouse 
                    has_avail instant;

%let full_predictors = &interval_vars &nominal_vars;

/* Correlation with target variable */
ods select PearsonCorr;
proc corr data=work.airbnb_clean nosimple noprob plots=none;
    var &interval_vars log_price;
    /*with log_price;*/
    title "Correlation of Predictors with Log Price";
    title2 "Identifying Important Predictors";
run;
ods select all;


/* Define reduced variable set after multicollinearity analysis */
%let interval_vars_reduced = accommodates_n bathrooms_num 
                     number_of_reviews_n reviews_per_month_n 
                     review_scores_rating_n review_scores_cleanliness_n 
                     review_scores_location_n minimum_nights_n maximum_nights_n 
                     availability_30_n availability_365_n host_resp_rate host_acc_rate;

%let full_predictors_reduced = &interval_vars_reduced &nominal_vars;



/* Check for multicollinearity among FINAL predictors */
title "Correlation Matrix - Checking for Multicollinearity";
title2 "High correlations (>0.7) indicate potential issues";
proc corr data=work.airbnb_clean plots=matrix(histogram);
    var &interval_vars_reduced;
run;


/* ================================
   10) Train/Test Split (80/20)
   ================================ */

proc surveyselect data=work.airbnb_clean method=srs samprate=0.8 seed=&seed
                  out=work.split outall;
run;

data work.train work.test;
    set work.split;
    if selected then output work.train; 
    else output work.test;
    drop selected;
run;

/* Verify split */
proc sql; 
    select 'Training' as dataset, count(*) as n from work.train
    union all
    select 'Testing' as dataset, count(*) as n from work.test;
quit;

/* ================================
   11) Multiple Linear Regression with VIF
   ================================ */

title "Multiple Linear Regression - Full Model";
title2 "Checking VIF for Multicollinearity";
proc reg data=work.train plots(only)=(residualhistogram residualplot qqplot);
    model log_price = &full_predictors_reduced / vif;
    output out=work.reg_train p=pred_lr r=resid_lr;
    store work.reg_model;
run; 
quit;


/* Calculate Train RMSE */
%let _model_=Linear_Regression_Train;
%rmse(data=work.reg_train, actual=log_price, pred=pred_lr, outds=work.rmse1);

/* Score test data */
proc plm restore=work.reg_model;
    score data=work.test out=work.reg_test predicted=pred_lr;
run;

/* Calculate Test RMSE */
%let _model_=Linear_Regression_Test;
%rmse(data=work.reg_test, actual=log_price, pred=pred_lr, outds=work.rmse2);





/* ================================
   12) LASSO Regression (Variable Selection)
   ================================ */
  
title "LASSO Regression - Regularized Variable Selection";
title2 "Automatic feature selection using L1 penalty";
proc glmselect data=work.train plots(unpack)=all;
    model log_price = &full_predictors  
        / selection=lasso(choose=cp stop=20) stats=all;
    output out=work.lasso_train_pred p=pred_lasso;
    store work.lasso_model;  /* Store for later scoring */
run;


/* Calculate Train RMSE */
%let _model_=LASSO_Train;
%rmse(data=work.lasso_train_pred, actual=log_price, pred=pred_lasso, outds=work.rmse3);

/* Score test data using stored model */
proc plm restore=work.lasso_model;
    score data=work.test out=work.lasso_test_pred predicted=pred_lasso;
run;

/* Calculate Test RMSE */
%let _model_=LASSO_Test;
%rmse(data=work.lasso_test_pred, actual=log_price, pred=pred_lasso, outds=work.rmse4);


/* ================================
   13) Decision Tree (HPSPLIT)
   ================================ */

title "Decision Tree Regression";
title2 "Non-linear model with automatic interaction detection";
proc hpsplit data=work.train seed=&seed;
    model log_price = &full_predictors;
    partition fraction(validate=0.3);
    code file="&path./tree_score.sas";
run;



/* Score TRAIN data */
data work.tree_train_pred;
    set work.train;
    %include "&path./tree_score.sas";
    pred_tree = p_log_price;
run;

%let _model_=Decision_Tree_Train;
%rmse(data=work.tree_train_pred, actual=log_price, pred=pred_tree, outds=work.rmse5);

/* Score TEST data */
data work.tree_test_pred;
    set work.test;
    %include "&path./tree_score.sas";
    pred_tree = p_log_price;
run;

%let _model_=Decision_Tree_Test;
%rmse(data=work.tree_test_pred, actual=log_price, pred=pred_tree, outds=work.rmse6);

/* ================================
   14) Random Forest Regression
   ================================ */

title "Random Forest - Ensemble of Decision Trees";
title2 "Reduces overfitting through bootstrap aggregation";
proc hpforest data=work.train seed=&seed maxtrees=100;
    target log_price / level=interval;
    input &interval_vars / level=interval;
    input &nominal_vars / level=nominal;
    ods output variableimportance=work.var_importance;
    save file="&path./rfmodel.bin";
run;


/* Display Variable Importance */
proc sort data=work.var_importance;
    by descending MSEOOB;
run;

title "Top 10 Most Important Variables - Random Forest";
proc print data=work.var_importance(obs=10) noobs;
    var Variable NRules MSEOOB AAEOOB;
    format MSEOOB AAEOOB 8.4;
run;

/* Score TRAIN data */
proc hp4score data=work.train; 
    score file="&path./rfmodel.bin" out=work.rf_train_pred;
    id log_price;
run;

data work.rf_train_pred;
    set work.rf_train_pred;
    pred_rf = P_log_price;
run;

%let _model_=Random_Forest_Train;
%rmse(data=work.rf_train_pred, actual=log_price, pred=pred_rf, outds=work.rmse7);

/* Score TEST data */
proc hp4score data=work.test; 
    score file="&path./rfmodel.bin" out=work.rf_test_pred;
    id log_price;
run;

data work.rf_test_pred;
    set work.rf_test_pred;
    pred_rf = P_log_price;
run;

%let _model_=Random_Forest_Test;
%rmse(data=work.rf_test_pred, actual=log_price, pred=pred_rf, outds=work.rmse8);

/* ================================
   15) Final Model Comparison
   ================================ */

/* Combine all RMSE results */
data work.rmse_all;
    set work.rmse1 work.rmse2 work.rmse3 work.rmse4 
        work.rmse5 work.rmse6 work.rmse7 work.rmse8;
run;

/* Create structured comparison table */
proc sql;
    create table work.model_comparison as
    select 
        case 
            when model like '%Train%' then scan(model,1,'_')
            when model like '%Test%' then scan(model,1,'_')
            else model
        end as Model_Type,
        case 
            when model like '%Train%' then 'Train'
            when model like '%Test%' then 'Test'
            else 'Unknown'
        end as Dataset,
        value as RMSE format=8.4
    from work.rmse_all
    order by Model_Type, Dataset;
quit;

/* Pivot to show Train vs Test side by side */
proc transpose data=work.model_comparison out=work.comparison_wide prefix=RMSE_;
    by Model_Type;
    id Dataset;
    var RMSE;
run;

