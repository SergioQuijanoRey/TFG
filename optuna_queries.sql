-- SQL queries to execute on SQLITE3 database that optuna uses to store information

-- Sometimes hp tuning almost cannot produce combinations of parameters that lead
-- to succesful trainings. So it is useful to query for all succesful trainings,
-- so we can detect some problems
SELECT * FROM trial_values;

-- And sometimes optuna can't produce any succesful train. So this query will
-- return all trials (useful for debuggin)
SELECT * FROM trials;

-- Get the info of the best trial value (trial value is the float that we are
-- optimizing)
SELECT *  FROM trial_values WHERE value == (SELECT MAX(value) FROM trial_values);

-- Get the parameters of the best trial
SELECT param_name, param_value, distribution_json
FROM trial_params
WHERE trial_id == (
    SELECT trial_id  FROM trial_values WHERE value == (SELECT MAX(value) FROM trial_values)
);

