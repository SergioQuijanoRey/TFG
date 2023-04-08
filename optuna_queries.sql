-- SQL queries to execute on SQLITE3 database that optuna uses to store information

-- Get the info of the best trial value (trial value is the float that we are
-- optimizing)
SELECT *  FROM trial_values WHERE value == (SELECT MAX(value) FROM trial_values);

-- Get the parameters of the best trial
SELECT param_name, param_value
FROM trial_params
WHERE trial_id == (
    SELECT trial_id  FROM trial_values WHERE value == (SELECT MAX(value) FROM trial_values)
);
